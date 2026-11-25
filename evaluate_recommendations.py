"""Evaluate four-channel recommendations and a popularity baseline on a temporal holdout."""

import argparse
import json
import os
from pathlib import Path
import random
import zlib

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')

import numpy as np
import pandas as pd
from tqdm import tqdm

from build_graph_model import DeepWalk, Node2Vec, get_item_graph
from project_paths import DATA_DIR, ROOT_DIR, find_data_file
from run_pipeline import Recaller, build_associations, rank, read_clicks, read_content, train_ranker
from sklearn.preprocessing import StandardScaler

TOP_K = 50
RECALL_K = 500
HIT_CUTOFFS = (1, 5, 10, 20, TOP_K)
GRAPH_SETTINGS = {'walk_length': 12, 'num_walks': 6, 'dimensions': 64, 'epochs': 3}


def select_examples(history_clicks, target_clicks):
    """Select each returning user's first future click if it is a novel item."""
    histories = history_clicks.groupby('user_id').item_id.agg(list).to_dict()
    first_click = target_clicks.drop_duplicates('user_id').set_index('user_id').item_id
    returning = first_click[first_click.index.isin(histories)]
    examples = [(user, histories[user], item) for user, item in returning.items()
                if item not in histories[user]]
    return examples, {
        'users_in_target_window': len(first_click),
        'excluded_without_history': len(first_click) - len(returning),
        'excluded_repeat_target': len(returning) - len(examples),
        'eligible_users': len(examples),
    }


def stable_hash(value):
    return zlib.crc32(value.encode('utf-8'))


def fit_graph_vectors(clicks, seed):
    """Train graph embeddings using only the earliest click window."""
    random.seed(seed)
    np.random.seed(seed)
    config = GRAPH_SETTINGS
    vectors = []
    for algorithm, directed, options in (
        (DeepWalk, False, {}),
        (Node2Vec, True, {'p': 2, 'q': 0.5}),
    ):
        print(f'从历史窗口训练 {algorithm.__name__}', flush=True)
        graph = get_item_graph(clicks, 'user_id', 'item_id', direction=directed)
        embedding = algorithm(graph, walk_length=config['walk_length'],
                              num_walks=config['num_walks'], workers=1, **options)
        model = embedding.train(embed_size=config['dimensions'], window_size=5, workers=1,
                                iters=config['epochs'], seed=seed, hashfxn=stable_hash)
        model.wv.fill_norms()
        vectors.append(model.wv)
    return vectors


def fit_ranker(recall, examples, epochs, seed):
    rng = np.random.default_rng(seed)
    batches, targets = [], []
    for _, history, target in tqdm(examples, desc='构造排序训练样本'):
        candidates, channels = recall.recall_for(history)
        if target not in candidates:
            continue
        negatives = [item for item in candidates if item != target]
        selected = [target] + rng.choice(negatives, min(len(negatives), 20), replace=False).tolist()
        batches.append(recall.features(history, selected, channels))
        targets.extend([1] + [0] * (len(selected) - 1))
    if not batches:
        raise ValueError('排序训练窗口中没有被召回的正样本')
    values = np.concatenate(batches)
    scaler = StandardScaler().fit(values)
    model = train_ranker(scaler.transform(values).astype(np.float32),
                         np.asarray(targets, dtype=np.float32), epochs, seed)
    return model, scaler, len(batches), len(targets)


def target_rank(items, target):
    """Use zero for a target outside the ranked list."""
    return items.index(target) + 1 if target in items else 0


def evaluate_users(examples, recall, model, scaler):
    # Shared item catalog, with the baseline ordered only by earlier popularity.
    popular = sorted(recall.catalog, key=lambda item: (-recall.counts[item], item))
    rows = []
    for user, history, target in tqdm(examples, desc='评估同一批留出用户'):
        seen = set(history)
        baseline = []
        for item in popular:
            if item not in seen:
                baseline.append(item)
            if len(baseline) == RECALL_K:
                break
        ranked, candidates = rank(model, scaler, recall, history)
        baseline_rank = target_rank(baseline, target)
        rows.append({
            'user_id': user, 'target_item_id': target,
            'target_in_catalog': target in recall.allowed,
            'model_recalled': target in candidates,
            'popularity_recalled': baseline_rank > 0,
            'model_rank_at_50': target_rank(ranked, target),
            'popularity_rank_at_50': baseline_rank if baseline_rank <= TOP_K else 0,
        })
    return pd.DataFrame(rows)


def summarize(results, seed):
    summary = {}
    scores = []
    for method in ('model', 'popularity'):
        ranks = results[f'{method}_rank_at_50'].to_numpy()
        hits = ranks > 0
        ndcg = np.zeros(len(ranks))
        ndcg[hits] = 1 / np.log2(ranks[hits] + 1)
        scores.append(np.column_stack((hits, ndcg)))
        summary[method] = {
            'recall_at_500': float(results[f'{method}_recalled'].mean()),
            'hit_rate_at_50': float(hits.mean()),
            'hits_at_50': int(hits.sum()),
            'ndcg_at_50': float(ndcg.mean()),
            'hit_rate_by_k': {str(k): float((hits & (ranks <= k)).mean()) for k in HIT_CUTOFFS},
        }
    differences = scores[0] - scores[1]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(40):
        indices = rng.integers(len(results), size=(50, len(results)))
        estimates.append(differences[indices].mean(axis=1))
    limits = np.quantile(np.concatenate(estimates), (0.025, 0.975), axis=0)
    summary['paired_bootstrap'] = {
        'resamples': 2000,
        'hit_rate_at_50_difference': {'estimate': float(differences[:, 0].mean()),
                                      'ci_95': limits[:, 0].tolist()},
        'ndcg_at_50_difference': {'estimate': float(differences[:, 1].mean()),
                                'ci_95': limits[:, 1].tolist()},
    }
    return summary


def run(args):
    click_path = find_data_file(args.data_dir, 'train', f'underexpose_train_click-{args.phase}.csv')
    content_path = find_data_file(args.data_dir, 'train', 'underexpose_item_feat.csv')
    clicks = read_clicks(click_path)
    recall_cutoff, eval_cutoff = clicks.time.quantile([.70, .85])
    if recall_cutoff >= eval_cutoff:
        raise ValueError('数据的时间范围不足以划分三个窗口')
    past = clicks[clicks.time < recall_cutoff]
    rank_window = clicks[(clicks.time >= recall_cutoff) & (clicks.time < eval_cutoff)]
    holdout = clicks[clicks.time >= eval_cutoff]
    before_eval = clicks[clicks.time < eval_cutoff]
    train_examples, train_cohort = select_examples(past, rank_window)
    eval_examples, eval_cohort = select_examples(before_eval, holdout)
    if not train_examples or not eval_examples:
        raise ValueError('时间切分后没有足够的回访用户与新商品点击')
    print(f'历史点击 {len(past)} 条，排序训练用户 {len(train_examples)}，留出用户 {len(eval_examples)}',
          flush=True)
    associations, counts = build_associations(past.groupby('user_id').item_id.agg(list).to_dict())
    vectors = fit_graph_vectors(past, args.seed)
    print('商品特征 PCA 降维', flush=True)
    content = read_content(content_path)
    recall = Recaller([*vectors, content], associations, counts, RECALL_K, 50,
                      allowed_items=content.index_to_key)
    recall.prepare_neighbors([history for _, history, _ in train_examples])
    model, scaler, positive_users, sample_count = fit_ranker(recall, train_examples, args.epochs, args.seed)

    recall.prepare_neighbors([history for _, history, _ in eval_examples])
    results = evaluate_users(eval_examples, recall, model, scaler)
    evaluation = {
        'phase': args.phase, 'seed': args.seed, 'click_input': str(click_path),
        'item_feature_input': str(content_path),
        'cutoffs': {'recall_fit_end_exclusive': recall_cutoff, 'evaluation_start_inclusive': eval_cutoff},
        'clicks_after_deduplication': len(clicks),
        'clicks_per_window': {'recall_fit': len(past), 'rank_training': len(rank_window), 'holdout': len(holdout)},
        'rank_training_cohort': train_cohort, 'evaluation_cohort': eval_cohort,
        'target_items_missing_from_catalog': int((~results.target_in_catalog).sum()),
        'rank_training_users_with_recalled_target': positive_users,
        'rank_training_samples': sample_count, 'rank_epochs': args.epochs,
        'candidate_limit': RECALL_K, 'neighbors_per_seed': 50,
        'graph_settings': GRAPH_SETTINGS,
        'static_catalog_assumption': 'Supplied item features are treated as available at both cutoffs.',
        'metrics': summarize(results, args.seed),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / 'per_user.csv', index=False)
    (args.output_dir / 'metrics.json').write_text(json.dumps(evaluation, indent=2) + '\n')
    print(json.dumps(evaluation, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=ROOT_DIR / 'outputs' / 'temporal_evaluation')
    parser.add_argument('--phase', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()
    if args.phase < 0 or args.epochs < 1 or not 0 <= args.seed < 2**32:
        parser.error('phase must be nonnegative, epochs positive and seed a 32-bit unsigned integer')
    run(args)


if __name__ == '__main__':
    main()
