"""Run recall, Wide&Deep training and Top50 prediction from raw CSV files."""
# 流程总览（在run函数中按步骤1～12执行）：
# 原始点击CSV → 留出最后一次点击作为标签 → 构建ItemCF → 加载图向量/商品特征
# → 四路召回 → 生成训练样本 → 标准化 → Wide&Deep训练 → 验证 → 测试用户Top50。
# 召回负责从大商品库中找候选，排序负责判断候选的优先顺序。
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from project_paths import DATA_DIR, ROOT_DIR, find_data_file


# 工具：读取点击流水，返回按时间从早到晚排列的DataFrame。
def read_clicks(path):
    # CSV无表头；ID按字符串读取，time按数值读取，便于排序和比较查询时间。
    frame = pd.read_csv(path, header=None, names=['user_id', 'item_id', 'time'],
                        dtype={'user_id': str, 'item_id': str, 'time': np.float64})
    # 提前拒绝空数据、缺失字段和无穷时间，避免后续训练出现难定位的问题。
    if frame.empty or frame.isna().any().any() or not np.isfinite(frame['time']).all():
        raise ValueError(f'点击数据为空或含无效值：{path}')
    # 去掉完全相同的点击记录；稳定排序保留相同时间记录的原始先后顺序。
    return frame.drop_duplicates().sort_values('time', kind='stable').reset_index(drop=True)


# 工具：把商品的文本/图像特征转成可用于相似商品检索的向量表。
def read_content(path):
    # 原CSV将两个128维向量展开为256列；两端的数字可能带有方括号。
    boundaries = (1, 128, 129, 256)
    frame = pd.read_csv(path, header=None,
        dtype={0: str, **{i: np.float32 for i in range(1, 257) if i not in boundaries}},
        converters={i: lambda x: float(x.strip(' []')) for i in boundaries})
    if frame.shape[1] != 257:
        raise ValueError('商品特征应为 item_id + 128维文本向量 + 128维图像向量，共257列')
    # 每个商品只保留一条特征记录，取出ID以外的256维浮点数。
    frame = frame.drop_duplicates(0)
    vectors = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError('商品特征包含无效数值')
    # PCA将256维压缩到最多50维，减少相似度计算开销；小数据时受样本数限制。
    size = min(50, *vectors.shape)
    vectors = PCA(n_components=size, svd_solver='randomized', random_state=2020).fit_transform(vectors)
    # 统一存成“商品ID → 向量”；与DeepWalk/Node2Vec使用同一种查询接口。
    model = KeyedVectors(vector_size=size)
    model.add_vectors(frame[0].tolist(), vectors)
    # 缓存向量长度，后续计算余弦相似度时无需重复求模长。
    model.fill_norms()
    return model


# 工具：ItemCF——统计同一用户历史中商品之间的关联强弱。
def build_associations(histories):
    # counts是商品点击次数；matrix[A][B]是从商品A关联到B的分数。
    counts = Counter(item for history in histories.values() for item in history)
    matrix = defaultdict(lambda: defaultdict(float))
    mean_length = max(1., np.mean([len(h) for h in histories.values()]))
    for history in tqdm(histories.values(), desc='ItemCF 共现统计'):
        # 两两比较的成本随历史长度平方增长，因此每位用户最多取最近50次点击。
        recent = history[-50:]
        # 对很长的用户历史降权，减少重度用户对共现统计的支配。
        length_factor = 1 + .8 * len(recent) / mean_length
        for i, left in enumerate(recent):
            for j, right in enumerate(recent):
                if left != right:
                    # A先于B时权重为1，反向为0.5；距离越近、A越不热门，关联贡献越大。
                    direction = 1. if i < j else .5
                    matrix[left][right] += direction / (.8 + abs(i-j)) / counts[left] / length_factor
    return {key: dict(value) for key, value in matrix.items()}, counts


# 召回器：统一管理ItemCF、DeepWalk、Node2Vec和商品属性这四路召回。
class Recaller:
    def __init__(self, models, associations, counts, recall, neighbors, allowed_items=None):
        # models按顺序存DeepWalk、Node2Vec、商品属性向量；ItemCF另存为关联字典。
        self.models = models
        self.associations = associations
        self.counts = counts
        # recall是每位用户最多保留的候选数；neighbors是每个历史商品找多少个相似商品。
        self.recall = recall
        self.neighbors = neighbors
        # 开启商品特征时，只允许推荐官方特征文件中出现的商品ID。
        self.allowed = None if allowed_items is None else set(allowed_items)
        # 热门商品优先，用于候选不足时补齐；catalog还包含向量表中的其他合法商品。
        self.popular = [item for item, _ in counts.most_common() if self.allowed is None or item in self.allowed]
        self.catalog = list(dict.fromkeys(self.popular + [item for model in models for item in model.index_to_key if self.allowed is None or item in self.allowed]))
        # 预先归一化向量、缓存相似商品列表，避免对不同用户反复计算同一商品。
        self.cache = {}
        self.normalized = [model.get_normed_vectors() for model in models]
        # 对ItemCF关联分数从高到低排序，每个商品仅保留最相关的neighbors个邻居。
        self.cf_neighbors = {item: sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:neighbors]
                             for item, scores in associations.items()}

    def prepare_neighbors(self, histories):
        # 只预计算用户最近6个历史商品；这些商品是后续召回的种子。
        seeds = sorted({item for history in histories for item in history[-6:]})
        for channel, model in enumerate(self.models):
            known = [item for item in seeds if item in model]
            vectors = self.normalized[channel]
            count = min(self.neighbors, len(model)-1)
            for begin in tqdm(range(0, len(known), 128), desc=f'预计算向量召回 {channel+1}/{len(self.models)}'):
                items = known[begin:begin+128]
                indices = [model.key_to_index[item] for item in items]
                # 每批128个种子与全商品库做矩阵乘法；单位向量点积就是余弦相似度。
                scores = vectors[indices] @ vectors.T
                for row, (item, own_index) in enumerate(zip(items, indices)):
                    # 商品自身相似度通常为1，这里排除自身，避免占用邻居名额。
                    scores[row, own_index] = -np.inf
                # 先快速取最大的若干分数，再对入选部分排序，节省全量排序的开销。
                best = np.argpartition(scores, -count, axis=1)[:, -count:]
                for row, item in enumerate(items):
                    positions = best[row][np.argsort(-scores[row, best[row]], kind='stable')]
                    self.cache[channel, item] = [(model.index_to_key[i], float(scores[row, i])) for i in positions]

    def nearest(self, channel, item):
        # 优先使用缓存；预计算未覆盖时才临时计算。商品不在该向量表中则返回空列表。
        key = (channel, item)
        if key not in self.cache:
            model = self.models[channel]
            if item not in model:
                return []
            count = min(self.neighbors, len(model)-1)
            scores = self.normalized[channel] @ model.get_vector(item, norm=True)
            scores[model.key_to_index[item]] = -np.inf
            positions = np.argpartition(scores, -count)[-count:]
            positions = positions[np.argsort(-scores[positions], kind='stable')]
            self.cache[key] = [(model.index_to_key[i], float(scores[i])) for i in positions]
        return self.cache[key]

    def recall_for(self, history):
        # 输入某位用户已知的点击历史；输出候选商品，以及每路召回各自的原始分数。
        seen = set(history)
        # channels[0]是ItemCF，后面依次是各向量模型；同一商品可被多路同时召回。
        channels = [defaultdict(float) for _ in range(len(self.models) + 1)]
        # 从最近一次点击开始遍历6条历史；时间位置越远，累加分数时的权重越小。
        for position, item in enumerate(reversed(history[-6:])):
            weight = 1 / (1 + .8 * position)
            # 第一路：找历史商品在ItemCF中的邻居，排除已点击和不在目标目录中的商品。
            for candidate, score in self.cf_neighbors.get(item, []):
                if candidate not in seen and (self.allowed is None or candidate in self.allowed):
                    channels[0][candidate] += weight * score
            # 其余各路：分别在DeepWalk、Node2Vec、商品特征空间中找相似商品并累加。
            for channel in range(len(self.models)):
                for candidate, score in self.nearest(channel, item):
                    if candidate not in seen and (self.allowed is None or candidate in self.allowed):
                        channels[channel + 1][candidate] += weight * max(0., score)
        # 用倒数排名融合：每路第rank名贡献1/(60+rank)，避免不同分数量纲直接相加。
        # fused只用于挑选候选；后面的排序特征仍使用channels里的各路原始分数。
        fused = defaultdict(float)
        for scores in channels:
            for rank, (item, _) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)):
                fused[item] += 1. / (60 + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)[:self.recall]
        # 不足recall个时，按热门商品、其余合法目录商品依次补齐，并持续去重、排除已点击。
        selected = set(candidates)
        for item in self.catalog:
            if len(candidates) >= self.recall:
                break
            if item not in seen and item not in selected:
                candidates.append(item)
                selected.add(item)
        if len(candidates) < 50:
            raise ValueError('未点击商品不足50个，无法输出Top50；请提供更完整的数据')
        return candidates, channels

    def features(self, history, candidates, channels):
        # 把“用户历史 + 一个候选商品”转换成一行数值特征，训练和预测共用此函数。
        # 默认四路召回、三个向量模型，共36维，列顺序如下：
        # f_0～f_3：ItemCF、DeepWalk、Node2Vec、商品属性四路召回分数。
        # f_4：log(1+用户历史点击数)；f_5：log(1+候选商品点击数)。
        # f_6～f_35：最近6个历史商品各提供5维，按由近到远排列：
        # [DeepWalk余弦相似度、Node2Vec余弦相似度、属性余弦相似度、
        #  历史商品→候选商品的ItemCF分数、候选商品→历史商品的ItemCF分数]。
        # 历史不足6条或某向量缺失时，相应特征保留为0；跳过属性召回时维度会减少。
        width = len(channels) + 2 + 6 * (len(self.models) + 2)
        # 行数=候选商品数，列数=特征数；先置0，再按固定顺序填入各组特征。
        values = np.zeros((len(candidates), width), dtype=np.float32)
        for channel, scores in enumerate(channels):
            values[:, channel] = [scores.get(item, 0.) for item in candidates]
        offset = len(channels)
        # 对点击次数取log(1+x)，压缩热门商品和长历史用户的数值范围。
        values[:, offset] = np.log1p(len(history))
        values[:, offset + 1] = np.log1p([self.counts[item] for item in candidates])
        offset += 2
        for item in reversed(history[-6:]):
            for model in self.models:
                if item in model:
                    positions = [i for i, candidate in enumerate(candidates) if candidate in model]
                    indices = [model.key_to_index[candidates[i]] for i in positions]
                    # 一次计算所有有效候选与当前历史商品的余弦相似度；限制分母防止除0。
                    norms = np.maximum(model.norms[indices], 1e-12)
                    seed = model.get_vector(item, norm=True)
                    values[positions, offset] = (model.vectors[indices] @ seed) / norms
                offset += 1
            # ItemCF具有方向性，所以将正向和反向关联分数作为两列分别输入模型。
            forward = self.associations.get(item, {})
            values[:, offset] = [forward.get(candidate, 0.) for candidate in candidates]
            values[:, offset + 1] = [self.associations.get(candidate, {}).get(item, 0.) for candidate in candidates]
            offset += 2
        return values


# 工具：把二维特征矩阵拆成Keras要求的字典，键名必须与网络输入名f_0、f_1等一致。
def model_input(array):
    return {f'f_{index}': array[:, index] for index in range(array.shape[1])}


# 工具：训练二分类排序模型；标签1表示候选是用户留出的真实下一次点击，0表示采样负例。
def train_ranker(array, labels, epochs, seed):
    import tensorflow as tf
    from final_rank import DenseFeat, Wide_Deep
    # 固定随机种子，尽量保持相同输入、参数下的运行结果可复现。
    tf.keras.utils.set_random_seed(seed)
    columns = [DenseFeat(f'f_{i}') for i in range(array.shape[1])]
    # Wide分支学习线性权重；Deep分支用64、32两个隐藏层学习非线性交互，训练时dropout=0.1。
    model = Wide_Deep(columns, columns, dnn_hidden_units=(64, 32), dnn_dropout=.1)
    # 使用Adam优化器最小化二元交叉熵；每批256条样本，训练epochs轮。
    model.compile('adam', 'binary_crossentropy')
    model.fit(model_input(array), labels, batch_size=256, epochs=epochs, verbose=2, shuffle=True)
    return model


# 工具：对一位用户先召回、再用模型打分，最终返回Top50和排序前的完整候选集。
def rank(model, scaler, recall, history):
    candidates, channels = recall.recall_for(history)
    # 预测时沿用训练阶段拟合的标准化器，不在测试数据上重新拟合。
    values = scaler.transform(recall.features(history, candidates, channels)).astype(np.float32)
    # training=False关闭dropout，并让批归一化使用训练时记录的统计量。
    probabilities = model(model_input(values), training=False).numpy().reshape(-1)
    # 按候选商品的点击热度对分数降权，减少热门商品的优势；分母最低为1。
    # 这里用商品ID查点击次数，而不是用模型输出的小数分数查字典。
    adjusted = probabilities / np.array([max(1., np.log10(recall.counts[item] + 1)) for item in candidates])
    # 从调整后的高分到低分取50个；CSV里的商品先后顺序就是这里的排序结果。
    order = np.argsort(-adjusted, kind='stable')[:50]
    return [candidates[i] for i in order], candidates


# ==================== 完整运行流程：从这里顺着步骤编号阅读 ====================
def run(args):
    # 步骤1：根据数据目录和阶段号定位原始CSV；兼容官方嵌套目录和扁平目录。
    data_dir = args.data_dir.resolve()
    required = [find_data_file(data_dir, 'train', f'underexpose_train_click-{args.phase}.csv'),
                find_data_file(data_dir, 'test', f'underexpose_test_click-{args.phase}.csv')]
    if not args.skip_item_features:
        required.append(find_data_file(data_dir, 'train', 'underexpose_item_feat.csv'))
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise SystemExit('缺少原始CSV；请按 README 准备数据后再运行：\n' +
                         '\n'.join(f'  {p}' for p in missing))
    # 步骤2：读取、去重、按时间排序，并确认训练用户与待预测用户不重叠。
    train, test = read_clicks(required[0]), read_clicks(required[1])
    if set(train.user_id) & set(test.user_id):
        raise ValueError('训练和测试用户重叠，无法可靠地留出最后一次点击')
    # 步骤3：按用户聚合点击序列；至少两次点击才能留下历史，并取出最后一次作标签。
    train_groups = train.groupby('user_id', sort=True).item_id.agg(list).to_dict()
    train_groups = {user: history for user, history in train_groups.items() if len(history) >= 2}
    if args.max_train_users:
        # 指定小样本模式时，固定种子随机抽取用户；默认0表示不限制训练用户数。
        keys = sorted(train_groups)
        selected = np.random.default_rng(args.seed).choice(keys, min(len(keys), args.max_train_users), replace=False)
        train_groups = {user: train_groups[user] for user in sorted(selected)}
    if len(train_groups) < 5:
        raise ValueError('至少需要5位有两次及以上点击的训练用户')
    # 例如[A,B,C]拆成历史[A,B]与标签C；C不进入该用户的召回/特征历史。
    histories = {user: history[:-1] for user, history in train_groups.items()}
    labels = {user: history[-1] for user, history in train_groups.items()}
    # 步骤4：准备测试用户历史；qtime文件给出真正需要预测的用户及其预测时间点。
    test_groups = test.groupby('user_id', sort=True).item_id.agg(list).to_dict()
    qtime = find_data_file(data_dir, 'test', f'underexpose_test_qtime-{args.phase}.csv')
    if qtime.is_file():
        query = pd.read_csv(qtime, header=None, names=['user_id', 'time'], dtype={'user_id': str})
        query_times = dict(zip(query.user_id, query.time))
        test_groups = {user: history for user, history in test_groups.items() if user in query_times}
        # 只保留查询时间之前的点击，防止使用预测时间点之后的行为。
        if any((test['time'] >= test['user_id'].map(query_times)).fillna(False)):
            filtered = test[test['time'] < test['user_id'].map(query_times)]
            test_groups = filtered.groupby('user_id', sort=True).item_id.agg(list).to_dict()
        # 查询名单中即使有用户完全没有历史，也要保留，之后由热门商品补齐候选。
        for user in query_times:
            test_groups.setdefault(user, [])
    if args.max_test_users:
        test_groups = dict(list(test_groups.items())[:args.max_test_users])
    if not test_groups:
        raise ValueError('没有可预测的测试用户')
    # 步骤5：构建ItemCF关联表和商品热度。所有训练用户均去掉最后一次点击。
    # 小样本参数只限制排序训练用户，其他用户的可见历史仍可贡献共现统计。
    # 测试用户在查询之前已知的历史也可用于共现统计，不使用其未知目标。
    visible = {user: history[:-1] for user, history in train.groupby('user_id').item_id.agg(list).items() if len(history) >= 2}
    associations, counts = build_associations({**visible, **test_groups})
    # 步骤6：加载已有DeepWalk、Node2Vec商品向量；本流程不会重新训练这两个图模型。
    models = []
    for filename in ('deepwalk_offline.bin', 'node2vec_offline.bin'):
        path = data_dir / filename
        if not path.is_file():
            raise SystemExit(f'缺少图模型：{path}；请按 README 训练或准备对应阶段的图向量')
        print('加载图模型:', path, flush=True)
        models.append(KeyedVectors.load_word2vec_format(str(path), binary=True))
        models[-1].fill_norms()
    # 默认再加入商品属性向量（先PCA降维），与ItemCF合起来共四路召回。
    if not args.skip_item_features:
        print('商品特征PCA降维', flush=True)
        models.append(read_content(required[2]))
    # 步骤7：建立召回器，限定合法商品目录，并批量缓存相似商品，减少重复计算。
    recall = Recaller(models, associations, counts, args.recall, args.neighbors,
                      allowed_items=None if args.skip_item_features else models[-1].index_to_key)
    recall.prepare_neighbors(list(histories.values()) + list(test_groups.values()))
    # 步骤8：按用户划分80%训练、20%验证，避免同一用户的候选同时进入两边。
    users_train, users_valid = train_test_split(sorted(histories), test_size=.2, random_state=args.seed)
    rng = np.random.default_rng(args.seed)
    batches, targets = [], []
    hits = 0
    # 每位训练用户先真实召回；命中标签时，构造1个正例与最多negatives个随机负例。
    # 默认每位命中的训练用户提供1+20=21条样本，每条样本是一对“用户、候选商品”。
    for user in tqdm(users_train, desc='训练候选与特征'):
        candidates, channels = recall.recall_for(histories[user])
        target = labels[user]
        if target not in candidates:
            # 召回没找到真实目标时跳过此训练用户，不把答案人为塞进候选。
            continue
        hits += 1
        negatives = [item for item in candidates if item != target]
        selected = [target] + rng.choice(negatives, size=min(args.negatives, len(negatives)), replace=False).tolist()
        batches.append(recall.features(histories[user], selected, channels))
        targets.extend([1.] + [0.] * (len(selected)-1))
    if not batches:
        raise ValueError('训练候选未召回任何正样本；请增大--recall或使用更多训练数据')
    # 步骤9：合并所有训练样本，仅用训练特征拟合均值和标准差，得到标准化器。
    features = np.concatenate(batches)
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features).astype(np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError('训练特征包含无效数值')
    print(f'训练用户={len(users_train)}，命中正样用户={hits}，样本={len(targets)}，特征={features.shape[1]}', flush=True)
    # 步骤10：训练排序模型。验证用户不参与梯度更新，也不参与标准化器的拟合。
    model = train_ranker(features, targets, args.epochs, args.seed)
    # 步骤11：为每位验证用户完整召回和排序，分别统计候选命中及Top50命中。
    # 分母包含全部验证用户，即使真实目标未被召回也计作未命中。
    valid_hit = valid_recall = 0
    for user in tqdm(users_valid, desc='按用户留出验证'):
        top50, candidates = rank(model, scaler, recall, histories[user])
        valid_hit += labels[user] in top50
        valid_recall += labels[user] in candidates
    # 步骤12：对测试用户预测并保存结果；这里使用刚才训练的同一个模型，不再重训。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = args.output_dir / 'recommendations.csv'
    with result.open('w') as handle:
        for user, history in tqdm(test_groups.items(), desc='生成测试用户Top50'):
            top50, _ = rank(model, scaler, recall, history)
            assert len(top50) == len(set(top50)) == 50
            assert not set(top50) & set(history)
            # 无表头CSV：第一列用户ID，后面50列是从高到低排序的商品ID，不写分数。
            handle.write(','.join([user] + top50) + '\n')
    # 同时保存模型和标准化器；以后加载模型时，必须使用相同的特征顺序和标准化器。
    model.save(str(args.output_dir / 'wide_and_deep.h5'))
    with (args.output_dir / 'scaler.pkl').open('wb') as handle:
        pickle.dump(scaler, handle)
    # 记录实际样本规模、参数和验证指标。图向量为预训练文件，可能含验证目标交互，
    # 因而这些指标用于运行诊断，不等于严格无泄漏的比赛成绩；限制也写进evaluation_note。
    metrics = {
        'data_dir': str(data_dir), 'phase': args.phase, 'seed': args.seed,
        'train_users': len(users_train), 'validation_users': len(users_valid), 'test_users': len(test_groups),
        'training_samples': len(targets), 'features': features.shape[1], 'epochs': args.epochs,
        'recall': args.recall, 'channels': ['itemCF', 'deepwalk', 'node2vec'] + ([] if args.skip_item_features else ['item_feature']),
        'validation_recall': valid_recall / len(users_valid), 'validation_hit_rate_at_50': valid_hit / len(users_valid),
        'evaluation_note': 'Diagnostic only: supplied pretrained graph models may include validation target interactions. Not a leakage-free reproduction of the original benchmark.',
        'synthetic_verification': args.synthetic_verification,
        'eligible_catalog_items': len(recall.catalog),
        'pretrained_graph_models': [str(data_dir / name) for name in ('deepwalk_offline.bin', 'node2vec_offline.bin')],
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print('推荐结果:', result.resolve(), flush=True)
    return result


# 命令行入口：默认phase=0、3轮训练、500个候选、每个种子50个邻居、20个负例。
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output-dir', type=Path, default=ROOT_DIR / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--phase', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--recall', type=int, default=500)
    parser.add_argument('--neighbors', type=int, default=50)
    parser.add_argument('--negatives', type=int, default=20)
    parser.add_argument('--max-train-users', type=int, default=0, help='0 = all train users')
    parser.add_argument('--max-test-users', type=int, default=0, help='0 = all test users')
    parser.add_argument('--skip-item-features', action='store_true', help='Explicitly run only ItemCF + graph recall')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--synthetic-verification', action='store_true', help=argparse.SUPPRESS)
    # 将终端里的--参数解析成args，检查范围后交给run执行完整流程。
    args = parser.parse_args(argv)
    if args.recall < 50 or min(args.epochs, args.neighbors, args.negatives) < 1:
        parser.error('recall must be >= 50; epochs, neighbors and negatives must be positive')
    if min(args.max_train_users, args.max_test_users, args.phase) < 0:
        parser.error('phase and user limits must be nonnegative')
    run(args)


if __name__ == '__main__':
    main()
