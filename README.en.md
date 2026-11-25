# Top-50 Product Recommender

[简体中文](README.md) · **English**

This system recommends 50 products based on a user's click history. It retrieves candidates through four channels (ItemCF, DeepWalk, Node2Vec, and product features), computes 36 ranking features, and scores candidates with a Wide & Deep model.

## Installation and quick checks

Use Python 3.11 and uv. You can run these commands before downloading the dataset.

```bash
uv sync --locked
uv run --locked python -m unittest -q test_pipeline
uv run --locked python check_setup.py --environment-only
```

## Prepare the data and run the pipeline

Download the official dataset from the [KDD Cup 2020 Debiasing track on Tianchi](https://tianchi.aliyun.com/competition/entrance/231785/introduction). Place the training and test directories under `data/` at the repository root. The pipeline supports plain CSV files, gzip-compressed CSV files, and the official nested directory layout.

```text
data/
├── underexpose_train/
│   ├── underexpose_item_feat.csv
│   └── underexpose_train_click-0.csv
└── underexpose_test/
    └── underexpose_test_click-0/
        ├── underexpose_test_click-0.csv
        └── underexpose_test_qtime-0.csv
```

Before the first Phase 0 run, train the DeepWalk and Node2Vec embeddings. The two resulting `.bin` files are saved under `data/` and can be reused. Prepare the dataset and trained models locally.

```bash
uv run --locked python check_setup.py --stage graph
uv run --locked python build_graph_model.py
uv run --locked python check_setup.py
uv run --locked python final_solution.py --max-train-users 100 --max-test-users 10 --epochs 1 --recall 100 --neighbors 15 --output-dir outputs/quickstart
```

Run the full Phase 0 pipeline and verify coverage of query users, uniqueness of recommendations, catalog membership, and model loading.

```bash
uv run --locked python final_solution.py --output-dir outputs/phase0
uv run --locked python validate_output.py --output-dir outputs/phase0 --check-model
```

Each row of `recommendations.csv` contains a user ID followed by 50 ranked product IDs. The CSV has no header. The output directory also contains `wide_and_deep.h5`, `scaler.pkl`, and `metrics.json`. Use `--phase N` to select another phase after preparing its dataset and graph embeddings.

## Time-based holdout evaluation

```bash
uv run --locked python evaluate_recommendations.py
uv run --no-project --python 3.11 --with 'matplotlib==3.11.2' python plot_evaluation.py
```

The Phase 0 evaluation splits clicks chronologically into history, ranker training, and holdout windows. The graph embeddings are retrained on the history window. The model and the popularity baseline use the same 8,394 evaluation users and the same product catalog. The evaluation assumes product features are available at the time cutoffs and counts targets missing from the catalog as misses. Use `--font` to specify a Chinese font for the plots.

| Metric | Four retrieval channels + Wide & Deep | Popularity baseline |
| --- | ---: | ---: |
| Candidate Recall@500 | 9.95% | 5.67% |
| Hit Rate@50 | 2.84% | 0.73% |
| NDCG@50 | 0.00891 | 0.00161 |

These results are from the classroom project's Phase 0 time-based holdout experiment. The main pipeline also reports diagnostics from a user-based split. Pretrained graph embeddings may contain target interactions from that validation split.

## Project files

| File | Purpose |
| --- | --- |
| `final_solution.py`, `run_pipeline.py` | Pipeline entry point and implementation |
| `build_graph_model.py`, `recall_function.py` | Graph embedding training and candidate retrieval |
| `final_rank.py` | Wide & Deep ranking model |
| `evaluate_recommendations.py`, `plot_evaluation.py` | Time-based holdout evaluation and plotting |
| `check_setup.py`, `validate_output.py`, `test_pipeline.py` | Environment, output, and regression checks |
