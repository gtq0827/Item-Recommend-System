# 商品 Top 50 推荐系统

**简体中文** · [English](README.en.md)

基于用户点击历史生成 50 个商品推荐。系统融合 ItemCF、DeepWalk、Node2Vec 和商品特征四路候选，提取 36 维特征，使用 Wide & Deep 排序。

## 安装与快速检查

使用 Python 3.11 和 uv。以下命令均可在下载数据前运行。

```bash
uv sync --locked
uv run --locked python -m unittest -q test_pipeline
uv run --locked python check_setup.py --environment-only
```

## 准备数据并运行

从天池的 [KDD Cup 2020 推荐 Debias 赛道](https://tianchi.aliyun.com/competition/entrance/231785/introduction) 获取官方数据，将训练和测试目录放入项目根目录的 `data/`。支持原始 CSV 和 Gzip 压缩的 CSV，也支持官方嵌套布局。

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

首次运行第 0 阶段时，先训练 DeepWalk 和 Node2Vec。生成的两个 `.bin` 文件保存在 `data/`，后续可复用。数据和训练产物由使用者在本地准备。

```bash
uv run --locked python check_setup.py --stage graph
uv run --locked python build_graph_model.py
uv run --locked python check_setup.py
uv run --locked python final_solution.py --max-train-users 100 --max-test-users 10 --epochs 1 --recall 100 --neighbors 15 --output-dir outputs/quickstart
```

运行完整第 0 阶段，并检查查询用户覆盖率、商品唯一性、目录合法性与排序模型加载。

```bash
uv run --locked python final_solution.py --output-dir outputs/phase0
uv run --locked python validate_output.py --output-dir outputs/phase0 --check-model
```

`recommendations.csv` 每行是一个用户 ID 和按排序排列的 50 个商品 ID，无表头。输出目录还包含 `wide_and_deep.h5`、`scaler.pkl` 和 `metrics.json`。`--phase N` 可选择其他阶段，运行前需准备对应阶段的数据和图向量。

## 时间留出评估

```bash
uv run --locked python evaluate_recommendations.py
uv run --no-project --python 3.11 --with 'matplotlib==3.11.2' python plot_evaluation.py
```

第 0 阶段评估按点击时间划分历史、排序训练和留出三个窗口。图向量在历史窗口重训，热门基线与模型使用同一批 8,394 位评估用户和同一商品目录。评估将商品特征视为在时间切点已可用，缺失目录的目标计为未命中。图表脚本可通过 `--font` 指定中文字体。

| 指标 | 四路召回 + Wide & Deep | 热门基线 |
| --- | ---: | ---: |
| 候选召回率@500 | 9.95% | 5.67% |
| 命中率@50 | 2.84% | 0.73% |
| NDCG@50 | 0.00891 | 0.00161 |

评估指标来自课堂项目的第 0 阶段时间留出实验。主流程另外输出按用户切分的诊断指标，预训练图向量可能包含该验证集的目标交互。

## 主要文件

| 文件 | 用途 |
| --- | --- |
| `final_solution.py`、`run_pipeline.py` | 推荐流程入口和实现 |
| `build_graph_model.py`、`recall_function.py` | 图向量训练与召回实现 |
| `final_rank.py` | Wide & Deep 排序网络 |
| `evaluate_recommendations.py`、`plot_evaluation.py` | 时间留出评估与绘图 |
| `check_setup.py`、`validate_output.py`、`test_pipeline.py` | 环境、输出与回归检查 |
