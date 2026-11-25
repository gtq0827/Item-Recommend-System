"""Validate a generated recommendation file against the official phase inputs."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from project_paths import DATA_DIR, ROOT_DIR, find_data_file


# 独立输出校验：运行训练脚本后执行，用官方输入核对CSV格式和推荐合法性。
def validate(output_dir, data_dir=DATA_DIR, phase=0, check_model=False):
    # 1. 读取结果、官方待预测用户名单以及用户历史点击。ID统一按字符串比较。
    result = pd.read_csv(output_dir / 'recommendations.csv', header=None, dtype=str)
    query = pd.read_csv(find_data_file(data_dir, 'test', f'underexpose_test_qtime-{phase}.csv'),
                        header=None, dtype={0: str})
    clicks = pd.read_csv(find_data_file(data_dir, 'test', f'underexpose_test_click-{phase}.csv'),
                         header=None, dtype={0: str, 1: str})
    seen = clicks.groupby(0)[1].agg(set).to_dict()
    # 2. 每位查询用户恰好一行，每行51列（用户ID+50个商品ID），不允许重复用户。
    if result.shape != (len(query), 51) or not result[0].is_unique:
        raise ValueError('推荐行数、列数或用户唯一性不符合提交格式')
    if set(result[0]) != set(query[0]):
        raise ValueError('推荐用户没有完整覆盖官方query用户')
    # 3. 逐用户检查50个商品不重复，并且没有出现在其已有点击记录中。
    for row in result.itertuples(index=False, name=None):
        if len(set(row[1:])) != 50 or set(row[1:]) & seen.get(row[0], set()):
            raise ValueError(f'用户{row[0]}含重复商品或已点击商品')
    # 4. 所有推荐商品都必须在官方商品特征目录中。
    catalog = set(pd.read_csv(find_data_file(data_dir, 'train', 'underexpose_item_feat.csv'),
                             header=None, usecols=[0], dtype=str)[0])
    missing = set(result.iloc[:, 1:].to_numpy().ravel()) - catalog
    if missing:
        raise ValueError(f'结果含{len(missing)}个不在官方商品目录中的商品')
    report = {'users': len(result), 'items_per_user': 50, 'all_query_users_covered': True,
              'unique_items_per_user': True, 'seen_items_excluded': True,
              'all_items_in_official_catalog': True}
    # 5. 可选检查：重新加载模型及标准化器，用两行占位特征检查能否正常推理。
    # 占位特征只验证模型可加载，不用于衡量推荐效果。
    if check_model:
        import tensorflow as tf
        from final_rank import DNN
        model = tf.keras.models.load_model(str(output_dir / 'wide_and_deep.h5'), custom_objects={'DNN': DNN})
        with (output_dir / 'scaler.pkl').open('rb') as handle:
            scaler = pickle.load(handle)
        values = scaler.transform(np.zeros((2, scaler.n_features_in_), dtype=np.float32))
        predictions = model({f'f_{i}': values[:, i] for i in range(values.shape[1])}, training=False).numpy()
        if not np.isfinite(predictions).all():
            raise ValueError('保存后的模型预测出现无效数值')
        report.update(saved_model_reload_and_prediction=True, feature_count=scaler.n_features_in_)
    # 6. 将以上格式和模型检查结果保存为独立报告。
    (output_dir / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT_DIR / 'outputs' / 'phase0')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--phase', type=int, default=0)
    parser.add_argument('--check-model', action='store_true')
    args = parser.parse_args()
    validate(args.output_dir, args.data_dir, args.phase, args.check_model)
