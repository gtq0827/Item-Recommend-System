"""Project paths, independent of the terminal/PyCharm working directory."""
from pathlib import Path

# 以本文件所在目录作为项目根目录；不依赖终端或PyCharm当前的工作目录。
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'underexpose_train'
TEST_DIR = DATA_DIR / 'underexpose_test'


def find_data_file(data_dir, split, filename):
    """Find a plain or gzip-compressed CSV in a flat or official nested layout."""
    # 先找扁平结构，例如underexpose_train/underexpose_train_click-0.csv。
    directory = Path(data_dir) / f'underexpose_{split}'
    direct = directory / filename
    # 再找官方嵌套结构；测试qtime文件与同阶段click文件放在同一click目录中。
    stem = Path(filename).stem
    parent = stem.replace('_qtime-', '_click-')
    nested = directory / parent / filename
    for path in (direct, nested):
        for candidate in (path, path.with_suffix('.csv.gz')):
            if candidate.is_file():
                return candidate
    return direct


# 按运行阶段列出必需输入：主流程仅需原始CSV和图向量，不要求预先存在排序模型。
def required_files(stage='solution', phase=0):
    if stage == 'rank':
        return [ROOT_DIR / 'feature_all_train_test.pkl']
    files = [find_data_file(DATA_DIR, 'train', f'underexpose_train_click-{phase}.csv'),
             find_data_file(DATA_DIR, 'test', f'underexpose_test_click-{phase}.csv')]
    if stage == 'graph':
        return files + [find_data_file(DATA_DIR, 'test', f'underexpose_test_qtime-{phase}.csv')]
    return files + [find_data_file(DATA_DIR, 'train', 'underexpose_item_feat.csv'),
                    DATA_DIR / 'deepwalk_offline.bin', DATA_DIR / 'node2vec_offline.bin']


# 一次列出所有缺失文件，避免执行到训练中途才因找不到文件退出。
def check_required_files(stage='solution', phase=0):
    missing = [path for path in required_files(stage, phase) if not path.is_file()]
    if missing:
        raise SystemExit('缺少以下输入文件：\n' + '\n'.join(f'  {path}' for path in missing))
