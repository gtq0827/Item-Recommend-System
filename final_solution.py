#!/usr/bin/env python
# coding: utf-8
"""Main entry: raw CSV -> multi-channel recall -> Wide&Deep -> Top50 CSV."""
# 阅读顺序：本文件启动程序 → run_pipeline.py 执行业务流程 → final_rank.py 定义排序网络。
import os

# 第一步：在导入NumPy之前限制矩阵运算线程数，避免笔记本上线程过多。
# setdefault只在尚未配置时设置，允许调用者通过环境变量自行调整。
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')

# 第二步：导入真正的流程入口；这里只负责启动，训练和预测逻辑位于run_pipeline.py。
from run_pipeline import main

# 第三步：直接运行此文件时解析命令行参数，然后执行完整流程。
# 示例：.venv/bin/python final_solution.py --output-dir outputs/phase0
if __name__ == '__main__':
    main()
