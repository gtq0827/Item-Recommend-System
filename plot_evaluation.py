"""Plot the temporal evaluation metrics using Matplotlib."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

from project_paths import ROOT_DIR

COLORS = {'model': '#126E6A', 'popularity': '#657383'}
LABELS = {'model': '四路召回 + Wide & Deep', 'popularity': '热门商品基线'}
METRICS = (
    ('recall_at_500', '候选召回率@500'),
    ('hit_rate_at_50', '命中率@50'),
    ('ndcg_at_50', 'NDCG@50'),
)


def configure_fonts(font_path):
    if font_path:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        for family in ('Hiragino Sans GB', 'Noto Sans CJK SC', 'Source Han Sans SC', 'Microsoft YaHei'):
            try:
                font_manager.findfont(family, fallback_to_default=False)
            except ValueError:
                continue
            break
        else:
            raise SystemExit('找不到中文字体。请用 --font 指定中文字体文件。')
    plt.rcParams.update({
        'font.family': [family, 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelcolor': '#273640',
        'xtick.color': '#445462',
        'ytick.color': '#445462',
        'text.color': '#263641',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.edgecolor': '#B5C0C9',
        'grid.color': '#E5EAEE',
        'axes.axisbelow': True,
        'pdf.fonttype': 3,
    })


def plot(report, output):
    metrics = report['metrics']
    holdout_size = report['evaluation_cohort']['eligible_users']
    phase = report['phase']
    figure, (overview, curve) = plt.subplots(1, 2, figsize=(11.4, 4.65))
    figure.subplots_adjust(left=.146, right=.965, top=.69, bottom=.22, wspace=.41)
    figure.text(.071, .90, '商品推荐离线评估', fontsize=21, weight='bold', color='#1B2933')
    figure.text(.071, .826, f'第 {phase} 阶段时间留出  ·  {holdout_size:,} 位用户',
                fontsize=10.5, color='#526371')
    legend = [
        Line2D([0], [0], label=LABELS[method], color=COLORS[method], linewidth=2,
               linestyle='-' if method == 'model' else '--',
               marker='o' if method == 'model' else 's')
        for method in LABELS
    ]
    figure.legend(handles=legend, frameon=False, loc='upper right', bbox_to_anchor=(.973, .903))

    overview.set_title('A  候选与排序指标', loc='left', pad=14, fontsize=12, weight='bold')
    highest = max(metrics[method][key] for method in LABELS for key, _ in METRICS)
    for method, offset in (('model', -.17), ('popularity', .17)):
        values = [metrics[method][key] for key, _ in METRICS]
        bars = overview.barh([i + offset for i in range(len(values))], values, height=.27,
                             color=COLORS[method], alpha=1 if method == 'model' else .82,
                             hatch=None if method == 'model' else '////',
                             edgecolor=COLORS[method] if method == 'model' else '#DCE2E8', linewidth=.65)
        overview.bar_label(bars, labels=[f'{value:.2%}' for value in values], padding=4, fontsize=9)
    overview.set_yticks(range(len(METRICS)), [label for _, label in METRICS])
    overview.tick_params(axis='y', length=0, pad=7)
    overview.invert_yaxis()
    overview.set_xlim(0, max(.1, highest * 1.30))
    overview.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    overview.xaxis.set_major_formatter(PercentFormatter(1))
    overview.grid(axis='x')

    curve.set_title('B  不同推荐数量下的命中率', loc='left', pad=14, fontsize=12, weight='bold')
    cutoffs = sorted(map(int, metrics['model']['hit_rate_by_k']))
    for method, style, marker in (('model', '-', 'o'), ('popularity', '--', 's')):
        values = [metrics[method]['hit_rate_by_k'][str(k)] for k in cutoffs]
        curve.plot(cutoffs, values, style, marker=marker, markersize=4.5,
                   linewidth=2, color=COLORS[method])
    curve.set_xticks(cutoffs)
    curve.set_xlim(0, max(cutoffs) + 3)
    curve.set_ylim(0, max(.01, max(metrics[method]['hit_rate_at_50'] for method in LABELS) * 1.18))
    curve.set_xlabel('每位用户的推荐数量 K')
    curve.set_ylabel('命中率')
    curve.yaxis.set_major_formatter(PercentFormatter(1))
    curve.grid(axis='y')

    effect = metrics['paired_bootstrap']['hit_rate_at_50_difference']
    low, high = effect['ci_95']
    figure.text(.071, .075,
                f"命中率@50 差值 {effect['estimate'] * 100:+.2f} 个百分点"
                f'  ·  用户配对 Bootstrap 95% 区间 [{low * 100:+.2f}, {high * 100:+.2f}]',
                fontsize=9, color='#526371')
    figure.savefig(output.with_suffix('.png'), dpi=300, facecolor='white')
    figure.savefig(output.with_suffix('.pdf'), facecolor='white')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', type=Path, nargs='?',
                        default=ROOT_DIR / 'outputs' / 'temporal_evaluation' / 'metrics.json')
    parser.add_argument('--font', type=Path, help='Path to a Chinese font file')
    args = parser.parse_args()
    configure_fonts(args.font)
    plot(json.loads(args.metrics.read_text()), args.metrics.with_name('evaluation'))


if __name__ == '__main__':
    main()
