import os
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent

DRIVER_DIC = {
    'not_stated': ('driver not officially stated', 'tab:gray'),
    'recipe': ('training data / recipe change', 'tab:green'),
    'architecture': ('architecture change', 'tab:red'),
    'architecture_and_data': ('architecture + data change', 'tab:orange'),
}

FIGURE_DIC = {
    'qwen_generation': {
        'title': 'Qwen benchmark scores across generations (~7-8B base)',
        'by_release': False,
    },
    'qwen_generation_architecture': {
        'title': 'Qwen GPQA-Diamond across generations (~27-32B dense):\n'
                 'architecture + data (3->3.5) vs. recipe (3.5->3.8)',
        'by_release': True,
    },
}


def process_df(args, figure):
    df = pd.read_csv(args.input_file)
    df = df[df['figure'] == figure]
    if df.empty:
        raise SystemExit(f'No rows for figure {figure} in {args.input_file}')
    df['release'] = pd.to_datetime(df['release'])
    # generations sort by release, the same for every benchmark
    order = df.groupby('generation')['release'].min().sort_values().index.tolist()
    df['generation'] = pd.Categorical(df['generation'], categories=order, ordered=True)
    return df.sort_values('generation', kind='stable')


def make_plot(args, df, figure):
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "grid.linewidth": args.bg_line_width,
            "figure.figsize": args.fig_size,
        })

    benchmarks = list(dict.fromkeys(df['benchmark']))
    fig, axs = plt.subplots(
        len(benchmarks) + 1, 1, sharex=True,
        gridspec_kw={'height_ratios': [3] * len(benchmarks) + [1]})

    gens = df.drop_duplicates('generation').sort_values('generation')
    if FIGURE_DIC[figure]['by_release']:
        x_of = dict(zip(gens['generation'], gens['release']))
    else:
        x_of = {g: i for i, g in enumerate(gens['generation'])}

    for ax, bench in zip(axs, benchmarks):
        sub = df[df['benchmark'] == bench]
        sns.lineplot(x=sub['generation'].map(x_of), y=sub['score'], marker=args.marker,
                     linewidth=args.line_width, ax=ax)
        ax.set(xlabel=None, ylabel=f'{bench} (%)')

    # one segment per step, coloured by the newer generation's driver
    strip = axs[-1]
    drivers = gens['driver'].tolist()
    for i in range(1, len(gens)):
        strip.plot([x_of[gens['generation'].iloc[i - 1]], x_of[gens['generation'].iloc[i]]],
                   [0, 0], color=DRIVER_DIC[drivers[i]][1], linewidth=args.strip_width,
                   solid_capstyle='butt')
    strip.set(yticks=[], ylabel='driver', xlabel=args.x_label)
    strip.set_xticks(list(x_of.values()), labels=list(x_of.keys()))
    if FIGURE_DIC[figure]['by_release']:
        plt.setp(strip.get_xticklabels(), rotation=args.x_rotation, ha='right')

    if args.despine:
        sns.despine(fig=fig, top=True, right=True, left=False, bottom=False)

    used = [d for d in DRIVER_DIC if d in drivers[1:]]
    handles = [Line2D([0], [0], color=DRIVER_DIC[d][1], linewidth=6) for d in used]
    # legend goes under the x label, whose height depends on tick rotation
    fig.canvas.draw()
    label_y = strip.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted()).y0
    fig.legend(handles, [DRIVER_DIC[d][0] for d in used], loc='upper center',
               bbox_to_anchor=(0.5, label_y - 0.01), ncol=2, frameon=False)
    fig.suptitle(args.title or FIGURE_DIC[figure]['title'])
    return fig


def parse_args():
    parser = argparse.ArgumentParser()

    # input / output
    parser.add_argument('--input_file', type=str, default=str(HERE / 'data.csv'))
    parser.add_argument('--figures', nargs='+', type=str, default=list(FIGURE_DIC),
                        choices=list(FIGURE_DIC))
    parser.add_argument('--results_dir', type=str, default=str(HERE))
    parser.add_argument('--save_format', type=str, default='png',
                        choices=['pdf', 'png', 'jpg'])

    # style -- same names and defaults as plot.py in the analysis repos
    parser.add_argument('--context', type=str, default='notebook')
    parser.add_argument('--style', type=str, default='whitegrid')
    parser.add_argument('--palette', type=str, default='colorblind')
    parser.add_argument('--font_family', type=str, default='serif')
    parser.add_argument('--font_scale', type=float, default=1.0)
    parser.add_argument('--bg_line_width', type=float, default=0.5)
    parser.add_argument('--line_width', type=float, default=1.6)
    parser.add_argument('--strip_width', type=float, default=8)
    parser.add_argument('--fig_size', nargs='+', type=float, default=[6, 6])
    parser.add_argument('--marker', type=str, default='o')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--x_label', type=str, default='Qwen generation')
    parser.add_argument('--x_rotation', type=float, default=30)
    parser.add_argument('--despine', action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    for figure in args.figures:
        df = process_df(args, figure)
        fig = make_plot(args, df, figure)
        output_file = os.path.join(args.results_dir, f'{figure}.{args.save_format}')
        fig.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print('Save plot to directory ', output_file)
    return 0


if __name__ == '__main__':
    main()
