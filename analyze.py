#!/usr/bin/env python

# TODO in BPGP we also plotted unique_values and unique_ratio. are they interesting?

import argparse
import os
import re
from matplotlib import pyplot as plt

STAT_FILE = 'run.log'
RESULTS_IN_PLOT = 4
WRITE_VALUES_ABOVE_POINTS = True


def get_dirs(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def get_run_result(d):
    with open(os.path.join(d, STAT_FILE)) as f:
        lines = f.readlines()

    # INFO:eckity.statistics.best_avg_worst_size_tree_statistics:fitness: best 0.14137016236782074, worst 0.0029264071490615606, average 0.08269203132484108. average size 43.95
    r = re.compile(
        r'.*best_avg_worst_size_tree_statistics:fitness: best (\d+\.\d+), worst (\d+\.\d+), average (\d+\.\d+). average size (\d+\.\d+)')
    bests = [float(r.match(l).group(1)) for l in lines if r.match(l)]
    worst = [float(r.match(l).group(2)) for l in lines if r.match(l)]
    average = [float(r.match(l).group(3)) for l in lines if r.match(l)]
    average_sizes = [float(r.match(l).group(4)) for l in lines if r.match(l)]
    assert len(bests) == len(worst) == len(average) == len(average_sizes)
    return {
        'run': d,
        'bests': bests,
        'worst': worst,
        'average': average,
        'average_sizes': average_sizes,
    }


def analyze_single_run(d):
    run_result = get_run_result(d)
    fig, ax = plt.subplots(1, 1)
    plot_single_run(ax, run_result)
    ax.legend()
    plt.show()


def plot_single_run(ax, r):
    out = ax.plot(r['bests'], '.-', label='best')
    ax.plot(r['worst'], '.-', label='worst')
    ax.plot(r['average'], '.-', alpha=0.6, label='average')
    # # plot nothing, just add to our legend
    ax.plot([], [], '.-', label='Average Sizes', color='tab:pink')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=min(1, max(r['bests']) * 2))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(r['run'])

    if WRITE_VALUES_ABOVE_POINTS:
        for x, y in zip(range(len(r['bests'])), r['bests']):
            plt.annotate("{:.2f}".format(y),  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 3),  # distance from text to points (x,y)
                         ha='center',  # horizontal alignment can be left, right or center
                         alpha=0.5,
                         fontsize='small')

    ax2 = ax.twinx()
    ax2.set_ylabel("Average Sizes")
    # ax2.set_ylim([0,1])
    ax2.plot(r['average_sizes'], '.-', label='Average Sizes', color='tab:pink')
    # create a separate legend
    # ax2.legend(loc=2)

    return out


def analyze(d, single_run):
    if single_run:
        analyze_single_run(d)
    else:
        analyze_regression(d)


def analyze_regression(regression_dir):
    results = []
    for d in get_dirs(regression_dir):
        results.append(get_run_result(d))
    all_results_to_plot = [r for r in results if r is not None]
    while all_results_to_plot:
        results_to_plot = all_results_to_plot[:RESULTS_IN_PLOT]
        all_results_to_plot = all_results_to_plot[RESULTS_IN_PLOT:]

        fig, ax = plt.subplots(len(results_to_plot), 1, figsize=(10, 10))
        for (index, axes) in enumerate(ax):
            plot_single_run(axes, results_to_plot[index])
        axes.legend(loc='lower left')
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Analyze regression results", )
    parser.add_argument("--dir", default='regression', help="directory containing run results, or regression")
    parser.add_argument("--single-run", "-s", action='store_true',
                        help="'dir' points to single run instead of regression")

    args = parser.parse_args()
    analyze(args.dir, args.single_run)
