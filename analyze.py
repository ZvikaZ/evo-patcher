#!/usr/bin/env python

# TODO in BPGP we also plotted unique_values and unique_ratio. are they interesting?

import argparse
import os
import re
import statistics
from pathlib import Path

from matplotlib import pyplot as plt

STAT_FILE = 'run.log'
RESULTS_IN_PLOT = 4
WRITE_VALUES_ABOVE_POINTS = True


def get_dirs(path):
    def sort_func(item):
        return int(Path(item).stem.split('_')[1])
    return sorted([os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))], key=sort_func)


def get_model_fail_rate(lines):
    # DEBUG:evolution_eval:gen_194_ind_116416 : fitness is 0.5320, model_fail_rate=0.3750, avg_prob_diff=0.5993
    r = re.compile(
        r'DEBUG:evolution_eval:gen_(\d+)_ind_(\d+) : fitness is (-?\d+\.\d+), model_fail_rate=(-?\d+\.\d+), avg_prob_diff=(-?\d+\.\d+)')
    cur_gen = None
    cur_fail_rates = None
    fail_rates = []
    for l in lines:
        m = r.match(l)
        if m:
            # print(f'gen: {m.group(1)}, ind: {m.group(2)}, fitness: {m.group(3)}, fail_rate: {m.group(4)}, avg_prob_diff: {m.group(5)}')
            gen = int(m.group(1))
            fail_rate = float(m.group(4))
            if gen != cur_gen:
                if cur_fail_rates:
                    fail_rates.append(statistics.mean(cur_fail_rates))
                cur_gen = gen
                cur_fail_rates = []
            cur_fail_rates.append(fail_rate)
    return fail_rates


def get_run_result(d):
    with open(os.path.join(d, STAT_FILE)) as f:
        lines = f.readlines()

    # INFO:eckity.statistics.best_avg_worst_size_tree_statistics:fitness: best 0.14137016236782074, worst 0.0029264071490615606, average 0.08269203132484108. average size 43.95
    r = re.compile(
        r'.*best_avg_worst_size_tree_statistics:fitness: best (-?\d+\.\d+), worst (-?\d+\.\d+), average (-?\d+\.\d+). average size (\d+\.\d+)')
    bests = [float(r.match(l).group(1)) for l in lines if r.match(l)]
    worst = [float(r.match(l).group(2)) for l in lines if r.match(l)]
    average = [float(r.match(l).group(3)) for l in lines if r.match(l)]
    average_sizes = [float(r.match(l).group(4)) for l in lines if r.match(l)]

    model_fail_rate = get_model_fail_rate(lines)
    try:
        print(f'{max(model_fail_rate):.4f}', d)
    except ValueError:
        pass

    assert len(bests) == len(worst) == len(average) == len(average_sizes) #== len(model_fail_rate)
    return {
        'run': d,
        'bests': bests,
        'worst': worst,
        'average': average,
        'average_sizes': average_sizes,
        'model_fail_rate': model_fail_rate,
    }


def analyze_single_run(d):
    run_result = get_run_result(d)
    fig, ax = plt.subplots(1, 1)
    plot_single_run(ax, run_result)
    ax.legend()
    plt.show()


def plot_single_run(ax, r, ranges=None):
    out = ax.plot(r['bests'], '.-', label='Best')
    ax.plot(r['worst'], '.-', label='Worst')
    ax.plot(r['average'], '.-', alpha=0.6, label='Average')
    ax.plot(r['model_fail_rate'], '.-', label='Model Fail Rate')
    # # plot nothing, just add to our legend
    ax.plot([], [], '.-', label='Average Sizes', color='tab:pink')
    if ranges:
        ax.set_xlim(left=0, right=ranges['generations'])
        ax.set_ylim(bottom=ranges['min_worst'], top=min(1, ranges['max_best'] * 1.2))
    else:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        try:
            ax.set_ylim(top=min(1, max(r['bests']) * 1.2))
        except ValueError:
            ax.set_ylim(top=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(r['run'])

    if WRITE_VALUES_ABOVE_POINTS:
        i = 0
        for x, y in zip(range(len(r['bests'])), r['bests']):
            i += 1
            # print numerical value for last point, and for every 10th one
            if (len(r['bests']) - i) % 10 == 0:
                label = "{:.3f}".format(y)
            else:
                label = ""
            # if label != prev:
            #     prev = label
            # else:
            #     # avoid consecutive similar values
            #     label = ""

            ax.annotate(label,  # this is the text
                        (x, y),  # these are the coordinates to position the label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 3),  # distance from text to points (x,y)
                        ha='center',  # horizontal alignment can be left, right or center
                        alpha=0.5,
                        fontsize='small')

    ax2 = ax.twinx()
    ax2.set_ylabel("Average Sizes")
    if ranges:
        ax2.set_ylim(bottom=0, top=ranges['max_size'] * 1.1)
    ax2.plot(r['average_sizes'], '.-', label='Average Sizes', color='tab:pink')
    # create a separate legend
    # ax2.legend(loc=2)

    return out


def analyze(d, single_run, write):
    if single_run:
        analyze_single_run(d)
        if d.write:
            raise NotImplementedError
    else:
        analyze_regression(d, write)


def get_ranges(all_results_to_plot):
    max_best = max([max(result['bests']) for result in all_results_to_plot if result['bests']])
    min_worst = min([min(result['worst']) for result in all_results_to_plot if result['worst']])
    max_size = max([max(result['average_sizes']) for result in all_results_to_plot if result['average_sizes']])
    generations = max([len(result['bests']) for result in all_results_to_plot])
    return {
        'max_best': max_best,
        'min_worst': min_worst,
        'max_size': max_size,
        'generations': generations,
    }


def analyze_regression(regression_dir, write):
    results = []
    for d in get_dirs(regression_dir):
        try:
            results.append(get_run_result(d))
        except FileNotFoundError:
            pass
    all_results_to_plot = [r for r in results if r is not None]
    ranges = get_ranges(all_results_to_plot)
    counter = 0
    while all_results_to_plot:
        results_to_plot = all_results_to_plot[:RESULTS_IN_PLOT]
        all_results_to_plot = all_results_to_plot[RESULTS_IN_PLOT:]

        fig, ax = plt.subplots(len(results_to_plot), 1, figsize=(10, 10))
        try:
            for (index, axes) in enumerate(ax):
                plot_single_run(axes, results_to_plot[index], ranges)
        except TypeError:
            plot_single_run(ax, results_to_plot[0], ranges)
        axes.legend(loc='lower left')
        fig.tight_layout()
        plt.show()
        if write:
            fig.savefig(f'results_{counter}.png')
        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Analyze regression results", )
    parser.add_argument("--dir", default='regression', help="directory containing run results, or regression")
    parser.add_argument("--write", "-w", action='store_true', help="write .png file(s) with the images")
    parser.add_argument("--single-run", "-s", action='store_true',
                        help="'dir' points to single run instead of regression")

    args = parser.parse_args()
    analyze(args.dir, args.single_run, args.write)
