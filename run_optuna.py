import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import optuna
from optuna.trial import Trial

from attack import attack, directories_mingling
import evolution_eval

study_name = "patcher-params"
storage_name = f"sqlite:///{study_name}.db"

max_fail_rate = 0


def init_max():
    global max_fail_rate
    max_fail_rate = 0


def my_report(model_fail_rate):
    global max_fail_rate
    if model_fail_rate > max_fail_rate:
        max_fail_rate = model_fail_rate


evolution_eval.report = my_report


def objective(trial: Trial):
    colors = trial.suggest_categorical("colors", ['BLACK', 'DOMINANT', 'INVERSE'])
    fitness_kind = trial.suggest_categorical("fitness_kind", ["diff", "first", "second"])
    prop_or_full = trial.suggest_categorical("prop_or_full", ["prop", "full"])
    abs_prob = f'{fitness_kind}_{prop_or_full}'
    weights = trial.suggest_categorical('weights', ['LEGACY', 'SWAPPED', 'PROB_ONLY'])
    if weights == 'LEGACY':
        fail_weight = 0.7
        prob_weight = 0.3
    elif weights == 'SWAPPED':
        fail_weight = 0.3
        prob_weight = 0.7
    elif weights == 'PROB_ONLY':
        fail_weight = 0
        prob_weight = 1
    else:
        raise ValueError

    orig_dir = directories_mingling(trial.number, "runs_optuna")
    try:
        init_max()
        attack(single_image=False,
               creation_max_depth=4,
               population_size=500,
               num_of_evolve_threads=1,
               num_of_images_threads=4,
               max_generation=100,
               random_seed=1,
               patch_ratio_x=0.4,
               patch_ratio_y=0.4,
               elitism_rate=0,
               bloat_weight=0.0001,
               imagenet_path='/cs_storage/public_datasets/ImageNet',
               batch_size=100,
               num_of_images=40,
               classes=['freight car', 'passenger car', 'sports car', 'streetcar', ],
               threshold_size_ratio=0.1,
               threshold_confidence=0.8,
               fail_weight=fail_weight,
               prob_weight=prob_weight,
               abs_prob=abs_prob,
               colors=colors)
    except RuntimeError as e:
        print("Optuna trial failed with:")
        print(e)
        return float('nan')
    finally:
        os.chdir(orig_dir)

    return max_fail_rate


def run_experiment():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                direction="maximize")

    study.optimize(objective, n_trials=2)  # 100)

    print('Best trials:')
    for trial in study.best_trials:
        print(trial)

    print("\n\nSUMMARY:\n")
    print('Best model_fail_rate: ', end='')
    print(study.best_value)
    print('Best params: ', end='')
    print(study.best_params)


def show_results():
    raise NotImplementedError
    # visualisations from https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True,
                                directions=["minimize", "maximize"])

    print('Best trials:')
    for trial in study.best_trials:
        print(trial)

    optuna.visualization.matplotlib.plot_pareto_front(study, target_names=["queries", "asr"])
    optuna.visualization.matplotlib.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="queries"
    )
    optuna.visualization.matplotlib.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="asr"
    )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs experiments to find evolutionary hyperparameters")
    parser.add_argument('--results', '-r', action='store_true', help='Show experiments results')
    args = parser.parse_args()

    if args.results:
        show_results()
    else:
        run_experiment()
