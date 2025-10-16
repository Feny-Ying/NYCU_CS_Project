"""
在跑完一些實驗後，這個程式可以用來畫圖，看看訓練曲線。

--color-group 是用來畫圖時，用不同的顏色來區分不同的 group。
例如，做grid_search時，我們有一堆組合，我今天想看reset_freq的影響，我就 --color-group reset_freq，這樣的話
相同reset_freq的會用相同的顏色來畫。若沒給 --color-group，則全部會用不同的顏色來畫。
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-dir', '--file_dir', type=str,
                        help='The directory containing the results', dest='file_dir', required=True)

    # The variables
    parser.add_argument('--color-group', default=None, type=str, help='The variable to be used as color',
                        required=False, dest='color_group')
    parser.add_argument('--var', nargs='+', type=str, help='The variables to be compared',
                        required=True, dest='var')

    # Data Checking

    # E.g.,
    #       --cut-after 2050000 --n-test 41 (we expect ~2000000, 50000 is a period)
    #                     => after cutting, 41 in total

    parser.add_argument('--cut-after', type=int, help='Cut the test results after this number of steps',
                        required=True, dest='cut_after')
    parser.add_argument('--n-test', type=int, help='Each experiment should have n-test test results',
                        required=True, dest='n_test')
    parser.add_argument('--start-step', type=int, help='Start step', required=True, dest='start_step')
    parser.add_argument('--interval', type=int, help='Interval', required=True, dest='interval')

    args = parser.parse_args()

    file_dir = args.file_dir
    variables = args.var
    n_var = len(variables)
    assert n_var > 0, 'No variables to compare'

    # Get all the results
    # all_x = defaultdict(list)
    all_y = defaultdict(list)
    for exp_dir in os.listdir(file_dir):
        exp_dir_path = os.path.join(file_dir, exp_dir)
        if exp_dir.startswith('_'):
            continue
        if not os.path.isdir(exp_dir_path):
            continue
        files = os.listdir(exp_dir_path)
        assert 'metrics.json' in files, f'{exp_dir} does not have metrics.json'
        assert 'config.json' in files, f'{exp_dir} does not have config.json'
        metric_path = os.path.join(exp_dir_path, 'metrics.json')
        config_path = os.path.join(exp_dir_path, 'config.json')
        metric_data = json.load(open(metric_path, 'r'))
        config_data = json.load(open(config_path, 'r'))

        test_return_step_to_mean = {step: mean for step, mean in zip(metric_data['test_return_mean']['steps'],
                                                                     metric_data['test_return_mean']['values'])}

        # Cut
        for step in list(test_return_step_to_mean.keys()):
            if step > args.cut_after:
                test_return_step_to_mean.pop(step)

        assert len(test_return_step_to_mean) == args.n_test, (
            f'{exp_dir} does not have {args.n_test} test results; '
            f'got {len(test_return_step_to_mean)}\n'
            f'; its steps: {test_return_step_to_mean.keys()}')

        #
        combination = []
        for var in variables:
            combination.append(config_data[var])
        combination = tuple(combination)

        # all_x[combination].append(list(test_return_step_to_mean.keys()))
        all_y[combination].append(list(test_return_step_to_mean.values()))

        # if args.color_group is not None:
        #     all_param_values.append(config_data[args.color_group])
    print(all_y.keys())
    # Mean over seeds
    all_y_over_seeds = {}
    all_y_over_seeds_95ci = {}
    for combination in all_y.keys():
        # # all_x[combination]: a list of lists
        # # All these lists must be the same
        # for x in all_x[combination]:
        #     assert x == all_x[combination][0], ('All x must be the same, but got different: \n'
        #                                         f'x = {x}\n'
        #                                         f'all_x[combination][0] = {all_x[combination][0]}')

        # Compute avg y over seeds
        y_mean_over_seeds = np.mean(np.array(all_y[combination]), axis=0)

        # all_x_over_seeds[combination] = all_x[combination][0]
        all_y_over_seeds[combination] = y_mean_over_seeds

        sem = stats.sem(np.array(all_y[combination]), axis=0)
        ci = stats.t.interval(0.95, len(all_y[combination]) - 1, loc=y_mean_over_seeds, scale=sem)
        all_y_over_seeds_95ci[combination] = ci

    # If color_group is not None
    if args.color_group is not None:
        color_group_var_idx = variables.index(args.color_group)
        color_group_values = set([combination[color_group_var_idx] for combination in all_y_over_seeds.keys()])
        color_group_values = list(color_group_values)
        # Not > 20 since using tab20
        assert len(color_group_values) <= 20, 'Too many groups'
        # Each with a different color
        color_group_values = {color_group_value: i for i, color_group_value in enumerate(color_group_values)}
        color_group_values = {combination: color_group_values[combination[color_group_var_idx]] for combination in
                              all_y_over_seeds.keys()}
    else:
        color_group_values = None

    # Plot
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 9))
    x = np.arange(args.start_step, args.cut_after, args.interval)
    for combination in all_y_over_seeds.keys():
        if args.color_group is not None:
            plt.plot(x, all_y_over_seeds[combination], label=f'{combination}',
                     color=plt.cm.tab20(color_group_values[combination]))
            plt.fill_between(x,
                             all_y_over_seeds_95ci[combination][0],
                             all_y_over_seeds_95ci[combination][1],
                             color=plt.cm.tab20(color_group_values[combination]), alpha=0.2)
        else:
            plt.plot(x, all_y_over_seeds[combination], label=f'{combination}')
            plt.fill_between(x,
                             all_y_over_seeds_95ci[combination][0],
                             all_y_over_seeds_95ci[combination][1],
                             alpha=0.2)
    plt.legend()
    plt.show()
