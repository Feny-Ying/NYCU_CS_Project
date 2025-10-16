from copy import deepcopy
from typing import List

import numpy as np

N_DATA_PER_LANE = 5

# # By alphabetical order
# ING7_ID_TO_SLICE = {
#     'gneJ210': slice(0 * N_DATA_PER_LANE, 7 * N_DATA_PER_LANE),
#     'gneJ260': slice(7 * N_DATA_PER_LANE, 12 * N_DATA_PER_LANE),
#     '32564122': slice(12 * N_DATA_PER_LANE, 24 * N_DATA_PER_LANE),
#     'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190': slice(
#         24 * N_DATA_PER_LANE, 33 * N_DATA_PER_LANE),
#     'gneJ207': slice(33 * N_DATA_PER_LANE, 40 * N_DATA_PER_LANE),
#     'gneJ143': slice(40 * N_DATA_PER_LANE, 50 * N_DATA_PER_LANE),
#     'cluster_1757124350_1757124352': slice(50 * N_DATA_PER_LANE, 58 * N_DATA_PER_LANE),
# }

# By the order of the agents in the env from north to south
ING7_ID_TO_SLICE = {
    'gneJ210': slice(0 * N_DATA_PER_LANE, 10 * N_DATA_PER_LANE),
    'gneJ260': slice(10 * N_DATA_PER_LANE, 18 * N_DATA_PER_LANE),
    '32564122': slice(18 * N_DATA_PER_LANE, 25 * N_DATA_PER_LANE),
    'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190': slice(
        25 * N_DATA_PER_LANE, 37 * N_DATA_PER_LANE),
    'gneJ207': slice(37 * N_DATA_PER_LANE, 44 * N_DATA_PER_LANE),
    'gneJ143': slice(44 * N_DATA_PER_LANE, 53 * N_DATA_PER_LANE),
    'cluster_1757124350_1757124352': slice(53 * N_DATA_PER_LANE, 58 * N_DATA_PER_LANE),
}

COL3_ID_TO_SLICE = {
    '360082': slice(0 * N_DATA_PER_LANE, 5 * N_DATA_PER_LANE),
    '360086': slice(5 * N_DATA_PER_LANE, 11 * N_DATA_PER_LANE),
    'GS_cluster_2415878664_254486231_359566_359576': slice(11 * N_DATA_PER_LANE, 19 * N_DATA_PER_LANE),
}

ENV_TO_SLICE_DICT = {
    'ingolstadt7': ING7_ID_TO_SLICE,
    'ingolstadt7r0.7': ING7_ID_TO_SLICE,
    'ingolstadt7r0.35': ING7_ID_TO_SLICE,
    'cologne3': COL3_ID_TO_SLICE,
    'cologne3fs5': COL3_ID_TO_SLICE,
    'cologne3fs5m': COL3_ID_TO_SLICE,
    'cologne3fs10': COL3_ID_TO_SLICE,
    'cologne3fs10m': COL3_ID_TO_SLICE,
}


def resco_preprocess(original_sight: int,
                     new_sight: int,
                     original_obs: List[np.ndarray],
                     n_agents: int,
                     args,
                     verbose: bool = False) -> List[np.ndarray]:
    """
    假設： agent 順序是由北到南

    =========
    處理的方式
    =========

    照順序來，我們知道當下是哪個 agent (traffic light) 的 obs。
    然後就能定位出這個 agent 的位置，以及這個 agent 的視野範圍。
    1.

    :param original_sight: Original sight of the agents in the original env (not used actually)
    :param new_sight: New sight of the agents
    :param original_obs: List of np.ndarray, each np.ndarray is the obs of an agent
    :param n_agents: Number of agents in the env; it's the number of traffic lights too
    :param args: Exp args. Used to know the map name
    :param verbose: Whether to print the debug info
    """
    # print(f'original_sight: {original_sight}, new_sight: {new_sight}, n_agents: {n_agents}')
    # # set precision
    # np.set_printoptions(precision=3)
    # #
    # print(f'-------------- original')
    # for i, obs in enumerate(original_obs):
    #     print(f'agent {i}')
    #     print(obs)

    if verbose:
        print(f'original_sight: {original_sight}, new_sight: {new_sight}, n_agents: {n_agents}')
        print(f'-------------- original')
        for i, obs in enumerate(original_obs):
            print(f'agent {i}')
            print(obs)

    resco_map_name = args.env_args['key'].split(':')[-1].split('-')[0]

    assert resco_map_name in [
        'ingolstadt7', 'ingolstadt7r0.7', 'ingolstadt7r0.35',
        'cologne3', 'cologne3fs5', 'cologne3fs10', 'cologne3fs5m', 'cologne3fs10m'
    ]

    # For each agent of traffic light, just put a mask to make non-visible area to be -1
    preprocessed_obs = []
    for i_ag, ag_obs in enumerate(original_obs):
        # print(f'----------- agent {i_ag} (original_sight: {original_sight}, new_sight: {new_sight})')
        mask = np.zeros_like(ag_obs, dtype=bool)
        for i_tl, tl in enumerate(ENV_TO_SLICE_DICT[resco_map_name].keys()):
            if np.abs(i_tl - i_ag) <= new_sight - 1:
                # print(f'agent {i_ag}, [{i_tl}] tl {tl} is visible')
                mask[ENV_TO_SLICE_DICT[resco_map_name][tl]] = True
            else:
                # print(f'agent {i_ag}, [{i_tl}] tl {tl} is not visible')
                mask[ENV_TO_SLICE_DICT[resco_map_name][tl]] = False
        # print(f'agent {i_ag}, mask: \n{mask}')
        # Mask out the non-visible area -> 0
        # 20240605 Bug fixed: since no deepcopy, the original obs will be modified
        preprocessed_obs.append(deepcopy(ag_obs) * mask)

    # print('-------------- preprocessed')
    # for i, obs in enumerate(preprocessed_obs):
    #     print(f'agent {i}')
    #     print(obs)

    return preprocessed_obs
