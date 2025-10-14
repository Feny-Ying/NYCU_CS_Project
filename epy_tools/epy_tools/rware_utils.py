from typing import List

import numpy as np


def print_self_info(obs):
    x, y = obs[:2]
    self_carrying = obs[2]
    facing = obs[3:7]
    highway = obs[7]
    print(f'x: {x}, y: {y}, self_carrying: {self_carrying}, facing: {facing}, highway: {highway}')


def print_one_grid_info(obs, grid_idx, sight, head_len, grid_len):
    agent_existing = obs[head_len + grid_idx * grid_len]
    agent_direction = obs[head_len + grid_idx * grid_len + 1:head_len + grid_idx * grid_len + 5]
    shelf_existing = obs[head_len + grid_idx * grid_len + 5]
    shelf_requested = obs[head_len + grid_idx * grid_len + 6]
    print(f'[ Grid {grid_idx} ]')
    print(f'agent_existing: {agent_existing}, agent_direction: {agent_direction}, '
          f'shelf_existing: {shelf_existing}, shelf_requested: {shelf_requested}')


def obs_parser(obs, sight):
    print('---------------------- Observation Parsing ----------------------')
    print_self_info(obs)

    #
    HEAD_LEN = 8  # Each obs contains these elements no matter the sight
    #
    n_grids = (2 * sight + 1) ** 2
    #
    GRID_LEN = 7  # Each grid contains these elements
    #
    for i in range(n_grids):
        # agent_existing = obs[HEAD_LEN + i * GRID_LEN]
        # agent_direction = obs[HEAD_LEN + i * GRID_LEN + 1:HEAD_LEN + i * GRID_LEN + 5]
        # shelf_existing = obs[HEAD_LEN + i * GRID_LEN + 5]
        # shelf_requested = obs[HEAD_LEN + i * GRID_LEN + 6]
        # print(f'=== Grid {i} ===')
        # print(f'agent_existing: {agent_existing}, agent_direction: {agent_direction}, '
        #       f'shelf_existing: {shelf_existing}, shelf_requested: {shelf_requested}')
        print_one_grid_info(obs, i, sight, HEAD_LEN, GRID_LEN)
    # Others
    print(f'=== Others ===')
    print(obs[HEAD_LEN + n_grids * GRID_LEN:])
    # Assume all the rest are zeros
    assert np.all(obs[HEAD_LEN + n_grids * GRID_LEN:] == 0)

def obs_parser_with_one_select_sight_info(obs, sight):
    print('---------------------- Observation Parsing ----------------------')
    print_self_info(obs)

    #
    HEAD_LEN = 8  # Each obs contains these elements no matter the sight
    #
    n_grids = (2 * sight + 1) ** 2
    #
    GRID_LEN = 7  # Each grid contains these elements
    #
    for i in range(n_grids):
        # agent_existing = obs[HEAD_LEN + i * GRID_LEN]
        # agent_direction = obs[HEAD_LEN + i * GRID_LEN + 1:HEAD_LEN + i * GRID_LEN + 5]
        # shelf_existing = obs[HEAD_LEN + i * GRID_LEN + 5]
        # shelf_requested = obs[HEAD_LEN + i * GRID_LEN + 6]
        # print(f'=== Grid {i} ===')
        # print(f'agent_existing: {agent_existing}, agent_direction: {agent_direction}, '
        #       f'shelf_existing: {shelf_existing}, shelf_requested: {shelf_requested}')
        print_one_grid_info(obs, i, sight, HEAD_LEN, GRID_LEN)
    # Others
    print(f'=== Others ===')
    print(obs[HEAD_LEN + n_grids * GRID_LEN:])
    print(f'last (where cur select sight) -> {obs[-1]}')
    # Assume all the rest are zeros
    assert np.all(obs[HEAD_LEN + n_grids * GRID_LEN:-1] == 0)


def show_diff(obs1, obs2, sight):
    """像 ``obs_parser`` 一樣分區比較，但是比較兩個 observation 的不同"""
    assert obs1.shape == obs2.shape
    #
    n_grids = (2 * sight + 1) ** 2
    #
    GRID_LEN = 7  # Each grid contains these elements
    #
    for i in range(obs1.shape[0]):
        print(f'================= Agent {i}')
        #
        for j in range(8):
            if obs1[i, j] != obs2[i, j]:
                print(f'obs1[{i}, {j}]: {obs1[i, j]}')
                print_self_info(obs1[i])
                print(f'obs2[{i}, {j}]: {obs2[i, j]}')
                print_self_info(obs2[i])

        #
        for j in range(n_grids):
            for k in range(GRID_LEN):
                if obs1[i, 8 + j * GRID_LEN + k] != obs2[i, 8 + j * GRID_LEN + k]:
                    # Show grid index
                    print(f'Grid {j}')
                    print(f'=== obs1[{i}, {8 + j * GRID_LEN + k}]: {obs1[i, 8 + j * GRID_LEN + k]}')
                    print_one_grid_info(obs1[i], j, sight, 8, GRID_LEN)
                    print(f'=== obs2[{i}, {8 + j * GRID_LEN + k}]: {obs2[i, 8 + j * GRID_LEN + k]}')
                    print_one_grid_info(obs2[i], j, sight, 8, GRID_LEN)


def rware_preprocess(original_sight: int,
                     new_sight: int,
                     original_obs_list: List[np.ndarray],
                     verbose: bool = False):
    """每個obs有3個部分：前面的個人資訊（8），中間的視野（(2*original_sight+1)^2*7），後面的未使用部分"""

    if verbose:
        print('~~~~~~~~~~~~~~~~~~~~~ Before Preprocess ~~~~~~~~~~~~~~~~~~~~~')
        obs_parser(original_obs_list[0], sight=original_sight)

    #
    total_len = original_obs_list[0].shape[0]
    unused_len = total_len - 8 - (2 * original_sight + 1) ** 2 * 7

    grid_size = 2 * original_sight + 1
    new_grid_size = 2 * new_sight + 1

    # 計算新的視野範圍的起始和結束索引
    start_idx = original_sight - new_sight
    end_idx = original_sight + new_sight + 1

    # 將 tuple 中的數據轉換為一個大 array
    original_obs_array = np.array(original_obs_list)
    n_agents = len(original_obs_list)

    # 提取每個 agent 的自我資訊
    agent_info_array = original_obs_array[:, :8]

    # 提取每個 agent 的觀察部分
    obs_array = original_obs_array[:, 8: -unused_len]
    obs_array = obs_array.reshape((n_agents, grid_size, grid_size, 7))

    # 初始化所有新的觀察，並設置為全零
    new_obs_array = np.zeros((n_agents, grid_size, grid_size, 7))

    # 計算應複製的區域
    copy_start = (grid_size - new_grid_size) // 2
    copy_end = copy_start + new_grid_size

    # 將原始觀察的可見部分複製到新的觀察中
    new_obs_array[:, copy_start:copy_end, copy_start:copy_end, :] = obs_array[
                                                                    :,
                                                                    copy_start:copy_end,
                                                                    copy_start:copy_end, :]

    # 將新的觀察重塑回一維陣列
    new_obs_flat_array = new_obs_array.reshape((n_agents, -1))

    # 合併每個 agent 的自我資訊和更新後的觀察數據並加回unused部分
    final_obs_array = np.hstack((agent_info_array, new_obs_flat_array))

    # 建個和原本 size 一樣的 list of arrays 預設都是 0
    processed_obs = np.zeros_like(original_obs_list)

    # ``final_obs_array`` 含有前和中部分，將後部分加回去
    processed_obs[:, :final_obs_array.shape[1]] = final_obs_array

    if verbose:
        print(f'copy_start: {copy_start}, copy_end: {copy_end}')
        print('~~~~~~~~~~~~~~~~~~~~~ After Preprocess ~~~~~~~~~~~~~~~~~~~~~')
        obs_parser(processed_obs[0], sight=original_sight)

    return processed_obs
