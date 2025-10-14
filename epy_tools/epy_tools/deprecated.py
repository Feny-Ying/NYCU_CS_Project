from typing import Optional, List

import numpy as np


def lbf_preprocess_step_data_deprecated1(preprocess_samples: str, original_obs: List[np.ndarray],
                                         n_agents: int, n_fruits: Optional[int] = None, verbose: bool = False) -> List[
    np.ndarray]:
    if n_fruits is None:
        n_fruits = int(len(original_obs[0]) // 3) - n_agents
    if verbose:
        print(f'preprocess_samples: {preprocess_samples}')
    _split_by_arrow = preprocess_samples.split(':')[1].split('->')
    old_sight = int(_split_by_arrow[0][:-1])
    new_sight = int(_split_by_arrow[1][:-1])

    if old_sight == new_sight:
        return original_obs

    if verbose:
        print(f'old_sight: {old_sight}, new_sight: {new_sight}')

    # 預先建立新 observation 格式
    empty_obs = np.full((len(original_obs[0]),), -1, dtype=original_obs[0].dtype)
    empty_obs[2::3] = 0

    modified_obs = []
    for ag_obs in original_obs:
        # 直接用當前 agent 在視野窗內的位置 vs 物件在視野窗內的位置 來判斷如何調整
        # 這樣的話，就不用考慮視野窗的左/上是否卡在牆上的問題
        ag_y, ag_x = ag_obs[3 * n_fruits:3 * n_fruits + 2]
        obj_positions = ag_obs.reshape(-1, 3)[:, :2]
        # 判斷是否為視野卡左/上牆的狀況
        y_stuck_dist = old_sight - ag_y
        x_stuck_dist = old_sight - ag_x
        y_stuck = y_stuck_dist > 0
        x_stuck = x_stuck_dist > 0
        # 計算新的 y 和 x
        delta_y = old_sight - new_sight if not y_stuck else -min(y_stuck_dist - (old_sight - new_sight), 0)
        delta_x = old_sight - new_sight if not x_stuck else -min(x_stuck_dist - (old_sight - new_sight), 0)
        # 先做個 mask：舊視野原本看不到的，新視野也看不到(-1,-1的)
        # 那些 (-1,-1, ?) 的物標起來
        not_visible_old = (obj_positions[:, 0] == -1) & (obj_positions[:, 1] == -1)
        new_positions_y = obj_positions[:, 0] - delta_y
        new_positions_x = obj_positions[:, 1] - delta_x
        new_positions = np.column_stack((new_positions_y, new_positions_x))
        # 計算物件是否在新視野內
        is_fruit = np.arange(len(obj_positions)) < n_fruits
        new_ag_y, new_ag_x = ag_y - delta_y, ag_x - delta_x
        visible_fruit = is_fruit & (
                (new_ag_y - new_sight <= new_positions[:, 0]) & (new_positions[:, 0] <= new_ag_y + new_sight) &
                (new_ag_x - new_sight <= new_positions[:, 1]) & (
                        new_positions[:, 1] <= new_ag_x + new_sight)) & ~not_visible_old
        visible_agent = ~is_fruit & (0 <= new_positions[:, 0]) & (new_positions[:, 0] <= new_sight * 2) & \
                        (0 <= new_positions[:, 1]) & (new_positions[:, 1] <= new_sight * 2) & ~not_visible_old
        visible_mask = visible_fruit | visible_agent
        # 將新的位置和能見度存入 `new_ag_obs`
        new_ag_obs = empty_obs.copy()
        valid_indices = np.where(visible_mask)[0]
        is_fruit_sorted = is_fruit[valid_indices]
        fruit_indices = np.arange(np.sum(is_fruit_sorted)) * 3
        agent_indices = (np.arange(len(valid_indices) - np.sum(is_fruit_sorted)) * 3 + 3 * n_fruits)
        indices_to_put = np.hstack((fruit_indices, agent_indices))
        new_ag_obs[indices_to_put] = new_positions[:, 0][valid_indices].flatten()
        new_ag_obs[indices_to_put + 1] = new_positions[:, 1][valid_indices].flatten()
        new_ag_obs[indices_to_put + 2] = ag_obs.reshape(-1, 3)[valid_indices, 2]
        modified_obs.append(new_ag_obs)

    return modified_obs
