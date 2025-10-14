import numpy as np
from collections import deque
from typing import Dict, List


'''
使用範例
# 假設你已經有 CityFlow engine
intersection_ids = engine.get_intersection_list()
adjacency = build_adjacency(engine, intersection_ids)

# original_obs 是 dict[intersection_id] -> np.ndarray
# 假設每個路口 obs 是 [num_lanes, 3]，包含 [車輛數量, 平均等待時間, 紅綠燈狀態]
original_obs = {}
for inter_id in intersection_ids:
    lanes = engine.get_lane_vehicle_count(inter_id)  # 假設返回每條車道車輛數
    # 假資料: [num_lanes, 3]
    original_obs[inter_id] = np.array([[count, 0, 1] for count in lanes])

# 取得每個 agent 的 obs，sight=2 表示可看到二層鄰居
agent_obs_list = cityflow_topo_preprocess(original_obs, adjacency, sight=2)

# 現在 agent_obs_list 可以直接輸入到 DSR network
for idx, obs in enumerate(agent_obs_list):
    print(f"Agent {intersection_ids[idx]} obs shape: {obs.shape}")

'''

def build_adjacency(engine, intersection_ids: List[str]) -> Dict[str, List[str]]:
    """
    建立路口鄰接表
    engine: CityFlow engine
    intersection_ids: 所有路口 ID
    return: dict[intersection_id] -> list of neighbor intersection_ids
    """
    adjacency = {i: [] for i in intersection_ids}
    for road_id in engine.get_road_list():
        start = engine.get_road_property(road_id, "startIntersection")
        end = engine.get_road_property(road_id, "endIntersection")
        if start in adjacency:
            adjacency[start].append(end)
        if end in adjacency:
            adjacency[end].append(start)  # 若雙向
    return adjacency


def get_visible_intersections(agent_id: str, adjacency: Dict[str, List[str]], sight: int) -> List[str]:
    """
    BFS 計算 agent_id 視野內的 intersection IDs
    sight: BFS 層數
    """
    visited = set()
    queue = deque([(agent_id, 0)])
    visible = set()

    while queue:
        current, dist = queue.popleft()
        if dist > sight:
            continue
        if current not in visited:
            visited.add(current)
            visible.add(current)
            for neighbor in adjacency.get(current, []):
                queue.append((neighbor, dist + 1))
    return list(visible)


def cityflow_topo_preprocess(original_obs: Dict[str, np.ndarray],
                             adjacency: Dict[str, List[str]],
                             sight: int) -> List[np.ndarray]:
    """
    將 CityFlow 路口觀察轉成 DSR 可用的 list of obs
    original_obs: dict[intersection_id] -> np.ndarray (每個路口的觀察矩陣)
    adjacency: dict[intersection_id] -> list of neighbor intersections
    sight: BFS 層數
    return: List[np.ndarray], 每個 agent 一個 obs
    """
    obs_list = []
    for agent_id in original_obs.keys():
        visible_ids = get_visible_intersections(agent_id, adjacency, sight)
        # 合併可見路口的 obs
        obs_agent = [original_obs[vid] for vid in visible_ids if vid in original_obs]
        if len(obs_agent) == 0:
            # 若視野內沒有任何路口，用空矩陣填充
            obs_list.append(np.zeros_like(next(iter(original_obs.values()))))
        else:
            obs_list.append(np.vstack(obs_agent))
    return obs_list
