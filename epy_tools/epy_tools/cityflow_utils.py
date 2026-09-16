import numpy as np
from collections import deque
from typing import Dict, List, Union


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


from typing import Dict, List, Union
import numpy as np
'''
# [phase, waiting]
def cityflow_topo_preprocess(
    original_obs,
    use_dsr: bool,
    sight: int,
    per_node_dim: int = 3,
    keep_hop: bool = False
):
    #print("original_obs:", original_obs)

    if not use_dsr:
        return original_obs
    
    if isinstance(original_obs, np.ndarray):
        obs_list = list(original_obs)
    else:
        obs_list = original_obs

    max_obs_len = obs_list[0].shape[0]
    max_nodes = max_obs_len // per_node_dim

    max_obs_len = max_obs_len // per_node_dim * 2  # 調整為 phase, waiting, vehicle_count, avg_speed
    #print("max_obs_len:", max_obs_len)

    processed = []

    for obs in obs_list:
        nodes = obs.reshape(max_nodes, per_node_dim)
        #print("nodes shape:", nodes.shape)
        #print("nodes:", nodes)
        kept_nodes = []
        for node in nodes:
            hop = node[2]
            if hop == 0 or hop <= sight:
                kept_nodes.append(node[:2])  # phase, waiting, vehicle_count, avg_speed
            else:
                kept_nodes.append([0.0, 0.0])

        flat = np.array(kept_nodes, dtype=obs.dtype).reshape(-1)

        if flat.shape[0] < max_obs_len:
            pad = np.zeros(max_obs_len - flat.shape[0], dtype=obs.dtype)
            flat = np.concatenate([flat, pad])

        processed.append(flat)
    #print("processed:", processed)
    return processed
'''
'''
# [phase, waiting, hop]
def cityflow_topo_preprocess(
    original_obs,
    use_dsr: bool,
    sight: int,
    per_node_dim: int = 3,
    keep_hop: bool = True
):
    if not use_dsr:
        return original_obs

    if isinstance(original_obs, np.ndarray):
        obs_list = list(original_obs)
    else:
        obs_list = original_obs

    max_obs_len = obs_list[0].shape[0]
    max_nodes = max_obs_len // per_node_dim

    processed = []

    for obs in obs_list:
        nodes = obs.reshape(max_nodes, per_node_dim)

        new_nodes = []
        for node in nodes:
            hop = node[2]
            if hop == 0 or hop <= sight:
                new_nodes.append(node)
            else:
                new_nodes.append([0.0, 0.0, 0.0])

        flat = np.array(new_nodes, dtype=obs.dtype).reshape(-1)
        processed.append(flat)

    return processed
'''
# 0s=10 vehicles, 1s=15 vehicles, 2s=20 vehicles, 3s=25 vehicles
# def cityflow_topo_preprocess(
#     original_obs,
#     use_dsr: bool,
#     sight: int,
#     per_node_dim: int = 3,
#     keep_hop: bool = False
# ):

#     vehicle_count_mapping = {0: 10, 1: 15, 2: 20, 3: 30, 4: 50}  # 4代表超過sight的車道，統一視為200輛

#     if not use_dsr:
#         return original_obs

#     if isinstance(original_obs, np.ndarray):
#         obs_list = list(original_obs)
#     else:
#         obs_list = original_obs
#     #print("obs_list:", obs_list)
#     processed = []

#     max_vehicle = vehicle_count_mapping[sight]

#     for obs in obs_list:
#         obs = np.array(obs)

#         for i in range(len(obs)):
#             if obs[i] > max_vehicle:
#                 obs[i] = max_vehicle

#         processed.append(obs)
#     #print("processed:", processed)
#     return processed


def cityflow_topo_preprocess(original_obs, use_dsr: bool, sight, sight_buckets=[25,50,75,100]):
    processed_obs = []
    
    # 確保 sight 是 list 格式，如果只有單一值則擴展成跟 original_obs 一樣長
    if not isinstance(sight, list):
        sights = [sight] * len(original_obs)
    else:
        sights = sight
    
    # if sight[0] != sight[1]:
    #     print("sight:", sight)

    # 使用 enumerate 同時取得 agent 的索引 i 與其對應的觀測值 obs
    for i, obs in enumerate(original_obs):

        # 取得該 agent 專屬的 bucket 索引
        # 假設傳進來的 sight 是具體距離(如 100)，要轉成 bucket index (如 2)
        # 或者如果你傳進來的就是索引 0, 1, 2，就直接使用
        current_sight = sights[i]
        
        # 判定 bucket_idx：如果傳入的是 100 且 buckets 是 [50,75,100]，則 idx 為 2
        if current_sight in sight_buckets:
            bucket_idx = sight_buckets.index(current_sight)
        else:
            bucket_idx = current_sight # 假設已經是索引了

        new_obs = [obs[0]]  # 第一個維度通常是紅綠燈相位
        lane_count_start = 1
        
        # 計算每個車道有多少個距離層級的特徵
        num_lanes = (len(obs) - 1) // len(sight_buckets)

        for lane_i in range(num_lanes):
            # 根據該 Agent 選擇的距離層級 (bucket_idx)，提取對應位置的特徵
            # 公式：起始點 + 車道偏移 + 距離層偏移
            idx = lane_count_start + lane_i * len(sight_buckets) + bucket_idx
            
            if idx < len(obs):
                new_obs.append(obs[idx])
            else:
                # 防呆機制：若索引超出，補 0
                new_obs.append(0)
        
        processed_obs.append(new_obs)
        
    return processed_obs


