import os
import json
import shutil
from collections import defaultdict

# 基本目錄路徑，根據實際情況修改
# base_dir = "/home/ppo/Downloads/checkSacred_20241013142630_result_1011_lbf_4p4fCoop_UCB_Ablation/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241013142630_result_1011_lbf_4p4fCoop_UCB_Ablation/sacred/w_files"
# base_dir = "/home/ppo/Downloads/checkSacred_20241013163236_result_1011_asc2_QMIX_StateAblation_8mVs9m/sacred/qmix 999s noSightInfo concatObsAsState MaskNone StateLeaveObsNumber/8m_vs_9m"
# new_root = "/home/ppo/Downloads/checkSacred_20241013163236_result_1011_asc2_QMIX_StateAblation_8mVs9m/good"
# base_dir = "/home/ppo/Downloads/checkSacred_20241017021636_result_1016_asc2_QMIX_StateAblation_10mVs11m/sacred/qmix 999s noSightInfo concatObsAsState MaskNone StateLeaveObsNumber/10m_vs_11m"
# new_root = "/home/ppo/Downloads/checkSacred_20241017021636_result_1016_asc2_QMIX_StateAblation_10mVs11m/sacred/qmix 999s noSightInfo concatObsAsState MaskNone StateLeaveObsNumber/10m_vs_11m_split"

# base_dir = "/home/ppo/Downloads/checkSacred_20241103095424_result_1101_lbf_4p2fCoop_UCB_Ablation/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241103095424_result_1101_lbf_4p2fCoop_UCB_Ablation/sacred/sacred/qmix DSR c/split"

# base_dir = "/home/ppo/Downloads/checkSacred_20241104194027_result_1102_lbf_UCB_Ablation_extreme/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241104194027_result_1102_lbf_UCB_Ablation_extreme/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1/split"

# base_dir = "/home/ppo/Downloads/checkSacred_20241104194027_result_1102_lbf_UCB_Ablation_extreme/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241104194027_result_1102_lbf_UCB_Ablation_extreme/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1/split"

# base_dir = "/home/ppo/Downloads/checkSacred_20241112093321_result_1109_lbf_UCB_Ablation_basedOn1to10/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241112093321_result_1109_lbf_UCB_Ablation_basedOn1to10/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1/split"
# base_dir = "/home/ppo/Downloads/checkSacred_20241112093321_result_1109_lbf_UCB_Ablation_basedOn1to10/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241112093321_result_1109_lbf_UCB_Ablation_basedOn1to10/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1/split"
# base_dir = "/home/ppo/Downloads/checkSacred_20241112093321_result_1109_lbf_UCB_Ablation_basedOn1to10/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241112093321_result_1109_lbf_UCB_Ablation_basedOn1to10/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1/split"
# base_dir = "/home/ppo/Downloads/checkSacred_20241112093321_result_1109_lbf_UCB_Ablation_basedOn1to10/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241112093321_result_1109_lbf_UCB_Ablation_basedOn1to10/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1/split"

# base_dir = "/home/ppo/Downloads/checkSacred_20241112093456_result_1104_lbf_UCB_Ablation_longer1M/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241112093456_result_1104_lbf_UCB_Ablation_longer1M/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1/split"
# base_dir = "/home/ppo/Downloads/checkSacred_20241112093456_result_1104_lbf_UCB_Ablation_longer1M/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241112093456_result_1104_lbf_UCB_Ablation_longer1M/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1/split"
# base_dir = "/home/ppo/Downloads/checkSacred_20241112093456_result_1104_lbf_UCB_Ablation_longer1M/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241112093456_result_1104_lbf_UCB_Ablation_longer1M/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1/split"
# base_dir = "/home/ppo/Downloads/checkSacred_20241112093456_result_1104_lbf_UCB_Ablation_longer1M/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241112093456_result_1104_lbf_UCB_Ablation_longer1M/sacred/sacred/qmix DSR c/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1/split"


# base_dir = "/home/ppo/Downloads/checkSacred_20241116221056_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-2f-v1"
# base_dir = "/home/ppo/Downloads/checkSacred_20241116221056_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-6s-10x10-4p-2f-coop-v1"
# base_dir = "/home/ppo/Downloads/checkSacred_20241116221056_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-6s-10x10-4p-2f-v1"
# base_dir = "/home/ppo/Downloads/checkSacred_20241116221056_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-6s-10x10-4p-4f-coop-v1"


base_dir = "/home/ppo/Downloads/checkSacred_20241118173728_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-2f-v1"
base_dir = "/home/ppo/Downloads/checkSacred_20241118173728_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1"
base_dir = "/home/ppo/Downloads/checkSacred_20241118173728_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-2f-coop-v1"
base_dir = "/home/ppo/Downloads/checkSacred_20241118173728_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-6s-10x10-4p-2f-coop-v1"
base_dir = "/home/ppo/Downloads/checkSacred_20241118173728_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-6s-10x10-4p-2f-v1"
base_dir = "/home/ppo/Downloads/checkSacred_20241118173728_result_1113_lbf_UCB_Ablation_longer_s5000vsINF/sacred/sacred/qmix DSR w/lbforaging:Foraging-6s-10x10-4p-4f-coop-v1"


new_root = os.path.join(base_dir, 'split_by_name')
# 用於從每個 config.json 中提取 'name' 字段
def get_name_from_config(run_dir):
    config_path = os.path.join(base_dir, run_dir, 'config.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)
        return config_data.get('name')


# 創建一個 defaultdict 來存儲根據 name 分組的結果
runs_by_name = defaultdict(list)

# 遍歷目錄中的每個資料夾
for run_dir in os.listdir(base_dir):
    if run_dir.isdigit():  # 確保是數字資料夾
        name = get_name_from_config(run_dir)
        if name:
            runs_by_name[name].append(run_dir)

# 創建新的根目錄結構並複製資料夾
for name, dirs in runs_by_name.items():
    name_dir = os.path.join(new_root, name)  # 為每個 name 創建對應的目錄
    os.makedirs(name_dir, exist_ok=True)  # 如果目錄不存在則創建

    for run_dir in dirs:
        src_dir = os.path.join(base_dir, run_dir)
        dest_dir = os.path.join(name_dir, run_dir)
        shutil.copytree(src_dir, dest_dir)  # 複製 run 資料夾到新的位置

        print(f"Copied {src_dir} to {dest_dir}")

print("All directories copied successfully!")
