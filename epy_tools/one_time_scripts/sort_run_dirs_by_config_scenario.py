import os
import json
import shutil
from collections import defaultdict

# ★ 常常 for CAMA 用

# 基本目錄路徑，根據實際情況修改

# base_dir = "/home/ppo/Downloads/checkSacred_20241013142630_result_1011_lbf_4p4fCoop_UCB_Ablation/sacred/qmix DSR w/lbforaging:Foraging-10s-10x10-4p-4f-coop-v1"
# new_root = "/home/ppo/Downloads/checkSacred_20241013142630_result_1011_lbf_4p4fCoop_UCB_Ablation/sacred/w_files"

# base_dir = "/home/ppo/Downloads/1015_CAMA_testUnseenTrue/sacred/test_unseen icm_qmix_atten_nomi_a8"
# new_root = os.path.join(base_dir, 'split_by_scenario')

# base_dir = "/home/ppo/Downloads/CAMA__1112/test_unseen icm_qmix_atten_a8"
# new_root = os.path.join(base_dir, 'split_by_scenario')
# base_dir = "/home/ppo/Downloads/CAMA__1113/icm_qmix_atten_a8"
# new_root = os.path.join(base_dir, 'split_by_scenario')

# base_dir = "/home/ppo/Downloads/CAMA__1114/test_unseen dsr_icm_qmix_atten_nomi_a8"
# new_root = os.path.join(base_dir, 'split_by_scenario')
#
# base_dir = "/home/ppo/Downloads/CAMA__1114/icm_qmix_atten_nomi_a8"
# new_root = os.path.join(base_dir, 'split_by_scenario')

base_dir = "/home/ppo/Downloads/CAMA__1118/dsr_icm_qmix_atten"
new_root = os.path.join(base_dir, 'split_by_scenario')




# 用於從每個 config.json 中提取 'name' 字段
def get_name_from_config(run_dir):
    config_path = os.path.join(base_dir, run_dir, 'config.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)
        # return config_data.get('name')
        return config_data.get('scenario')


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
