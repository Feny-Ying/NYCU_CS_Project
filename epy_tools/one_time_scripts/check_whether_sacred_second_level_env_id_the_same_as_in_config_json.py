"""
給一資料夾路徑 f，
其下第一層為名稱命名方式為 `${alg}_${sight}`，
第二層為 `${env_id}`。
`env_id` 包含了 "-?s-" 的字串，其中 `?` 是一個數字。

第三層為 runs 的資料夾們。其含有 config.json 檔案，
讀起來，設它為變數 `config`。
`config['env_args']['key']` 是一個字串，指的是實際環境用的 `env_id`。

寫一個程式，檢查所有第三層資料夾下的 config.json，是否和第二層的 `env_id` 一致。
"""
import json
import os

FOLDER = "projects/LBF/sacred/"


def main():
    # 獲取所有子資料夾
    first_level_dirs = [d for d in os.listdir(FOLDER) if os.path.isdir(os.path.join(FOLDER, d))]

    for first_level_dir in first_level_dirs:
        print(f'first_level_dir: {first_level_dir}')
        second_level_dirs = [d for d in os.listdir(os.path.join(FOLDER, first_level_dir)) if
                             os.path.isdir(os.path.join(FOLDER, first_level_dir, d))]
        for second_level_dir in second_level_dirs:
            print(f'second_level_dir: {second_level_dir}')
            third_level_dirs = [d for d in os.listdir(os.path.join(FOLDER, first_level_dir, second_level_dir)) if
                                os.path.isdir(os.path.join(FOLDER, first_level_dir, second_level_dir, d))]
            for third_level_dir in third_level_dirs:
                if third_level_dir.startswith('_'):
                    continue
                print(f'third_level_dir: {third_level_dir}')
                config_path = os.path.join(FOLDER, first_level_dir, second_level_dir, third_level_dir, "config.json")
                with open(config_path, "r") as f:
                    config = json.load(f)
                    env_id = config["env_args"]["key"]
                    print(f'Real env_id: {env_id} and second_level_dir: {second_level_dir}')
                    assert env_id == second_level_dir, (f"env_id: {env_id} != second_level_dir: {second_level_dir}"
                                                        f" in {config_path}")
    print("All checks passed.")

if __name__ == "__main__":
    main()
