"""
給一資料夾路徑 f，
其下第一層為名稱命名方式為 `${alg}_${sight}`，
第二層為 `${env_id}`。
`env_id` 包含了 "-?s-" 的字串，其中 `?` 是一個數字。

寫一個程式，將所有第二層的資料夾的名稱改成符合其第一層資料夾的名稱的 ?s
例如：
"iql_6s/lbforaging:Foraging-15s-15x15-1p-5f-v1" -> "iql_6s/lbforaging:Foraging-6s-15x15-1p-5f-v1"

所以就是讓第二層的 sight 符合第一層的 sight。
在此例中，第一層的 sight 是 6s，所以第二層的 sight 要改成 6s。
也就是 "-15s-" 要改成 "-6s-"

"""
import os

FOLDER = "projects/LBF/sacred/"


def main():
    # 獲取所有子資料夾
    first_level_dirs = [d for d in os.listdir(FOLDER) if os.path.isdir(os.path.join(FOLDER, d))]

    for first_level_dir in first_level_dirs:
        # 尋找 `??s` 格式的子資料夾
        alg, sight = first_level_dir.split('_')
        if str.isdigit(sight[:-1]) and sight[-1] == 's':
            # 擷取 `??s` 部分
            sight_str = sight
            first_level_path = os.path.join(FOLDER, first_level_dir)
            second_level_dirs = [d for d in os.listdir(first_level_path) if
                                 os.path.isdir(os.path.join(first_level_path, d))]
            for second_level_dir in second_level_dirs:
                if 'lbforaging:' in second_level_dir:
                    assert second_level_dir.split('-')[1][-1] == 's', (
                        f"{second_level_dir} is not a sight, path: {os.path.join(first_level_path, second_level_dir)}")
                    second_level_sight = second_level_dir.split('-')[1]
                    # Modify the sight
                    new_second_level_dir = second_level_dir.replace(second_level_sight, sight_str)
                    print(f"Renaming {second_level_dir} to {new_second_level_dir}")
                    os.rename(os.path.join(first_level_path, second_level_dir),
                              os.path.join(first_level_path, new_second_level_dir))


if __name__ == "__main__":
    main()
