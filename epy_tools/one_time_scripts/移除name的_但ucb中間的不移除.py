import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str, required=True, help='Directory path to the results')
args = parser.parse_known_args()[0]

directory_path = args.dir

# 展開使用者的家目錄
directory_path = os.path.expanduser(directory_path)

# 遍歷目錄中的所有項目
for dir_name in os.listdir(directory_path):
    # 確保是目錄
    if os.path.isdir(os.path.join(directory_path, dir_name)):
        # 保留 "ucb(...)_c..._w..." 的 _ 字元，並替換其他 _
        parts = re.split(r'(ucb\(.*?\)_c\d+_w\d+)', dir_name)
        new_name = ''
        for part in parts:
            if re.match(r'ucb\(.*?\)_c\d+_w\d+', part):
                new_name += part  # 保留這部分不變
            else:
                new_name += part.replace('_', ' ')  # 替換其他部分的 "_"

        # 重命名目錄
        os.rename(os.path.join(directory_path, dir_name), os.path.join(directory_path, new_name))

print("重命名完成")
