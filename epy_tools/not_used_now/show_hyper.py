"""
❯ tree
.
├── 1
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 10
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 11
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 12
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 13
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 14
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 15
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 16
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 17
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 18
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 19
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 2
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 20
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 21
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 22
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 23
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 24
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 25
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 26
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 27
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 28
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 29
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 3
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 30
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 4
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 5
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 6
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 7
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 8
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
├── 9
│   ├── config.json
│   ├── cout.txt
│   ├── info.json
│   ├── metrics.json
│   └── run.json
└── _sources
    ├── logging_7799699b534f806632c1d618a6ef7e02.py
    ├── main_8c7e54bd25ec6d93b5ad0a51a0b7dd44.py
    └── run_93f61c00b4dd5384cd409b7f630f35f7.py

31 directories, 153 files

```


 寫一個python script
input 為 hyper parameter (json key)(>=1個)
秀出所有 1, 2, 3, ... 資料夾的 config.json 該 input 的 value
"""
import argparse
import json
import os


def read_config(directory, keys):
    config_path = os.path.join(directory, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            result = {key: config.get(key, "Key not found") for key in keys}
            return result
    else:
        return None


def process_folders(root_dir, keys):
    # sort 先按字數排序，再按字母排序
    sorted_folders = sorted(os.listdir(root_dir), key=lambda x: (len(x), x))
    for folder_name in sorted_folders:
        if folder_name.startswith('_'):
            continue
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            config_values = read_config(folder_path, keys)
            if config_values:
                print(f"Folder {folder_name}: {config_values}")
            else:
                print(f"Folder {folder_name}: No config file found")

def main():
    parser = argparse.ArgumentParser(description="Process config files in subdirectories.")
    parser.add_argument("-f", "--root_dir", type=str, help="Root directory path")
    parser.add_argument("-hyper", "--hyperparameters", nargs="+", type=str,
                        help="Hyperparameters to retrieve from config files")
    args = parser.parse_args()
    process_folders(args.root_dir, args.hyperparameters)


if __name__ == "__main__":
    main()
