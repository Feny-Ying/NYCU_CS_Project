"""
適用於 Col3, Ing7, Dayuan 的 rou file
用於計算一個 rou file 裡有幾輛車/多少比例經過幾個"幹道路口區間" (main segment)
幹道路口區間的定義是: 完全穿越一組相鄰的幹道路口間的路段。
"""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from tqdm import tqdm

#
AVAILABLE_ENVS = ['Col3', 'Ing7', 'Dayuan']

COL3_SEGMENTS = [
    ['241660955#0', '241660955#4', '241660955#6', '241660955#7'],  # South to North 1
    ['241660955#10', '241660955#11', '241660955#13', '241660955#14'],  # South to North 2
    ['-241660955#16', '-241660955#13', '-241660955#12', '-241660955#10', ],  # North to South 1
    ['-241660955#9', '-241660955#6', '-241660955#5', '-241660955#3', ],  # North to South 2
]

ING7_OUTER_EDGE_CORRESPONDING_INTERSECTION_NUMBER = {
    # 這個字典是用來對應外環線的路段對應到哪個路口(North to South 1~7)
    # 有.5的是中間的路口
    # 最後的差是取無條件進位到整數
    '124812856#0': 7,
    '201956810': 7,
    '201956820': 7,
    '-173169611#0': 7,
    '124812856#1': 7,
    '10425609#0': 6,
    '201956811#0': 6,
    '653473569#5': 5,
    '-653473569#5': 5,
    '25145012#7': 4.5,
    '-104010328': 4.5,
    '104010475#0': 4.5,
    '104010354': 4.5,
    '27920078#0': 4,
    '202070434#2': 4,
    '202070434#0': 4,
    '24599188#0.94': 3.5,
    '-83304175#2': 3.5,
    '22716549#0': 3.5,
    '-22716549#6': 3.5,
    '-32124744': 3.5,
    '-201089423#1': 3.5,
    '-32124743': 3.5,
    '-201089423#2': 3.5,
    '201089423#2': 3.5,
    '32124743': 3.5,
    '32124744': 3.5,
    '285716192#0': 3.5,
    '24693977#1': 3,
    '-24693977#1': 3,
    '37386279': 2.5,
    '-37386279': 2.5,
    '-24634415': 2.5,
    '24634415': 2.5,
    '-24634414#4': 2.5,
    '24634414#4': 2.5,
    '-24634414#5': 2.5,
    '24634414#5': 2.5,
    '32999110#0': 2.5,
    '-32999434#1': 2.5,
    '25149219#1': 2.5,
    '201963537#1': 2.5,
    '-315358253#1': 2,
    '315358253#1': 2,
    '315358253#2': 2,
    '201956821#0': 2,
    '-24608846#1': 1.5,
    '24608846#1': 1.5,
    '402600768#1': 1.5,
    '51857517#0': 1.5,
    '168702040#1': 1.5,
    '51857517#1': 1.5,
    '32978638#0': 1,
    '-32978638#0': 1,
    '-24608844': 1,
    '24608844': 1,
    '-266565295#5': 1,
    '266565295#5': 1,
    '51857516#1': 1,
    '32021112#0': 1,
    '32124637#1': 1,
    '51857518#1': 1,
    '32999435': 1,
}


# "General-SUMO-RL/nets/dayuan/vehroute_dayuan_20210406_noMotor_truckTo2Cars.rou.xml"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, required=True, help='Environment name', choices=AVAILABLE_ENVS)
    parser.add_argument('-r', '--rou', type=str, required=True, help='Path to the rou file')
    args = parser.parse_args()
    return args


def parse_rou_file(env, rou_file):
    tree = ET.parse(rou_file)
    root = tree.getroot()

    if env == 'Col3':
        n_vehicle = len(root.findall('vehicle'))

        n_veh_to_main_segments = defaultdict(int)

        for vehicle in root.findall('vehicle'):
            route = vehicle.find('route')
            edges = route.get('edges')

            # See if the vehicle is in the main segment
            n_main_segments = 0
            for main_segment in COL3_SEGMENTS:
                if all(edge in edges for edge in main_segment):
                    n_main_segments += 1

            if n_main_segments > 2:
                # Print details of the vehicle
                print('-' * 50)
                print(f'Vehicle {vehicle.get("id")}:')
                print(f'Route: {edges}')
                print(f'Number of main segments: {n_main_segments}')
            n_veh_to_main_segments[n_main_segments] += 1

    elif env == 'Ing7':
        n_vehicle = len(root.findall('trip'))
        n_veh_to_main_segments = defaultdict(int)
        for trip in tqdm(root.findall('trip')):
            from_edge = trip.get('from')
            to_edge = trip.get('to')

            from_number = ING7_OUTER_EDGE_CORRESPONDING_INTERSECTION_NUMBER[from_edge]
            to_number = ING7_OUTER_EDGE_CORRESPONDING_INTERSECTION_NUMBER[to_edge]

            if to_number >= from_number:
                diff = np.floor(np.floor(to_number) - from_number)
            else:
                diff = np.floor(np.floor(8 - to_number) - (8 - from_number))

            # Get rid of the decimal part
            n_main_segments = max(0, int(diff))

            n_veh_to_main_segments[n_main_segments] += 1

    elif env == 'Dayuan':
        # 大園的 link 表的很清楚，直接用名稱上面 link_{from}_{to}
        # 起點看 to，終點看 from
        n_vehicle = len(root.findall('vehicle'))

        n_veh_to_main_segments = defaultdict(int)

        for vehicle in root.findall('vehicle'):
            route = vehicle.find('route')
            edges = route.get('edges')

            edge_names = edges.strip().split(' ')
            from_no = int(edge_names[0].split('_')[-1])
            to_no = int(edge_names[-1].split('_')[-2])
            diff = abs(to_no - from_no)
            print('---------------------------------')
            print(f'Vehicle {vehicle.get("id")}:')
            print(f'Route: {edges}')
            print(f'from: {from_no}, to: {to_no}, diff: {diff}')

            n_main_segments = diff
            n_veh_to_main_segments[n_main_segments] += 1

    else:
        raise NotImplementedError(f'Environment {env} is not supported')
    print('-' * 50)
    print()

    # Sort by number of main segments
    n_veh_to_main_segments = dict(sorted(n_veh_to_main_segments.items()))

    print(f'Number of vehicles: {n_vehicle}')
    print('Number of vehicles to main segments:')
    for n_main_segments, n_veh in n_veh_to_main_segments.items():
        print(f'{n_main_segments} main segments: {n_veh} vehicles')

    assert sum(
        n_veh_to_main_segments.values()) == n_vehicle, 'The sum of vehicles in each main segment is not equal to the total number of vehicles.'

    if env == 'Ing7':
        print('注意，for Ing7，只是大概的數字，因為edge太多有一些標的不完全正確，尤其是在路口與路口間。'
              '還有Ing7 floor那邊的公式高機率也有點問題。手動標number到頭昏眼花了。')


def main():
    args = get_args()
    env = args.env
    rou_file = args.rou
    parse_rou_file(env, rou_file)


if __name__ == '__main__':
    main()
