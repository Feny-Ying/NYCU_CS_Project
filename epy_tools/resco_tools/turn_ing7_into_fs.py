"""
Find the vehicles that pass through equivalent or more than 2 main segments in cologne3, and copy them to 5 copies.

"""
import xml.etree.ElementTree as ET

import numpy as np

from resco_tools.check_main_segments import COL3_SEGMENTS

# Settings
MAINTAIN_TOTAL_NUM_VEH = True
N_COPIES = 5

# Default paths
COL3_ORIGINAL_ROU_PATH = 'envs/RESCO/resco_benchmark/environments/ingolstadt7/ingolstadt7.rou.xml'
DEST_PATH_TEMPLATE = 'envs/RESCO/resco_benchmark/environments/ingolstadt7fs{}{}/ingolstadt7.rou.xml'

# Set np random seed
np.random.seed(0)


def main():
    tree = ET.parse(COL3_ORIGINAL_ROU_PATH)
    root = tree.getroot()
    vehicles_to_copy = []

    for vehicle in root.findall('vehicle'):
        route = vehicle.find('route')
        edges = route.get('edges')

        n_main_segments = 0
        for main_segment in COL3_SEGMENTS:
            if all(edge in edges for edge in main_segment):
                n_main_segments += 1

        if n_main_segments >= 2:
            vehicles_to_copy.append(vehicle)

            for i_copy in range(N_COPIES - 1):
                new_vehicle = ET.Element('vehicle')
                new_vehicle.attrib = vehicle.attrib.copy()
                new_route = ET.Element('route')
                new_route.attrib = vehicle.find('route').attrib.copy()
                new_route.set('edges', vehicle.find('route').get('edges'))
                new_vehicle.set('id', f'{vehicle.get("id")}_{i_copy + 1}')
                new_vehicle.append(new_route)
                vehicles_to_copy.append(new_vehicle)
        else:
            # 有機率拿掉
            NUM_VEH_LESS_THAN_2_SEG = 4165
            NUM_VEH_2_SEG = 329
            prob_to_retain = 1 - ((N_COPIES - 1) * NUM_VEH_2_SEG / NUM_VEH_LESS_THAN_2_SEG)
            if np.random.rand() < prob_to_retain:
                vehicles_to_copy.append(vehicle)

    # 美觀的輸出
    # Make a new root
    new_root = ET.Element('routes')
    new_root.attrib = root.attrib

    # <vType id="pkw" vClass="passenger" speedDev="0.1" length="4.3" minGap="1.5"/> 加到 routes 下面
    vtype = ET.Element('vType')
    vtype.attrib = {'id': 'pkw', 'vClass': 'passenger', 'speedDev': '0.1', 'length': '4.3', 'minGap': '1.5'}
    new_root.append(vtype)

    #
    new_root.extend(vehicles_to_copy)

    ET.indent(new_root)
    new_tree = ET.ElementTree(new_root)
    save_dest_path = DEST_PATH_TEMPLATE.format(N_COPIES, 'm' if MAINTAIN_TOTAL_NUM_VEH else '')
    new_tree.write(save_dest_path)


if __name__ == '__main__':
    main()
