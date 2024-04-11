import h5py
import src.XML_cache_constellation.constellation_entity.constellation as CONSTELLATION
from src.XML_cache_constellation.constellation_entity.shell import shell
import numpy as np

# import sys

# sys.path.append()

def read_constellation_from_h5(h5_file_path, constellation_name):
    # 使用h5py库读取HDF5文件
    with h5py.File(h5_file_path, 'r') as file:
        for key in file.keys():
            print("key:", key)
        # 读取position组
        position = file['position']
        
        # 初始化一个列表，用于存储所有shell的实例
        all_shells = []
        
        # 遍历position组中的每个shell
        for shell_name in position.keys():
            current_shell_group = position[shell_name]

        # read dataset
            timeslots = [ds for ds in current_shell_group if ds.startswith('timeslot')]
            sorted_timeslots = sorted(timeslots, key=lambda s: int(s.split('timeslot')[1]))
            for timeslot in sorted_timeslots:
                satellite_positions = current_shell_group[timeslot].value
                for position in satellite_positions:

            
            
            
            # 读取当前shell的轨道数据
            orbits_data = []
            for orbit_index in range(shell_group["timeslot1"]['number_of_orbit']):
                orbit_data = shell_group[f'orbit{orbit_index + 1}']
                orbits_data.append(orbit_data)
                
            # 创建shell实例
            shell_instance = shell(
                altitude=int(shell_group['altitude']),
                number_of_satellites=int(shell_group['number_of_satellites']),
                number_of_orbits=int(shell_group['number_of_orbit']),
                inclination=float(shell_group['inclination']),
                orbit_cycle=int(shell_group['orbit_cycle']),
                phase_shift=int(shell_group['phase_shift']),
                shell_name=shell_name
            )
            
            # 将当前shell实例添加到列表中
            all_shells.append(shell_instance)
        
        # 创建并返回星座对象
        target_constellation = CONSTELLATION(
            constellation_name=constellation_name,
            number_of_shells=len(all_shells),
            shells=all_shells
        )
        
        return target_constellation

if __name__ == "__main__":
# 调用函数，传入H5文件路径和星座名称
    h5_path = "data/XML_constellation/Starlink.h5"
    constellation_name = "Starlink"
    constellation = read_constellation_from_h5(h5_path, constellation_name)