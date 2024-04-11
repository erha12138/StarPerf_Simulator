import src.XML_cache_constellation.cache_constellation_generation.constellation_configuration as constellation_configuration
from src.XML_cache_constellation.cache_constellation_generation.constellation_generation_from_h5 import read_constellation_from_h5
from kits.get_h5file_satellite_position_data import get_h5file_satellite_position_data
from kits.get_h5file_tree_structure import print_hdf5_structure
import h5py


# 现在就用xml，tle可以下载，但是要自己输入每个shell的轨迹数量，现在不好说每个轨道是用什么轨迹数量比较合适
# 就用每个constellation_configuration，来生成这一个周期的位置信息
dT = 200  # 时间间隔
constellation_name = "Starlink"
h5_path = 'data/XML_constellation/Starlink.h5'

def main():
    print("start cache")
    constellation1 = constellation_configuration.constellation_configuration(dT=dT, 
                                                                            constellation_name=constellation_name)
    
    # with h5py.File(h5_path, 'r') as file:
    #     for key in file.keys():
    #         print("key:", key)
    #     # 读取position组
    #     position = file['position']    
    #     print_hdf5_structure(position)
    
    # get_h5file_satellite_position_data(h5_path)
    
    # constellation2 = read_constellation_from_h5(h5_path, constellation_name)
    print("end test")

## 每个卫星除了轨迹位置，还需要缓存信息，加在satellite的entity中

## 轨迹就在constellation里，每次都要计算生成，看能不能直接冲 h5里面直接读取

## h5里面存了position position里存了shell信息，shell里面存了每个卫星的信息，每个shell有，


## 先把轨迹与用户位置的对应关系可视化
## 先把 delay 的group加上去，看看 原本加入的position group怎么用，position里面有shell，shell里面有对应的轨迹和卫星位置数据集


if __name__ == "__main__":
    main()
