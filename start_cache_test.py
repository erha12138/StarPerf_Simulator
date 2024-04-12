import src.XML_cache_constellation.cache_constellation_generation.constellation_configuration as constellation_configuration
from kits.get_h5file_satellite_position_data import get_h5file_satellite_position_data
from kits.get_h5file_tree_structure import print_hdf5_structure
import h5py
from src.XML_cache_constellation.function.get_coverage import get_current_coverage_from_shell

# 现在就用xml，tle可以下载，但是要自己输入每个shell的轨迹数量，现在不好说每个轨道是用什么轨迹数量比较合适
# 就用每个constellation_configuration，来生成这一个周期的位置信息
dT = 200  # 时间间隔
constellation_name = "Starlink"
h5_path = 'data/XML_constellation/Starlink.h5'
ground_station_file = 'config/ground_stations/Starlink.xml'
def main():
    print("start cache")
    constellation = constellation_configuration.constellation_configuration(dT=dT, 
                                                                            constellation_name=constellation_name)
    timeslots = len(constellation.shells[0].orbits[00].satellites[00].altitude) # 有几个时间戳
    
#    """ 
#     生成了用户位置信息，考虑了，先给了默认值，10个用户位置
#     num_users=10, 
#     start_longitude = 60, 
#     start_latitude = 70, 
#     end_longitude = 40, 
#     end_latitude = 50
#     """
    from src.XML_cache_constellation.constellation_entity.user import user, generate_users
    users = generate_users() 
    visible_sattilates_in_all_shells_and_time = []
    for shell in constellation.shells:
        visible_sattilates_in_all_shells_and_time.append(get_current_coverage_from_shell(tt=timeslots,sh=shell,ground_station_file= \
                                        ground_station_file, users= users))
    
    print("end test")

## 每个卫星除了轨迹位置，还需要缓存信息，加在satellite的entity中  OK

## 轨迹就在constellation里，每次都要计算生成，看能不能直接冲 h5里面直接读取  give up

## 先把轨迹与用户位置的对应关系可视化，我要获取是否有覆盖，哪个更近，也与groundstation有关 trying

## 先把 delay 的group加上去，看看 原本加入的position group怎么用，position里面有shell，shell里面有对应的轨迹和卫星位置数据集
## h5里面存了position position里存了shell信息，shell里面存了每个卫星的信息，每个shell有，

if __name__ == "__main__":
    main()
