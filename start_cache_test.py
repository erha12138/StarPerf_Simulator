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

USER_NUM = 10 # 用户数量
VIDEO_TYPE = 10 # 视频类型

USER_POSITION_RANGE = [-73.9,-36.4,-63.9,-26.4] # 开始经度，开始维度，结束经度，结束纬度

def main():
    print("\t\033[31mStarting XML Constellations with cache Running...\033[0m")

    print("\t\t\033[31mRunning(01) : constellation generation\033[0m")
    constellation = constellation_configuration.constellation_configuration(dT=dT, 
                                                                            constellation_name=constellation_name)
    time_len = len(constellation.shells[0].orbits[00].satellites[00].altitude) # 有几个时间戳

    print("\t\t\033[31mfinish(01) : constellation generation success\033[0m")


    print("\t\t\033[31mRunning(02) : Video list generation\033[0m")
    ## 我需要先生成video_list
    from src.XML_cache_constellation.constellation_entity.content import Request, Video, generate_requests_set_for_per_user
    video_list = [Video(id=video_id, num_chunks=19, chunk_size=1) for video_id in range(VIDEO_TYPE)]  # size单位为50mbps，
    print("\t\t\033[31mfinish(02) : Video list generation success\033[0m")

    print("\t\t\033[31mRunning(03) : User list generation\033[0m")
    # 生成了用户列表，users_set[0] 表示有几个用户，其中包含了用户的地理位置坐标，
    # user.request是个列表，包含了每个时刻的用户请求，目前设计的对同一个VIDEO顺序请求
    from src.XML_cache_constellation.constellation_entity.user import user, generate_users
    users_set = generate_users(num_users = USER_NUM, 
                               user_position_range = USER_POSITION_RANGE, 
                               time_len = time_len,
                               video_list = video_list) 
    print("\t\t\033[31mfinish(03) : User list generation success\033[0m")

    ## 请求的集合，可以直接用来作为外部环境输入算法
    

    
    print("\t\t\033[31mRunning(04) : get visible sattilates by users\033[0m")
    ## 找到每个用户当前可以请求到的卫星id，时间戳哪去了？？？
    visible_sattilates_in_all_shells_and_time = []
    for shell in constellation.shells:
        visible_sattilates_in_all_shells_and_time.append(get_current_coverage_from_shell(tt=time_len,
                                                                                         sh=shell,
                                                                                         ground_station_file=ground_station_file, 
                                                                               users= users_set))
    print("\t\t\033[31mfinish(04) : get visible sattilates by users success\033[0m")


    # 下面要加入每个卫星与地面基站的能耗考虑
    # 可以每个直接获取下一时刻的用户内容，

    ## 要不要在这里开始用t来做，visible_sattilates_in_all_shells_and_time已经把需要用的卫星都记录下来了
    ## 注意卫星的结构问题：最底层的卫星经纬度是已经包含了所有时间戳的列表
    ## visible_sattilates_in_all_shells_and_time shell 下面就马上到了时间戳层
    ## 每个卫星类中只记录存储空间最大容量与最大能耗值

    ## 具体的存储内容在外卖用一个cache content类去维护，在外面把用时间戳把每一个不一样的奇奇怪怪的结构统一起来
    ## 每个时间戳有个缓存内容，也包含了能耗  吧
    cache_content_in_timeslot = [] 


    print("end test")

## 每个卫星除了轨迹位置，还需要缓存信息，加在satellite的entity中  OK

## 轨迹就在constellation里，每次都要计算生成，看能不能直接冲 h5里面直接读取  give up

## 先把轨迹与用户位置的对应关系可视化，我要获取是否有覆盖，哪个更近，也与groundstation有关 OK

## 针对VIDEO的用户请求 OK

## 为每个卫星加入cache空间与能耗的描述，并想清楚他们的关系，每个卫星可以收到用户请求 trying



## 先把 delay 的group加上去，看看 原本加入的position group怎么用，position里面有shell，shell里面有对应的轨迹和卫星位置数据集
## h5里面存了position position里存了shell信息，shell里面存了每个卫星的信息，每个shell有，

# 我有下一时刻的轨迹，要不直接可以预测来做

if __name__ == "__main__":
    main()
