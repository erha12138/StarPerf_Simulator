'''

Author : yunanhou

Date : 2023/08/24

Function : This file defines the satellite class satellite, which is the base class for all satellites in this project.
           Determining the satellite's position in orbit requires parameters such as longitude, latitude, distance from
           the earth's surface, true periapsis angle, etc.

'''
import random

total_cachespace = random.randint(10, 50)

class satellite:
    def __init__(self , nu , orbit , true_satellite, total_cachespace = total_cachespace):
        # longitude (degree), because the satellite is constantly moving, there are many longitudes. Use the list type
        # to store all the longitudes of the satellite.
        self.longitude = [] # 这里经纬度是列表，所有时间的经纬度都获取到
        # latitude (degree), because the satellite is constantly moving, there are many latitudes. Use the list type
        # to store all the latitudes of the satellite.
        self.latitude = []
        # altitude (km), because the altitude is constantly moving, there are many altitudes. Use the list type
        # to store all the altitudes of the satellite.
        self.altitude = []
        # the current orbit of the satellite
        self.orbit = orbit
        # list type attribute, which stores the current satellite and which satellites have established ISL, stores
        # the ISL object
        self.ISL = []
        # True periapsis angle is a parameter that describes the position of an object in orbit. It represents the
        # angle of the object's position in orbit relative to the perigee. For different times, the value of the true
        # periapsis angle keeps changing as the object moves in its orbit.
        self.nu = nu
        # the id number of the satellite, which is the number of the satellite in the shell where it is located. If the
        # constellation has multiple shells, the id of each satellite is the number of the shell in which it is located.
        # Each shell is numbered starting from 1. The ID number is initially -1, and the user does not need to specify
        # it manually.
        self.id = -1  ## 我要找到轨迹的话，satellite id得设出来
        # real satellite object created with sgp4 and skyfield models
        self.true_satellite = true_satellite

        ## 每个卫星的缓存空间的描述
        self.total_cachespace = total_cachespace # 总共拥有的缓存内容
        self.now_cachedsapce = [] # 初始为0，在一个缓存管理决策的方法中更新，也是一个关于时间戳的list
        self.cache_content = [] # 初始缓存内容为空，看看要不要用，在外面的cache content类里面维护也可以，在外面写第二个循环，逻辑更好控制
        # 这是应该video chunk的id，那就存入二元组，(video_id,chunk_id)，每一杠时隙一个

        ## 更新内容时会有写入能耗，写入就是更新的内容时产生的能耗，
        ## 本质上应该减少更新频率来减少能耗
        self.energy_capacity = 10 ## 写入时会消耗能量，这是能量上限
        self.energy_per_time = [] # 也是时间戳的list
        # 需要每一时刻的能耗吗，不要
        self.cache_update_agent = None # 用这个直接等于
        ## 每一时刻做缓存决策，要不要先把卫星生成了，再在时间轴上做缓存决策的判断，
        ## 再修改哪些卫星的状态，因为我已经获取了每一时刻为用户工作的卫星了
    # 有VIDEO_TYPE中视频


    ## 我只针对video缓存这种业务，每个MEC中有专门用来做这个的缓存空间，且video都是成块的
    ## 用户请求是要考虑的，但是考虑的是之前的用户请求的分布，来进行决策，然后产生缓存结果，就是先产生结果，在缓存结果中，考虑用户的效用问题
    ## 缓存的目的是整体缓存容量
    ## 当然整体缓存决策算法当然要考虑效用，也就是考虑能耗等

    def cache_mananage(self, request, other_information): # 考虑xxx，xxx来实现缓存内容
        pass
    def cache_energy_consumption(self): # 考虑xxx，缓存读写要考虑能耗，主要是存入写入要考虑能耗
        pass

class satellite_pertime:
    def __init__(self, time_slot, satellite, shell):
        # longitude (degree), because the satellite is constantly moving, there are many longitudes. Use the list type
        # to store all the longitudes of the satellite.
        self.timeslot = time_slot
        self.longitude = satellite.longitude[time_slot] # 这里经纬度是值
        self.latitude =  satellite.latitude[time_slot]
        self.altitude = satellite.altitude[time_slot]
        self.ISL = []
        self.nu = satellite.nu
        self.shell = shell
        self.id = satellite.id  ## 我要找到轨迹的话，satellite id得设出来
        # real satellite object created with sgp4 and skyfield models
        ## 每个卫星的缓存空间的描述
        self.total_cachespace = satellite.total_cachespace # 总共拥有的缓存内容
        self.now_cachedsapce = 0 # 初始为0
        self.cache_content = [] # 初始缓存内容为空，看看要不要用，在外面的cache content类里面维护也可以，在外面写第二个循环，逻辑更好控制
        # 这是应该video chunk的id，那就存入二元组，(video_id,chunk_id)，每一时隙一个
        
        ## 更新内容时会有写入能耗，写入就是更新的内容时产生的能耗，
        ## 本质上应该减少更新频率来减少能耗
        self.energy_capacity = 10 ## 写入时会消耗能量，这是能量上限
        self.energy_per_time = 0 # 每一时刻的能耗
        # self.cache_update_agent = None # 用这个直接等于
        ## 每一时刻做缓存决策，要不要先把卫星生成了，再在时间轴上做缓存决策的判断

    def to_json(self):
        return {
            'timeslot': self.timeslot,
            'longitude': self.longitude,
            'latitude': self.latitude,
            'altitude': self.altitude,
            'ISL': self.ISL,
            'nu': self.nu,
            'shell': self.shell,
            'id': self.id,
            'total_cachespace': self.total_cachespace,
            'now_cachedsapce': self.now_cachedsapce,
            'cache_content': self.cache_content
            # 如果还有其他属性需要序列化，可以继续添加
        }

def get_new_satellite_list(visible_sattilates_in_all_shells_and_time, user_name, time_slot):
    new_list = {"server_with_GS":[], "server_without_GS":[]}
    shell_id = 0
    for shell in visible_sattilates_in_all_shells_and_time:
        for satellite in shell[time_slot]["server_with_GS"][user_name]:
            new_list["server_with_GS"].append(satellite_pertime(time_slot = time_slot,
                                              satellite = satellite,
                                              shell = shell_id).to_json())
        for satellite in shell[time_slot]["server_without_GS"][user_name]:
            new_list["server_without_GS"].append(satellite_pertime(time_slot = time_slot,
                                              satellite = satellite,
                                              shell = shell_id).to_json())
        shell_id += 1
    return new_list
            

