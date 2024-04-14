'''

Author : yunanhou

Date : 2023/08/24

Function : This file defines the satellite class satellite, which is the base class for all satellites in this project.
           Determining the satellite's position in orbit requires parameters such as longitude, latitude, distance from
           the earth's surface, true periapsis angle, etc.

'''
import random

total_cachespace = random.randint(0, 20)

class satellite:
    def __init__(self , nu , orbit , true_satellite, total_cachespace = total_cachespace):
        # longitude (degree), because the satellite is constantly moving, there are many longitudes. Use the list type
        # to store all the longitudes of the satellite.
        self.longitude = []
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

        ## 每个卫星需要 计算、缓存空间的描述
        self.total_cachespace = total_cachespace # 总共拥有的缓存内容
        self.cached = 0 # 初始为0，在一个缓存管理决策的方法中更新

    ## 我只针对video缓存这种业务，每个MEC中有专门用来做这个的缓存空间，且video都是成块的
    ## 用户请求是要考虑的，但是考虑的是之前的用户请求的分布，来进行决策，然后产生缓存结果，就是先产生结果，在缓存结果中，考虑用户的效用问题
    ## 缓存的目的是整体缓存容量
    ## 当然整体缓存决策算法当然要考虑效用，也就是考虑能耗等

    def cache_mananage(self, request, other_information): # 考虑xxx，xxx来实现缓存内容
        pass
    def cache_energy_consumption(self, ): # 考虑xxx，缓存读写要考虑能耗，主要是存入写入要考虑能耗
        pass

