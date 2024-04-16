import xml.etree.ElementTree as ET
import src.XML_cache_constellation.constellation_entity.ground_station as GS
# from src.XML_cache_constellation.constellation_entity.user import user, generate_users
import math

# The Tile class is a class used to represent a certain area on the earth's surface. The area is represented by four latitude and longitude coordinates.
# Tile类是用来表示地球表面某个区域的类。 该区域由四个纬度和经度坐标表示
class Tile:
    def __init__(self, longitude_start, longitude_end, latitude_start, latitude_end):
        self.longitude_start = longitude_start
        self.longitude_end = longitude_end
        self.latitude_start = latitude_start
        self.latitude_end = latitude_end

# Read xml document
def xml_to_dict(element):
    if len(element) == 0:
        return element.text
    result = {}
    for child in element:
        child_data = xml_to_dict(child)
        if child.tag in result:
            if type(result[child.tag]) is list:
                result[child.tag].append(child_data)
            else:
                result[child.tag] = [result[child.tag], child_data]
        else:
            result[child.tag] = child_data
    return result

# Read xml document
def read_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return {root.tag: xml_to_dict(root)}



# Function : given a longitude and latitude coordinate position on the earth's surface, find all the satellites that can
#            be seen at the position at time t, and return a list set composed of these satellites.
# Parameters:
# user : the user, which is a User class object
# t : the t-th timeslot
# sh : the shell of the constellation
# minimum_elevation : the minimum elevation angle of the ground observation point, in degrees
def user_visible_all_satellites(user , t , sh , minimum_elevation):
    # calculate the coordinates of the user in the three-dimensional Cartesian coordinate system
    user_x,user_y,user_z = latilong_to_descartes(user , "User")
    # define a list to represent the collection of all satellites that the user can see.
    user_visible_all_satellites_list = []
    # traverse all satellites in sh
    for orbit in sh.orbits:
        for satellite in orbit.satellites:
            # calculate the coordinates of the satellite satellite in the three-dimensional Cartesian
            # coordinate system at time t
            sat_x,sat_y,sat_z = latilong_to_descartes(satellite , "satellite" , t)
            # determine whether satellite and user are visible. If visible, add satellite to
            # user_visible_all_satellites_list
            if judgePointToSatellite(sat_x,sat_y,sat_z , user_x,user_y,user_z , minimum_elevation):
                user_visible_all_satellites_list.append(satellite)
    return user_visible_all_satellites_list  # 这里直接拿到该用户可见的所有卫星



# Function : convert the latitude and longitude coordinates of ground GS/POP points/user terminals/satellites into
#            three-dimensional Cartesian coordinates
# Parameters:
# transformed_object : the GS class object/satellite class object that needs to be converted
# object_type : the type of the parameter transformed_object. The type of object_type is a string. The value of the
#               string is "GS" or "satellite" or "POP" or "User".
# t : the timeslot number, starting from 1. Among them, when object_type is "GS", this parameter is invalid. When
#     object_type is "satellite", this parameter represents the tth timeslot of the satellite.
# Return value : the x, y, and z coordinates of the converted GS, and xyz are all in meters.
def latilong_to_descartes(transformed_object , object_type , t=None):
    a = 6371000.0
    e2 = 0.00669438002290
    if object_type == "satellite":
        longitude = math.radians(transformed_object.longitude[t])
        latitude = math.radians(transformed_object.latitude[t])
        fac1 = 1 - e2 * math.sin(latitude) * math.sin(latitude)
        N = a / math.sqrt(fac1)
        # the unit of satellite height above the ground is meters
        h = transformed_object.altitude[t] * 1000
        X = (N + h) * math.cos(latitude) * math.cos(longitude)
        Y = (N + h) * math.cos(latitude) * math.sin(longitude)
        Z = (N * (1 - e2) + h) * math.sin(latitude)
        return X, Y, Z
    else:
        longitude = math.radians(transformed_object.longitude)
        latitude = math.radians(transformed_object.latitude)
        fac1 = 1 - e2 * math.sin(latitude) * math.sin(latitude)
        N = a / math.sqrt(fac1)
        h = 0  # GS height from the ground, unit is meters
        X = (N + h) * math.cos(latitude) * math.cos(longitude)
        Y = (N + h) * math.cos(latitude) * math.sin(longitude)
        Z = (N * (1 - e2) + h) * math.sin(latitude)
        return X, Y, Z


# Function : given a point on land (user, POP or GS, etc.) and the coordinates of a satellite in the three-dimensional
#            Cartesian system, determine whether the point on land can see the satellite.
# Parameters:
# sat_x, sat_y, sat_z : respectively represent the xyz coordinates of the satellite in the three-dimensional Cartesian
#                       coordinate system
# point_x, point_y, point_z : respectively represent the xyz coordinates of points on land in the three-dimensional
#                             Cartesian coordinate system
# minimum_elevation : the minimum elevation angle at which a point on land can see the satellite
# Return value : Returning True means it can be seen, False means it can't be seen.
# Basic idea: Calculate the vector from the ground point to the satellite and the vector from the ground point to the
#             center of the earth respectively, and then calculate the angle between the two vectors. If the angle is
#             greater than or equal to (90°+minimum_elevation), it means it is visible, otherwise it means it is
#             invisible.
def judgePointToSatellite(sat_x , sat_y , sat_z , point_x , point_y , point_z , minimum_elevation):
    A = 1.0 * point_x * (point_x - sat_x) + point_y * (point_y - sat_y) + point_z * (point_z - sat_z)
    B = 1.0 * math.sqrt(point_x * point_x + point_y * point_y + point_z * point_z)
    C = 1.0 * math.sqrt(math.pow(sat_x - point_x, 2) + math.pow(sat_y - point_y, 2) + math.pow(sat_z - point_z, 2))
    angle = math.degrees(math.acos(A / (B * C))) # find angles and convert radians to degrees
    if angle < 90 + minimum_elevation or math.fabs(angle - 90 - minimum_elevation) <= 1e-6:
        return False
    else:
        return True

# Function : Determine whether among all the satellites in the satellites collection, there is at least one satellite
#            with at least 1 GS within its visible range. If so, return true, otherwise return false
# Parameters:
# satellites : the set of all satellites within the visible range of the user at time t
# t : the t-th timeslot
# GSs : a collection of ground base stations, each element of which is a ground_station class object
# minimum_elevation : the minimum elevation angle of the ground observation point, in degrees
def judge_user_coveraged(satellites , t , GSs , minimum_elevation):
    for sat in satellites:
        # calculate the coordinates of satellite sat in the three-dimensional Cartesian coordinate system
        sat_x, sat_y, sat_z = latilong_to_descartes(sat, "satellite", t)
        # traverse all base stations in GSs and use sat to determine whether they are visible one by one.
        for gs in GSs:
            # calculate the coordinates of base station gs in the three-dimensional Cartesian coordinate system
            gs_x,gs_y,gs_z = latilong_to_descartes(gs , "GS")
            # determine whether the satellite sat can see gs, if so, return true
            if judgePointToSatellite(sat_x, sat_y, sat_z , gs_x,gs_y,gs_z , minimum_elevation):
                return True
            
            ## 每个用户对应一些可用卫星，每个可用卫星会对应一些GS，
            ## 每个时刻都有这么一个值，是不是应该由用户去对GS合理一点，因为用户是不动的
            ## 这里是可以把GS的信息拿到的，但是会不会有点冗杂
            ## 不在这里算，这里算完是全局的，所有timeslot以及所用用户都有了，没必要，最后拿到用户和卫星再去拿这个代码去计算吧
    return False

def judge_satellite_and_GS_coveraged(satellites_list , t , GSs , minimum_elevation):
    new_satellite_list_with_server = []
    new_satellite_list_without_server = []
    with_server_flag = 0
    for sat in satellites_list:
        # calculate the coordinates of satellite sat in the three-dimensional Cartesian coordinate system
        sat_x, sat_y, sat_z = latilong_to_descartes(sat, "satellite", t)
        # traverse all base stations in GSs and use sat to determine whether they are visible one by one.
        for gs in GSs:
            # calculate the coordinates of base station gs in the three-dimensional Cartesian coordinate system
            gs_x,gs_y,gs_z = latilong_to_descartes(gs , "GS")
            
            if judgePointToSatellite(sat_x, sat_y, sat_z , gs_x,gs_y,gs_z , minimum_elevation):
                ## 如果有一个GS，就可以了
                with_server_flag = 1
                new_satellite_list_with_server.append(sat)
                break
            else:
                pass
                # 如果是False，说明当前这个卫星不能为这个用户提供服务，则不加入新的卫星列表，
        # 不可以更新缓存，但原有缓存还存在着，是可以提供服务的，这颗卫星还是可以为用户提供服务的，用自己原有的缓存
        if not with_server_flag:
            new_satellite_list_without_server.append(sat)
        with_server_flag = 0
    return new_satellite_list_with_server, new_satellite_list_without_server







# Function : Calculate the coverage of the constellation in bent-pipe mode
# Parameters:
# dT : how often to record a timeslot
# sh : a shell class object, representing a shell in the constellation
# ground_station_file: the data file of the satellite constellation ground base station GS (given in the form
#                      of path + file name)
# minimum_elevation : the minimum elevation angle of the ground observation point, in degrees
# tile_size : the size of the square block on the earth's surface. For example, the default is to cut every 10°,
#             that is, each block occupies 10° longitude and 10° latitude.
# 当前时刻还是全部时刻，全部时刻的覆盖都可以直接计算出来
def get_current_coverage_from_shell(tt, sh, ground_station_file, users, minimum_elevation = 25):
    # Tiles = []
    # for lon in range(-180, 180, tile_size):   # 先分地球上的tiles
    #     for lat in range(-90, 90, tile_size):
    #         tile = Tile(lon, lon + tile_size, lat, lat + tile_size)
    #         Tiles.append(tile)
    
    # read ground base station data
    ground_station = read_xml_file(ground_station_file)
    # generate GS
    GSs = []
    for gs_count in range(1, len(ground_station['GSs']) + 1, 1):
        gs = GS.ground_station(longitude=float(ground_station['GSs']['GS' + str(gs_count)]['Longitude']),
                                latitude=float(ground_station['GSs']['GS' + str(gs_count)]['Latitude']),
                                description=ground_station['GSs']['GS' + str(gs_count)]['Description'],
                                frequency=ground_station['GSs']['GS' + str(gs_count)]['Frequency'],
                                antenna_count=int(ground_station['GSs']['GS' + str(gs_count)]['Antenna_Count']),
                                uplink_GHz=float(ground_station['GSs']['GS' + str(gs_count)]['Uplink_Ghz']),
                                downlink_GHz=float(ground_station['GSs']['GS' + str(gs_count)]['Downlink_Ghz']))
        GSs.append(gs)    ## 有多少个GS都导入进去了
        
        
    # define a list to represent the constellation coverage of each timeslot
    # 什么时候、并且是哪几号卫星，覆盖到这个区域
    coverage_all_timeslot = []
    for t in range(0,tt,1):  # 对的上，就从0开始吧
        coverage_current_timeslot = {"server_with_GS":[],"server_without_GS":[]}

        # define a list to represent the set of tiles that can be covered by the constellation at time t
        users_visible_sattilates_with_GS_per_timeslot = {}
        users_visible_sattilates_without_GS_per_timeslot = {}
        for user in users:
            ## 拿到这个用户的当前时刻的可见卫星
            user_visible_all_satellites_list = user_visible_all_satellites(user, t, sh, minimum_elevation)
            ## 现在来遍历 user_visible_all_satellites_list，看其中的卫星是否可以看见GS，看不见的删掉，剩下的是真的能为用户提供服务的用户
            lis11, list2 = judge_satellite_and_GS_coveraged(user_visible_all_satellites_list , t , GSs , minimum_elevation)
            users_visible_sattilates_with_GS_per_timeslot[user.user_name] = lis11
            users_visible_sattilates_without_GS_per_timeslot[user.user_name] = list2
            # if judge_user_coveraged(user_visible_all_satellites_list , t , GSs , minimum_elevation): 
            ## 这里的逻辑是有卫星能看见地面站就返回，但不意味着当前用户可见的所有卫星都可以看见地面站，这里得改

            ## 当前用户的可见卫星需要考虑这附近是否可见GS，有GS才可以使用，是不是可以找到有几个GS
            # 拿不到用户ID，很多用户拿不到GS，说明动态性很强，要不要画一张点图
        coverage_current_timeslot["server_with_GS"] = users_visible_sattilates_with_GS_per_timeslot
        coverage_current_timeslot["server_without_GS"] = users_visible_sattilates_without_GS_per_timeslot
        coverage_all_timeslot.append(coverage_current_timeslot)
    # 返回每个 timeslot 每个用户 的可视卫星，以及其 GS 状态
    # 考虑确认每个ID，用字典的形式，别用列表，看不懂，每个satellite已经有ID了
    # 每一个时刻，存着每一个用户可以见的卫星，可能有用户不能看见被GS服务的卫星
    return coverage_all_timeslot
