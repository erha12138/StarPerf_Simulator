'''

Author : yunanhou

Date : 2023/11/07

Function : This file defines the user terminal class, the source endpoint and destination endpoint that need to be
           specified in communication, and these endpoints are the instantiation objects of this class. User terminals
           are all located on the surface of the earth, and the longitude and latitude must be passed in when
           instantiating this class.

'''

import random
from .content import Request, Video, generate_requests_set_for_per_user
from tqdm import tqdm

class user:
    def __init__(self , longitude, latitude , user_name = None, user_request = None): 
        self.user_name = user_name # the name of user
        self.longitude = longitude # the longitude of user
        self.latitude = latitude # the latitude of user
        self.request = user_request # 用户请求直接把所有时隙得包含进去

## 给定用户的经纬度范围，保证在同一片区域
def generate_users(num_users=10, user_position_range = [60,52,70,62], time_len = 0, video_list = None): # 加拿大的经纬度
    request_set = generate_requests_set_for_per_user(user_num = num_users, 
                                                     time_len=time_len, 
                                                     video_list=video_list)  
    users = []
    for i in tqdm(range(num_users)):
        # Generate random longitude and latitude within a reasonable range
        # Here we assume the range is between -180 to 180 for longitude and -90 to 90 for latitude
        longitude = random.uniform(user_position_range[0], user_position_range[2])
        latitude = random.uniform(user_position_range[1], user_position_range[3])       
        # Create a new user instance
        new_user = user(longitude, latitude, i, user_request = request_set[i].request_list)
        users.append(new_user)

    return users