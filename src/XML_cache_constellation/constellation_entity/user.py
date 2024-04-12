'''

Author : yunanhou

Date : 2023/11/07

Function : This file defines the user terminal class, the source endpoint and destination endpoint that need to be
           specified in communication, and these endpoints are the instantiation objects of this class. User terminals
           are all located on the surface of the earth, and the longitude and latitude must be passed in when
           instantiating this class.

'''

import random

class user:
    def __init__(self , longitude, latitude , user_name=None):
        self.user_name = user_name # the name of user
        self.longitude = longitude # the longitude of user
        self.latitude = latitude # the latitude of user

## 给定用户的经纬度范围，保证在同一片区域
def generate_users(num_users=10, start_longitude = 60, start_latitude = 70, end_longitude = 40, end_latitude = 50):
    users = []
    for i in range(num_users):
        # Generate random longitude and latitude within a reasonable range
        # Here we assume the range is between -180 to 180 for longitude and -90 to 90 for latitude
        longitude = random.uniform(start_longitude, end_longitude)
        latitude = random.uniform(start_latitude, end_latitude)
        
        # Create a new user instance
        new_user = user(longitude, latitude, i)
        users.append(new_user)

    
    return users