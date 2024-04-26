## single satellite，环境是单用户环境
## 先确定用户数量
## 再不定用户数量




    def update_buffer(self, throughput):
        # Assume each video chunk provides 10 seconds of play time
        video_chunk_play_time = self.chunk_duration
        
        # Calculate chunk size based on the current bitrate (in Megabits)
        chunk_size_megabits = self.current_bitrate * video_chunk_play_time
        
        # Calculate download time (in seconds) based on throughput (assumed to be in Megabits per second)
        download_time = chunk_size_megabits / throughput if throughput > 0 else float('inf')
        
        # Video consumption: decrease buffer by one second of play time
        self.buffer_size -= 2 if self.buffer_size > 0 else 0 # 现在假设2s发起一次请求
        
        # self.buffer_size = 
        # Buffer underflow check
        if self.buffer_size <= 0:
            self.buffer_size = 0  # Can't go negative, this would be rebuffering time
            download_time = 0  # If buffer is empty, play as soon as chunk is downloaded
            self.rebuffer_event = True # rebuffer事件在这个请求中是否可以
        
        # Add the play time of the chunk to the buffer, minus the download time
        # This could be zero or negative if the download time exceeds the chunk play time
        self.buffer_size += max(video_chunk_play_time - download_time, 0)
        
        # Cap the buffer size at the buffer capacity
        self.buffer_size = min(self.buffer_size, self.buffer_capacity)







import numpy as np
import gym
from gym import spaces
import random

def ABR(): # 模拟简单的ABR逻辑，并决策ABR的请求间隔
    pass

class Video:
    def __init__(self) -> None:
        self.chunk_size = random.randint(2, 5) # 视频长度2-5s
        self.bitrate_size = [640,1600,2400,4000,6000,8400] # kbps

class User:
    def __init__(self):
        self.video = Video() # 实例化
        self.request = 1 # 初始化请求等级，随时间t的变化而变化，是ABR算法的决策
        self.buffer = 0 # 初始化为0
        self.max_buffer = 20 # 最大缓存空间
        self.rebuffer_event = 0 # 记录rebuffer事件
        self.request_change = 0 # 码率的相减，可能做个归一化吧
        self.QoE = 0 # 后面在时间循环里去计算


class NOMAEnv(gym.Env):
    """
    A simple representation of a Non-Orthogonal Multiple Access (NOMA) environment.
    """
    
    def __init__(self, max_n_users, max_power):
        super(NOMAEnv, self).__init__()
        
        self.max_n_users = max_n_users
        # Define the number of users
        # self.n_users = n_users # 环境中有几个用户，可以不定用户数量，
        # 用户数量，20个时隙一变

        
        # Define action and observation space
        # Actions are the power allocated to each user
        # Observations may include QoE, buffer, etc. for each user (to be defined)
        self.action_space = spaces.Box(low = 0, high = max_power, dtype=np.float32) # 动作空间就是1，为每个智能体去设计动作
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_users, 4), dtype=np.float32) # 观测空间有哪些内容？
        # 为什么环境要写观测信息啊，可以在智能体的地方再写？，动作空间也是啊
        

        # Initialize state and power levels
        self.state = None
        self.reset()

    def reset(self):
        # Reset the state of the environment to an initial state
        # For example, the initial state could be randomly generated
        self.state = np.random.rand(self.n_users, 4)
        return self.state

    def step(self, action):
        # Execute one time step within the environment，接收到action后执行下一步动作
        self._take_action(action)
        self._update_state()
        
        reward = self._get_reward()
        done = False  # For now we'll say we're never done
        
        return self.state, reward, done, {}

    def _take_action(self, action):
        # Update the environment state based on the action.
        # This could involve updating the power levels and calculating the new QoE.
        pass

    def _update_state(self):
        # Update the observation space with new state
        # Calculate new QoE, buffer size etc.
        pass

    def _get_reward(self):
        # Calculate the reward from the current state.
        # This could involve QoE and power level.
        reward = 0
        return reward

    def render(self, mode='console'):
        # Render the environment to the screen
        if mode == 'console':
            print(f"Current state: {self.state}")
        else:
            raise NotImplementedError("Only console rendering is supported.")
