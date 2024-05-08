import numpy as np
import gym
from gym import spaces
import random

# bitrate_list = [640,1600,2400,4000,6000,8400] # kbps

class SingleUserVideoStreamingEnv(gym.Env):
    """user
    A simple environment for a single  video streaming scenario with ABR logic.
    """
    
    def __init__(self, bitrate_list, buffer_capacity):
        super(SingleUserVideoStreamingEnv, self).__init__()
        
        self.bitrate_list = bitrate_list  # Possible bitrates that can be requested
        self.buffer_capacity = buffer_capacity  # The buffer capacity in seconds
        self.chunk_duration = 4 # 4s
        self.max_duration = 4 * random.randint(1000, 2500) # 400s - 1000s
        
        self.reset()
        
        # Define the action space to be the throughput and the observation space
        self.action_space = spaces.Box(low=0.40, high=2, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, -1, -1]),  # # qoe 的均值与方差，功率的均值与方差，buffer的差值；这个与环境本身关系不大，要在训练时重新计算，考虑是不是给个归一化？
            high=np.array([1000, 1000, 1000, 1000, buffer_capacity, len(bitrate_list)-1, buffer_capacity, 1, 1]), 
            dtype=np.float32
        )

    def reset(self):
        # Reset the state to a initial state
        # self.buffer_size = np.random.rand() * self.buffer_capacity
        self.buffer_size = 0.8 * self.buffer_capacity        
        self.current_bitrate = np.random.choice(self.bitrate_list)
        self.rebuffer_event = False
        self.rate_switch_event = False
        self.QoE = 0
        self.next_bitrate = self.current_bitrate
        self.video_timeslot = 0 # 从第0s开始看
        
        return np.array([self.bitrate_list.index(self.current_bitrate), self.buffer_size, self.rebuffer_event, self.rate_switch_event])

    def step(self, action):
        self.rebuffer_event = False
        self.rate_switch_event = False

        # Simulate one time step within the environment
        throughput = action # action 应该是吞吐量
        
        # ABR logic to decide the next bitrate based on throughput and buffer size
        self.next_bitrate = self.abr_decision(throughput)
        # 更新buffer
        self.update_buffer(throughput)

        # Calculate bitrate switch and update QoE
        bitrate_switch_penalty = self.calculate_bitrate_switch_penalty(self.current_bitrate, self.next_bitrate)
        
        self.QoE = self.calculate_QoE(bitrate_switch_penalty)
        
        # Define the reward as the QoE
        reward = self.QoE
        
        # Update current bitrate
        self.current_bitrate = self.next_bitrate
        
        self.video_timeslot += self.chunk_duration


        if self.video_timeslot > self.max_duration:
            done = True # 视频播放完成
        else:
            done = False
        
        # Return the next state, reward, done and extra info
        next_state = np.array([self.bitrate_list.index(self.next_bitrate), self.buffer_size, self.rebuffer_event, self.rate_switch_event])
        # print("now time slot:", self.video_timeslot)
        return next_state, reward, done, {}

    # def abr_decision(self, throughput):
    #     # Implement your ABR algorithm logic here to decide the next bitrate
    #     # For now, we simply choose the highest possible bitrate that can be supported by the current throughput
    #     affordable_bitrates = [b for b in self.bitrate_list if b <= throughput]
    #     next_bitrate = max(affordable_bitrates) if affordable_bitrates else min(self.bitrate_list) # 我觉得可以先这样
    #     if self.buffer_size <= 2: # 考虑一部分buffer，在我的决策中，buffer_size可以控制在大于2，小于5的状态 
    #         next_bitrate = self.bitrate_list[self.bitrate_list.index(self.next_bitrate) - 1]
    #     return next_bitrate
    def abr_decision(self, throughput):
        # BOLA的效用函数通常取决于视频的码率和V参数（这是一个调整参数，影响缓冲区大小和码率选择）
        V = 0.54  # V参数，根据具体场景进行调整
        # gp = 4  # gp参数，这个参数通常取决于视频播放的特性，可以调整
        # chunk_size_megabits = self.current_bitrate * self.chunk_duration

        # download_time = chunk_size_megabits / throughput if throughput > 0 else float('inf')

        # 计算每个码率的效用值
        utility_list = [np.log(br) + V * (self.buffer_size - 2 + self.chunk_duration - (br*self.chunk_duration)/throughput if throughput > 0 else float('inf')) for br in self.bitrate_list]
        
        # 选择最高效用值的码率，但吞吐量必须支持该码率
        next_bitrate = max((br for br, u in zip(self.bitrate_list, utility_list) if br <= throughput), 
                            key=lambda br: utility_list[self.bitrate_list.index(br)],
                            default=min(self.bitrate_list))
        
        return next_bitrate
            
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
    
    def calculate_bitrate_switch_penalty(self, current_bitrate, next_bitrate):
        # Define the penalty for bitrate switching.
        # This can be a function of how much the bitrate has changed.
        # For example, we can use a quadratic difference.
        if next_bitrate != current_bitrate:
            self.rate_switch_event = True
        bitrate_change = abs(self.bitrate_list.index(next_bitrate) - self.bitrate_list.index(current_bitrate))
        penalty = bitrate_change
        return penalty


    def calculate_QoE(self, bitrate_switch_penalty):
        # Define and calculate the QoE based on bitrate, rebuffering, etc.
        # For simplicity, we can start with bitrate minus rebuffering penalties
        qoe = self.bitrate_list.index(self.current_bitrate) * 4 # 这个也是要改的
        if self.rebuffer_event:
            qoe -= 5  # Example penalty for rebuffering
        qoe -= bitrate_switch_penalty  # Subtract the bitrate switch penalty
        return qoe

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Buffer size: {self.buffer_size}, Current bitrate: {self.current_bitrate}, Rebuffer event: {self.rebuffer_event}, Rate switch event: {self.rate_switch_event}, QoE: {self.QoE}")
        else:
            raise NotImplementedError("Only console rendering is supported.")



class OUNoise(object):
    '''Ornstein–Uhlenbeck噪声
    '''
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu # OU噪声的参数
        self.theta        = theta # OU噪声的参数
        self.sigma        = max_sigma # OU噪声的参数
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
    def reset(self):
        self.obs = np.ones(self.action_dim) * self.mu
    def evolve_obs(self):
        x  = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.obs = x + dx
        return self.obs
    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period) # sigma会逐渐衰减
        return np.clip(action + ou_obs, self.low, self.high) # 动作加上噪声后进行剪切



if __name__ == "__main__":
    bitrate_list = [640,1600,2400,4000,6000,8400]
    env = SingleUserVideoStreamingEnv(bitrate_list=bitrate_list,buffer_capacity=20)
    for i in range(300):
        state = env.step([random.uniform(4500, 15000)])
        print(state)