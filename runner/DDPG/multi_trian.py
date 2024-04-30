import sys,os
module_path = 'D:\\Pyproject\\StarPerf_Simulator\\runner'
if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)


import gym
import torch
import numpy as np
import random
import threading
import copy
from torch.nn import functional as F
from torch.optim import Adam
from DDPG.agent import DDPG  # 假设已有DDPG智能体实现
from env.SingleUserVideoStreamingEnv import SingleUserVideoStreamingEnv, OUNoise  # 环境正规化
from collections import deque
from torch import nn
import statistics
import datetime

import csv

curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加父路径到系统路径sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间



multi_record = {}
record_lock = threading.Lock()
change_agent_lock = threading.Lock()

bitrate_list = [640,1600,2400,4000,6000,8400]
buffer_capacity = 20

class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG' # 算法名称
        self.env = 'satellite_NOMA' # 环境名称
        self.result_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/models/'  # 保存模型的路径 我只保存 global 的 模型
        self.train_eps = 300 # 训练的回合数
        self.eval_eps = 50 # 测试的回合数
        self.gamma = 0.99 # 折扣因子
        self.critic_lr = 1e-3 # 评论家网络的学习率
        self.actor_lr = 1e-4 # 演员网络的学习率
        self.memory_capacity = 8000 
        self.batch_size = 128
        self.target_update = 2
        self.hidden_dim = 256
        self.soft_tau = 1e-2 # 软更新参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def env_agent_config(cfg, seed=1): 
    env = SingleUserVideoStreamingEnv(bitrate_list=bitrate_list,buffer_capacity=buffer_capacity) 
    # env.seed(seed) # 随机种子
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = GlobalNetwork(state_dim,action_dim, cfg)
    return env,agent

class GlobalNetwork(DDPG):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__(state_dim=state_dim, action_dim=action_dim, cfg=cfg) # 网络结构直接完全继承DDPG的结构
        # 定义全局网络结构，此处简化实现,直接继承DDPG
        self.lock = threading.Lock()

    def sync_with_global(self, local_networks):
        """将全局网络参数同步到本地网络"""
        with self.lock:
            for local_network in local_networks:
                with local_network.lock:
                    local_network.agent.actor.load_state_dict(self.actor.state_dict())
                    local_network.agent.critic.load_state_dict(self.critic.state_dict())
                    local_network.agent.target_actor.load_state_dict(self.target_actor.state_dict())
                    local_network.agent.target_critic.load_state_dict(self.target_critic.state_dict())

    def update_from_locals(self, local_networks):
        """从所有本地网络收集参数更新全局网络"""
        with self.lock:
            # 取平均更新
            global_actor_dict = self.actor.state_dict()
            global_critic_dict = self.critic.state_dict()
            global_target_actor_dict = self.target_actor.state_dict()
            global_target_critic_dict = self.target_critic.state_dict()

            for key in global_actor_dict.keys():
                global_actor_dict[key] = torch.mean(torch.stack([local.agent.actor.state_dict()[key] for local in local_networks]), 0)
            for key in global_critic_dict.keys():
                global_critic_dict[key] = torch.mean(torch.stack([local.agent.critic.state_dict()[key] for local in local_networks]), 0)
            for key in global_target_actor_dict.keys():
                global_target_actor_dict[key] = torch.mean(torch.stack([local.agent.target_actor.state_dict()[key] for local in local_networks]), 0)
            for key in global_target_critic_dict.keys():
                global_target_critic_dict[key] = torch.mean(torch.stack([local.agent.target_critic.state_dict()[key] for local in local_networks]), 0)

            self.actor.load_state_dict(global_actor_dict)
            self.critic.load_state_dict(global_critic_dict)
            self.target_actor.load_state_dict(global_target_actor_dict)
            self.target_critic.load_state_dict(global_target_critic_dict)

class LocalAgent:  
    def __init__(self, global_network, env_id, cfg):
        self.cfg = cfg
        self.env = SingleUserVideoStreamingEnv(bitrate_list=bitrate_list,buffer_capacity=buffer_capacity) # 这个name要是不一样,还可以self.env本身还可以不一样吗
        self.agent = copy.deepcopy(global_network) # 本地副本
        self.lock = threading.Lock()
        self.env_id = env_id
        print(self.env_id) # 看看是不是不一样

        with record_lock:
            multi_record[self.env_id] = {
                "QoE": deque([0]*5, maxlen=5),
                "power": deque([0]*5, maxlen=5),
                "buffer": deque([0]*5, maxlen=5)
            }

    def add_record_to_csv(self, filepath, data_dict):
        with open(filepath, 'a', newline='') as file:  # 'a' 模式用于追加数据
            writer = csv.DictWriter(file, fieldnames=data_dict.keys())
            if file.tell() == 0:  # 检查文件是否为空，如果为空则写入表头
                writer.writeheader()
            writer.writerow(data_dict)

    def get_whole_state(self, state, QoE_all_user, power_all_user, buffer_change):
        QoE_state = [statistics.mean(QoE_all_user), statistics.variance(QoE_all_user)] # QoE特征
        power_state = [statistics.mean(power_all_user), statistics.variance(power_all_user)]
        state.insert(0, QoE_state)
        state.insert(1, power_state)
        state.insert(2, buffer_change)

        return np.array(state)
    
    def find_max_difference(self, lst):
        if len(lst) < 2:  # 如果列表中的元素少于两个，则无法计算差值
            return 0
        max_value = max(lst)
        min_value = min(lst)
        return max_value - min_value
    
    def get_new_reward(self, QoE, power, QoE_all_user, buffer_change, power_all_user):
        reward = QoE \
                - self.find_max_difference(QoE_all_user) \
                - power \
                - self.find_max_difference(power_all_user) \
                - buffer_change
        return reward
    
    def get_throughput(self, action):
        h = 1 # 模拟每个人都是1
        bandwidth = 100 # 随便模拟一下，到时候再改
        noise = 10 # 不知道靠不靠谱啊

        other_power = []
        with record_lock:
            for env, info in multi_record.items(): # 开始迭代
                if self.env_id != env: # 说明不是自己的内容
                    other_power.append(info["power"][-1])
        
        snr = h**2 * action / (sum(other_power) + noise) 

        throughput = bandwidth * np.log2(1 + snr)
                                
        return throughput
    
    def get_global_reward(self, QoE_all_user, power_all_user):
        average_QoE = np.mean(QoE_all_user)
        max_difference = max(QoE_all_user) - min(QoE_all_user)
        average_power = np.mean(power_all_user)
        return average_QoE - max_difference - average_power

# TODO: 
# 0、将multi_record的锁机制考虑好 OK
# 1、再考虑state，state除了env提供还需要外面的其他线程中智能体的信息 OK
# 2、reward需要考虑 OK
# 3、详细修改训练过程 OK
# 4  每时刻计算全局reward, 来验证效果，然后记录每一个单步reward，用csv的格式 OK
    def run(self):
        """运行一个完整的episode，同步参数，并更新全局网络""" 

        print('开始训练！')
        print(f'环境：{self.cfg.env}，算法：{self.cfg.algo}，设备：{self.cfg.device}')
        
        data_to_record = {
            "global_reward": 0,
            "step_reward": 0,
            "QoE": 0,
            "buffer_size": 0,
            "rebuffer_event": 0,
            "rate_switch_event": 0,
            "bitrate": 0
            }

        ou_noise = OUNoise(env.action_space)  # 动作噪声

        state = self.env.reset()
        state = self.get_whole_state(state)
        ou_noise.reset()
        total_reward = 0
        done = False
        i_step = 0
        while not done:
            i_step += 1
            action = self.agent.choose_action(state)  # action是分配功率，但环境中的是吞吐量,其中差了一NOMA的香农公式
            action = ou_noise.get_action(action, i_step)  

            throughput = self.get_throughput(action) # 转化为吞吐量
            next_state, QoE, done, _ = self.env.step(throughput) 
            # 在get_whole_state之前修改好其中内容

            QoE_all_user = []
            power_all_user = []
            with record_lock:
                multi_record[self.env_id]["QoE"].append(QoE)
                multi_record[self.env_id]["power"].append(action)
                multi_record[self.env_id]["buffer"].append(next_state[1])  # 假设next_state[1]是buffer长度
                
                for env, info in multi_record.items(): # 开始迭代
                    QoE_all_user.append(info["QoE"][-1])  # 拿到所有人最新的QoE信息
                    power_all_user.append(info["power"][-1]) # 拿到所有人最新的power，决策

                if self.env_id == env: # 说明是自己的内容
                    buffer_change = info["buffer"][-1] - info["buffer"][-2]
            ## next state 与 reward 都需要修改

            next_state = self.get_whole_state(next_state, QoE_all_user, power_all_user, buffer_change)
            reward = self.get_new_reward(QoE, action, QoE_all_user, buffer_change, power_all_user)

            global_reward = self.get_global_reward(QoE_all_user, power_all_user)

            self.agent.memory.push(state, action, reward, next_state, done)

            with self.lock:
                self.agent.update()
            state = next_state
            total_reward += reward 
            
            # 记录数据
            data_to_record["global_reward"] = global_reward
            data_to_record["step_reward"] = reward
            data_to_record["QoE"] = QoE
            data_to_record["buffer_size"] = next_state[1]
            data_to_record["rebuffer_event"] = next_state[2]
            data_to_record["rate_switch_event"] = next_state[3]
            data_to_record["bitrate"] = next_state[0]
            self.add_record_to_csv(self.cfg.result_path+self.env_id+".csv", data_to_record)
        
        # 执行到这里就是一个视频已经看完
        # 可以把这个全局记录池中的信息删掉了
        with record_lock:
            if self.env_id in multi_record:
                del multi_record[self.env_id]

        # # 同步全局网络
        # self.global_network.sync_with_global(self.local_network)
        # # 同步本地网络
        # self.local_network.sync_with_global(self.global_network)
        return total_reward

# def main():



if __name__ == "__main__":
    episode_now = 0
    num_agents = 1

    cfg = DDPGConfig()  # 假设已定义配置
    global_network, _ = env_agent_config(cfg)
    # global_network = GlobalNetwork(cfg.state_dim, cfg.action_dim, cfg)
    agents = [LocalAgent(global_network, i, cfg) for i in range(num_agents)]
    # agents = []

    episode_lock = threading.Lock()
    agents_lock = threading.Lock()

    def manage_agents():
        global episode_now

        if episode_now == 0: # 第一次进入循环
            for current_agent in agents:
                t = threading.Thread(target=train_agent, args=(current_agent,))
                t.start()

        while episode_now < 300:
            with agents_lock:
                current_agents_count = len(agents)
                if current_agents_count < 20:
                    new_agents_needed = max(5 - current_agents_count, 0)
                    new_agent = random.randint(new_agents_needed, 5)  # 一直在这死循环啊，没道理
                    for _ in range(new_agents_needed):
                        new_agent = LocalAgent(global_network, episode_now, cfg)
                        agents.append(new_agent)
                        t = threading.Thread(target=train_agent, args=(new_agent,))
                        t.start()

    def train_agent(agent):
        global episode_now
        agent.run()   
        
        # 执行训练完了
        with agents_lock:
            agents.remove(agent)

        with episode_lock:
            episode_now += 1
            if episode_now >= 300:
                return  # 停止添加更多agents

        with agents_lock:
            global_network.update_from_locals(agents)     # 更新全局网络
            global_network.sync_with_global(agents)       # 更新每个local至新的全局网络

            # 记录 global_network 的网络参数
            global_network.save(cfg.model_path)


    # 启动管理器线程
    manager_thread = threading.Thread(target=manage_agents)
    manager_thread.start()

    manager_thread.join()  # 等待管理器线程完成
