import gym
import torch
import numpy as np
import threading
import copy
from torch.nn import functional as F
from torch.optim import Adam
from DDPG.agent import DDPG  # 假设已有DDPG智能体实现
from DDPG.env import NormalizedActions  # 环境正规化
from collections import deque
from torch import nn
import statistics

multi_record = {}
record_lock = threading.Lock()

class GlobalNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GlobalNetwork, self).__init__()
        # 定义全局网络结构，此处简化实现
        self.actor = nn.Linear(state_dim, action_dim)
        self.critic = nn.Linear(state_dim + action_dim, 1)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.lock = threading.Lock()

    def sync_with_global(self, local_network):
        """将全局网络参数同步到本地网络"""
        local_network.load_state_dict(self.state_dict())

    def update_from_locals(self, local_networks):
        """从所有本地网络收集参数更新全局网络"""
        with self.lock:
            # 取平均更新
            global_actor_dict = self.actor.state_dict()
            global_critic_dict = self.critic.state_dict()
            for key in global_actor_dict.keys():
                global_actor_dict[key] = torch.mean(torch.stack([local.actor.state_dict()[key] for local in local_networks]), 0)
            for key in global_critic_dict.keys():
                global_critic_dict[key] = torch.mean(torch.stack([local.critic.state_dict()[key] for local in local_networks]), 0)

            self.actor.load_state_dict(global_actor_dict)
            self.critic.load_state_dict(global_critic_dict)

class LocalAgent:
    def __init__(self, global_network, env_name, cfg):
        self.env = NormalizedActions(gym.make(env_name))
        self.agent = DDPG(self.env.observation_space.shape[0], self.env.action_space.shape[0], cfg)
        self.global_network = global_network
        self.local_network = copy.deepcopy(global_network)  # 本地副本
        with record_lock:
            multi_record[str(self.env)] = {
                "QoE": deque([0]*5, maxlen=5),
                "power": deque([0]*5, maxlen=5),
                "buffer": deque([0]*5, maxlen=5)
            }

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
## 应该还要加参数的


# TODO: ，将multi_record的锁机制考虑好
# 1、再考虑state，state除了env提供还需要外面的其他线程中智能体的信息
# 2、reward需要考虑
# 3、详细修改训练过程

    def run(self):
        """运行一个完整的episode，同步参数，并更新全局网络""" # 训练过程再详细改改
        state = self.env.reset()
        state = self.get_whole_state(state)
        total_reward = 0
        done = False
        while not done:
            action = self.agent.choose_action(state)
            next_state, QoE, done, _ = self.env.step(action) 
            # 在get_whole_state之前修改好其中内容

            QoE_all_user = []
            power_all_user = []
            with record_lock:
                multi_record[str(self.env)]["QoE"].append(QoE)
                multi_record[str(self.env)]["power"].append(action)
                multi_record[str(self.env)]["buffer"].append(next_state[1])  # 假设next_state[1]是buffer长度
                
                for env, info in multi_record.items(): # 开始迭代
                    QoE_all_user.append(info["QoE"][-1])  # 拿到所有人最新的QoE信息
                    power_all_user.append(info["power"][-1]) # 拿到所有人最新的power，决策

                if str(self.env) == env: # 说明是自己的内容
                    buffer_change = info["buffer"][-1] - info["buffer"][-2]
            ## next state 与 reward 都需要修改

            next_state = self.get_whole_state(next_state, QoE_all_user, power_all_user, buffer_change)
            reward = self.get_new_reward(QoE, action, QoE_all_user, buffer_change, power_all_user)

            self.agent.memory.push(state, action, reward, next_state, done)
            self.agent.update()
            state = next_state
            total_reward += reward 


        # 执行到这里就是一个视频已经看完
        # 可以把这个全局记录池中的信息删掉了
        with record_lock:
            if str(self.env) in multi_record:
                del multi_record[str(self.env)]

        # 同步全局网络
        self.global_network.sync_with_global(self.local_network)
        # 同步本地网络
        self.local_network.sync_with_global(self.global_network)

        return total_reward

def main():
    num_agents = 5 # 同时开多少个来练，少了再加，写个简单的逻辑来加
    env_name = 'Pendulum-v0'
    cfg = DDPGConfig()  # 假设已定义配置

    global_network = GlobalNetwork(cfg.state_dim, cfg.action_dim)
    agents = [LocalAgent(global_network, env_name, cfg) for _ in range(num_agents)]

    def train_agent(agent):
        while True:
            agent.run()

    threads = []
    for agent in agents:
        t = threading.Thread(target=train_agent, args=(agent,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()  # 等待所有线程完成

if __name__ == "__main__":
    main()
