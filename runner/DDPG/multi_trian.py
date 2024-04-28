import gym
import torch
import numpy as np
import threading
import copy
from torch.nn import functional as F
from torch.optim import Adam
from DDPG.agent import DDPG  # 假设已有DDPG智能体实现
from env.SingleUserVideoStreamingEnv import SingleUserVideoStreamingEnv  # 环境正规化
from collections import deque
from torch import nn
import statistics
import datetime
import sys,os

curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path) # 父路径
sys.path.append(parent_path) # 添加父路径到系统路径sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

multi_record = {}
record_lock = threading.Lock()
change_agent_lock = threading.Lock()

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
    env = SingleUserVideoStreamingEnv(gym.make(cfg.env)) 
    env.seed(seed) # 随机种子
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPG(state_dim,action_dim,cfg)
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
    def __init__(self, global_network, env_name, cfg):
        self.env = SingleUserVideoStreamingEnv(gym.make(env_name)) # 这个name要是不一样,还可以self.env本身还可以不一样吗
        self.agent = copy.deepcopy(global_network) # 本地副本
        self.lock = threading.Lock()
        # self.agent = DDPG(self.env.observation_space.shape[0], self.env.action_space.shape[0], cfg) # 给一个agent
        ## 这里训练的都是self.agent,但同步的都是这个所谓的local_network,有问题!!!!!!!!!!
        # self.global_network = global_network # 内部维护一个自己的global_network
        # self.local_network = copy.deepcopy(global_network)  # 本地副本
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
# TODO: 
# 0、将multi_record的锁机制考虑好 OK
# 1、再考虑state，state除了env提供还需要外面的其他线程中智能体的信息 OK
# 2、reward需要考虑 OK
# 3、详细修改训练过程 ing
# 4  每时刻计算全局reward, 来验证效果  
    def run(self):
        """运行一个完整的episode，同步参数，并更新全局网络""" 
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

            with self.lock:
                self.agent.update()
            state = next_state
            total_reward += reward 


        # 执行到这里就是一个视频已经看完
        # 可以把这个全局记录池中的信息删掉了
        with record_lock:
            if str(self.env) in multi_record:
                del multi_record[str(self.env)]

        # # 同步全局网络
        # self.global_network.sync_with_global(self.local_network)
        # # 同步本地网络
        # self.local_network.sync_with_global(self.global_network)
        return total_reward

def main():

    episode_now = 0 # 计数

    num_agents = 5 # 同时开多少个来练，少了再加，写个简单的逻辑来加
    env_name = 'Pendulum-v0'
    cfg = DDPGConfig()  # 假设已定义配置

    global_network = GlobalNetwork(cfg.state_dim, cfg.action_dim, cfg) # global 网络初始化
    agents = [LocalAgent(global_network, env_name, cfg) for _ in range(num_agents)] # 每次name可以不一样

    def train_agent(agent):
        while True:
            agent.run()
            if True: # 这个agent执行完了
                global_network.update_from_locals(agents)     # 更新全局网络,用当前所有的网络
                global_network.sync_with_global(agents)       # 将新的全局网络更新至当前每个local
                break
        agents.remove(agent) # 从列表里删除此agent

        with change_agent_lock:
            episode_now += 1
            # 计数
            # 增加用户智能体

    
    threads = []
    for agent in agents:
        t = threading.Thread(target=train_agent, args=(agent,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()  # 等待所有线程完成

if __name__ == "__main__":
    main()
