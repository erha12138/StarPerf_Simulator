import datetime
import gym
import torch
import numpy as np
from runner.env.SingleUserVideoStreamingEnv import SingleUserVideoStreamingEnv, OUNoise
from DDPG.agent import DDPG
from common.utils import save_results, make_dir
from common.plot import plot_rewards, plot_rewards_cn

class DDPGConfig:
    def __init__(self):
        self.initial_agents = 5  # 初始智能体数量
        self.envs = ['satellite_NOMA_{}'.format(i) for i in range(self.initial_agents)]
        self.algo = 'DDPG'
        self.train_eps = 300 # 训练的回合数
        self.eval_eps = 50 # 测试的回合数
        self.gamma = 0.99 # 折扣因子
        self.critic_lr = 1e-3 # 评论家网络的学习率
        self.batch_size = 128
        self.target_update = 2
        self.hidden_dim = 256
        self.soft_tau = 1e-2 # 软更新参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def env_agents_config(cfg):
    envs = [SingleUserVideoStreamingEnv(gym.make(env_name)) for env_name in cfg.envs]
    agents = [DDPG(env.observation_space.shape[0], env.action_space.shape[0], cfg) for env in envs]
    return envs, agents

def manage_agents(cfg, envs, agents, add_agents=0, remove_indices=[]):
    # 添加新环境
    # 添加智能体
    for _ in range(add_agents):
        new_env_name = 'satellite_NOMA_{}'.format(len(envs))
        new_env = SingleUserVideoStreamingEnv(gym.make(new_env_name)) # 添加智能体时再加入新环境
        new_agent = DDPG(new_env.observation_space.shape[0], new_env.action_space.shape[0], cfg)
        envs.append(new_env)
        agents.append(new_agent)
        cfg.envs.append(new_env_name)
    
    # 移除指定的智能体和环境
    for index in sorted(remove_indices, reverse=True):
        del envs[index]
        del agents[index]
        del cfg.envs[index]

def train(cfg, envs, agents):
    print('开始训练！')

    for i_ep in range(cfg.train_eps):  # 一个episode 应该是观看一个视频看完，所以应该是一个用户一个episode，一个step是一个chunk
        # if i_ep % 50 == 0 and i_ep > 0:  # 每10个回合动态调整智能体数量 
            # manage_agents(cfg, envs, agents, add_agents=1)  # 示例：每10个回合添加一个智能体
            # if len(agents) > 10:
            #     manage_agents(cfg, envs, agents, remove_indices=[0])  # 当智能体数量超过10时，移除一个

            # 应该是一个视频看完了再变用户数量
            for i in range(len(agents)): # 每个用户单独训练，练自己的模型，迁移模型给下一个用户，同一套模型参数，给不同的用户去练




        states = [env.reset() for env in envs]
        ou_noises = [OUNoise(env.action_space) for env in envs]
        [ou_noise.reset() for ou_noise in ou_noises]
        done = False
        ep_rewards = [0] * len(agents)
        while not done:
            actions = [agent.choose_action(state) for agent, state in zip(agents, states)]
            actions = [ou_noises[i].get_action(actions[i], i_ep) for i in range(len(actions))]
            
            for i, env in enumerate(envs):
                next_state, reward, done, _ = env.step(actions[i])
                agents[i].memory.push(states[i], actions[i], reward, next_state, done)
                agents[i].update()
                states[i] = next_state
                ep_rewards[i] += reward
        print(f'回合：{i_ep+1}/{cfg.train_eps}, 奖励：{ep_rewards}')

    print('完成训练！')

if __name__ == "__main__":
    cfg = DDPGConfig()
    envs, agents = env_agents_config(cfg)
    train(cfg, envs, agents)
    # 保存和绘图逻辑可以根据需要添加
