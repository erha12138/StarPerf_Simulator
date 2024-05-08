import sys,os
# module_path = 'D:\\Pyproject\\StarPerf_Simulator\\runner'
module_path = '/home/debian/StarPerf_Simulator/runner' # linux

if module_path not in sys.path:
    sys.path.append(module_path)
print(sys.path)

import numpy as np
import random
import threading
from DDPG.multi_trian import DDPGConfig, LocalAgent, env_agent_config  # 假设已有DDPG智能体实现
import datetime

import csv

# curr_path = os.path.dirname(os.path.abspath(__file__)) # 当前文件所在绝对路径
# parent_path = os.path.dirname(curr_path) # 父路径
# sys.path.append(parent_path) # 添加父路径到系统路径sys.path
# curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间

multi_record = {}
record_lock = threading.Lock()
change_agent_lock = threading.Lock()
global_lock = threading.Lock()

bitrate_list = [640,1600,2400,4000,6000,8400]
buffer_capacity = 20

# TODO:

if __name__ == "__main__":  # 暂时只能写在主线程里，全局变量只能在主线程中使用

    num_agents = 21 # 先试试固定数量能不能练出来，把正常逻辑给写对先
    h_every_user = [random.uniform(0.99,1.01) for i in range(num_agents)] # 暂时暂时先不考虑用户数量的增加

    episode_now = 0


################## DC-DDPG
    cfg = DDPGConfig()  # 假设已定义配置
    _, global_network = env_agent_config(cfg)
    # global_network = GlobalNetwork(cfg.state_dim, cfg.action_dim, cfg)
    agents = [LocalAgent(global_network, i, cfg, h_every_user[i]) for i in range(num_agents)]

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
                if current_agents_count > 20:
                    continue
                if current_agents_count < 5:
                    new_agents_needed = max(5 - current_agents_count, 0)
                    new_agent = random.randint(new_agents_needed, 5)  
                    print("\n num of new_agent:", new_agent)
                    print("\n num of now_agent:", agents)
                    for _ in range(new_agent):
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