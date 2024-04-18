from collections import defaultdict

# 这里都是在某一时间戳下进行的计算
class LFUCache: # 单个个体的缓存更新方法，送入的就是某个时刻看见的某用户与某卫星，从with_GS里面送进来
    def __init__(self, satellite, selected_users, video_set, timeslot): # 不一定是所有用户都往这里打，在外面区分一下，
        self.cache_capacity = satellite.total_cachespace
        self.energy_capacity = satellite.energy_capacity
        self.users = selected_users
        self.timeslot = timeslot
        self.cache_content = satellite.cache_content # 初始缓存内容为0
        self.now_cache = satellite.now_cachedsapce
        self.video_set = video_set    
        self.frequency = defaultdict(int)
        self.min_frequency = 0

    def update(self):
        for user in self.users:
            now_video_request = user.request[self.timeslot]
            if now_video_request not in self.cache: # (video_id, chunk_id)，应该一起判断，先判断是否足够加入，够的话直接存，不够的话就得判断优先级，或者再更新？反正都得写入的，我按一次写入的能耗来写就行
                if self.now_cache + self.video_set[now_video_request[0]].chunk_size <= self.cache_capacity:
                    # 加入逻辑
                    self.cache_content.append(now_video_request)
                    self.now_cache = self.now_cache + self.video_set[now_video_request[0]].chunk_size
                    self.
                else:
                    
                # 1、如果缓冲区没满，直接加入
                # 2、消耗能量

    def get(self, key):
        if key in self.cache:
            self.frequency[key] += 1
            return self.cache[key]
        else:
            return None

    def put(self, key, value):
        if self.capacity == 0:
            return

        if key in self.cache:
            self.cache[key] = value
            self.frequency[key] += 1
        else:
            if len(self.cache) >= self.capacity:
                self.evict()
            self.cache[key] = value
            self.frequency[key] = 1
            self.min_frequency = 1

    def evict(self):
        min_freq_keys = [key for key in self.frequency if self.frequency[key] == self.min_frequency]
        evict_key = min_freq_keys[0]
        for key in min_freq_keys:
            if self.frequency[key] < self.frequency[evict_key]:
                evict_key = key
        del self.cache[evict_key]
        del self.frequency[evict_key]

        self.update_min_frequency()

    def update_min_frequency(self):
        if not self.frequency:
            self.min_frequency = 0
        else:
            self.min_frequency = min(self.frequency.values())

    def process_request(self, key):
        if key in self.cache:
            self.frequency[key] += 1
        else:
            if len(self.cache) >= self.capacity:
                self.evict()
            self.cache[key] = None
            self.frequency[key] = 1
            self.min_frequency = 1