import random
from collections import defaultdict

class VideoChunk:
    """
    视频块类，代表视频中的一个数据块。
    """
    def __init__(self, video_id, chunk_id, size):
        self.video_id = video_id  # 视频的唯一标识
        self.chunk_id = chunk_id  # 块的序号
        self.size = size  # 块的大小，单位为MB

class Video:
    """
    视频类，包含视频的基本信息和它的块。
    """
    def __init__(self, id, num_chunks, chunk_size):
        self.id = id  # 视频的唯一标识
        self.num_chunks = num_chunks  # 视频包含的块数量
        self.chunk_size = chunk_size  # 每个块的大小，要是考虑复杂的话，点播每个视频chunk应该是不一样大的，
        # 这对缓存决策其实会造成一些困难，我们想通过某种方式去解决它

class Request:
    """
    用户请求类，描述每个用户在特定时刻对特定视频块的请求。
    """
    def __init__(self, user_id, time_len, video_list):
        self.user_id = user_id  # 用户的唯一标识
        self.request_list = self.generate_request(time_len, video_list)  # 把所有时间戳的请求的内容都生成出来

    def generate_request(self, time_len, video_list):
        max_video_id = len(video_list)
        request_list = []
        timeslot = 0
        while(1):
            video_id = random.randint(0, max_video_id-1)   # 固定就有这么多种类型的视频，int是可以到后面那个值的
            max_video_chunk_id = video_list[video_id].num_chunks
            for j in range(max_video_chunk_id):
                if timeslot < time_len:
                    request_list.append((video_id,j))
                else:
                    return request_list
                timeslot += 1
            # 如果循环正常结束，再去请求别的视频的chunk


def generate_requests_set_for_per_user(user_num, time_len, video_list):
    request_list = []
    for user_id in range(user_num):
        request_list.append(Request(user_id, time_len, video_list))
    return request_list



if __name__ == "__main__":
    # 示例：创建一个video_list
    video_list = [Video(id=0,num_chunks=10,chunk_size=50),Video(id=1,num_chunks=20,chunk_size=30)]  # size单位为50mbps
    request = Request(user_id=1, timeslot=55, video_list=video_list)
    print(request.request_list)  # 输出请求的详细信息