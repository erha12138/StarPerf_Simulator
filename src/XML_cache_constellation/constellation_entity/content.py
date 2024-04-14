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
        self.chunk_size = chunk_size  # 每个块的大小，要是考虑复杂的话，点播每个视频chunk应该是不一样大的，这对缓存决策其实会造成一些困难，我们想通过某种方式去解决它
        self.chunks = [VideoChunk(id, i, chunk_size) for i in range(num_chunks)]