import math

def calculate(T, P, N):
    """
       计算信息传输速率（ITR）

       参数:
       N : int : 分类类别总数
       P : float : 准确率 (0 <= P <= 1)
       T : float : 传输所需时间（秒）

       返回:
       float : 信息传输速率（ITR）
       """
    itr = (60 / T) * (math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1)))
    return itr