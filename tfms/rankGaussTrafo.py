from scipy.special import erfinv
from collections import OrderedDict
from math import sqrt
import numpy as np

def cdfinv(y):
    """简化的公式，与原NormalCDFInverse等价，且精度更高

    公式推导参见https://www.cnblogs.com/htj10/p/8621771.html
    """
    return sqrt(2) * erfinv(2 * y - 1)

def rankGaussTrafo(dataIn):
    hist = dict()       # hist统计元素的出现频率
    for i in dataIn:
        if i not in hist:
            hist[i] = 1
        else:
            hist[i] += 1

    hist = OrderedDict([t for t in sorted(hist.items(), key=lambda d:d[0])])    # 按照key排序

    trafoMap = dict()
    if len(hist) == 1:      # unary column: trafo all to 0
        trafoMap[list(hist.keys())[0]] = 0.0
    elif len(hist) == 2:    # binary column: trafo to 0 / 1
        trafoMap[list(hist.keys())[0]] = 0.0
        trafoMap[list(hist.keys())[1]] = 1.0
    else:                   # more than 2 unique values
        mean = 0.0
        cnt = 0
        N = len(dataIn)

        for key, value in hist.items():
            rankV = cnt * 1.0 / N       # 累计次数占总次数的比例，取值[0,1]，单调递增（和分布函数F的性质对应）

            rankV = rankV * 0.998 + 1e-3    # 注意到cdfinv(0) = -inf，而cdfinv(1) = inf。这个操作使得rankV限制在[0.001,0.999]，而cdfinv(rankV)限制在[-3.09,3.09]，避免了极端情况的发生

            scale_factor = 1.0      # 使用0.7可以得到原cpp中的结果，对分布有收缩作用。建议使用1.0，可保持std=1.0
            rankV = cdfinv(rankV) * scale_factor     # 将其作为分布函数F的值，逆向求N(0,1)的α分位数

            mean += value * rankV   # value是出现次数，乘以rankV。rankV可以看作是value的权重。注意到hist是按key从小到大排序的，排位越后的数对均值的贡献越大
            trafoMap[key] = rankV   # 记录为trafoMap的值
            cnt += value            # 累计次数
        
        mean /= N

        for key in trafoMap.keys():
            trafoMap[key] -= mean   # 每个rankV减去均值，得到最终trafoMap

    dataOut = dataIn.copy()
    for i in range(len(dataIn)):    # 这里简单地把trafoMap映射到输出
        dataOut[i] = trafoMap[dataIn[i]]
    return dataOut