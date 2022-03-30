import numpy as np
import math

"""原始数据"""
data = [936.5,938.8,1033.5,1239.9,1327.1,1425.5,1593.5,1861.5]

"""计算级比"""
def calculus_ratios(data):
    ratios=[]
    for i in range(1,len(data)):
        ratio = data[i-1]/data[i]
        ratios.append(ratio)
    #print(ratios)
    return ratios
print(calculus_ratios(data))

"""判断级别是否在界区之内"""
def deal_data(data):
    ratios = calculus_ratios(data)
    for i in range(len(ratios)):
        if math.exp(2/(len(data)+1)) > ratios[i] > math.exp(-2/(len(data)+1)):
            print("the ratio is available")
        else:
            print("the ratio is not available")

deal_data(data)

"""gm(1,1)的建模"""
def gm_11(data):
    x0 = np.array(data)
    x1 = np.cumsum(x0)
    B1 = -(x1[:len(data)-1]+x1[1:len(data)])/2
    B = np.array([B1,np.ones_like(B1)]).T
    Y = data[1:]
    P = np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),Y)
    a,b = P[0],P[1]
    return np.array([round(((x0[0]-b/a) * math.exp(-a*i)+b/a),4) for i in range(len(data))])
print(gm_11(data))

result1 = np.ediff1d(gm_11(data))
result = np.insert(result1,0,936.5)
print(result)