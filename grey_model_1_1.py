import numpy as np
import math

"""原始数据"""
data = [936.5,938.8,1033.5,1239.9,1327.1,1425.5,1593.5,1861.5]

#第一步首先进行级比检验
"""预设c = 0 平移系数 自定"""

def deal_data(data,c=0):
    calculate_ratios(data)
    for i in range(len(ratios)):
        if math.exp(-2/len(data)+1) < ratios[i] <math.exp(2/len(data)+1):
            print("ratio"+str(i)+"passes the test")
        else:
            find_c(data,c=0)

    return [ratios,data,c]

#print(deal_data(data,0)[0])
#print(deal_data(data,0)[1])

def calculate_ratios(data): #计算级比
    ratios = []
    for i in range(1, len(data)):
        ratio = data[i - 1] / data[i]
        ratios.append(ratio)
    return ratios

def find_c(data,c=0): #寻找合适的平移系数c
    calculate_ratios(data)
    for ratio in ratios:
        while ratio < math.exp(-2 / len(data) + 1) or ratios > math.exp(2 / len(data) + 1):
            c += 0.1
            for i in range(len(data)):
                data[i] += c
                calculate_ratios(data)
#平移变换求解之后还需要减去c值

#第二步进行GM(1,1)建模
def gm1_1(data):
    x0 = np.array(data)
    x1 = np.cumsum(x0)
    B1 = -(x1[:len(x1)-1]+x1[1:])/2.0
    B = np.array([B1,np.ones_like(B1)]).T
    Y = data[1:]
    P = np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),Y)
    a,b = P[0],P[1]
    #返回的时间影响序列应为：
    return np.array([round(((x0[0]-b/a) * math.exp(-a*i)+b/a),4) for i in range(len(data))])
#此函数的返回值即为gm(1,1)模型的预测的生成数列

#下一步计算模型的还原值
result1 = np.ediff1d(gm1_1(data)) #递减，即求模型的还原值

"""在此还需要减去平移系数，还原模拟值"""

result = np.insert(result1,0,[936.5]) #此即为模拟的结果加上第一项 insert(x,pos,value)

#print(result)
print("原始数据为：")
print(data)
print("模型生成的序列为：")
print(result)

#检验GM(1,1)模型的精度
#残差
def residual_error(data,result):
    errors=[]
    for i in range(len(data)):
        error = round((data[i] - result[i]),4)
        errors.append(error)
    return errors
errors = residual_error(data, result)
print("残差为：" )
print(errors)

#相对误差epsilon
def epsilons(data,result):
    epsilons = []
    for i in range(len(data)):
        epsilon = round((data[i]-result[i])/data[i],4)
        #epsilon = "%.2f%%"%(epsilon1 * 100)
        epsilons.append(epsilon)
    return epsilons
epsilons_k = epsilons(data,result)
print("相对误差为：")
print(epsilons_k)

"""平均相对误差"""
def epsilons_accuracy(epsilons_k,data):
    epsilon_accus = []
    for i in range(len(data)):
        epsilon_accu = round((1 - abs((data[i]-result[i])/data[i])),4)
        epsilon_accus.append(epsilon_accu)
    return epsilon_accus
epsilon_Accuracy = epsilons_accuracy(epsilons_k,data)
print("精度为：")
print(epsilon_Accuracy)

"""平均精度"""
def accuracy_avg(epsilons_Accuracy):
    Sum = 0
    for i in range(1,len(epsilons_Accuracy)):
        Sum = Sum + epsilons_Accuracy[i]
    p_0 = Sum/(len(epsilons_Accuracy)-1)
    return round(p_0,4)
p_0 = accuracy_avg(epsilon_Accuracy)
print("平均精度为：")
print(p_0)

"""后验差检验、小误差频率"""
"""关联度检验"""
"""级比偏差（指数律差异值）"""

#第五步 通过检测的GM(1,1)模型进行预测与预报
def predicts(data,k):
    predicts = []
    x0 = np.array(data)
    x1 = np.cumsum(x0)
    B1 = -(x1[:len(x1) - 1] + x1[1:]) / 2.0
    B = np.array([B1, np.ones_like(B1)]).T
    Y = data[1:]
    P = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    a, b = P[0], P[1]
    #np.array([(x0[0] - b / a) * math.exp(-a * i) + b / a for i in range(len(data))])
    for i in range(k):
        predict = round(((x0[0] - b / a) * math.exp(-a * i) + b / a),4)
        predicts.append(predict)
    predicts1 = np.ediff1d(predicts)
    predicts_origin  = np.insert(predicts1, 0, [936.5])
    return predicts_origin,k

print("10期的预测值如下：")
print(round(predicts(data,10)[0][9],4))