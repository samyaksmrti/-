import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb,gamma
class FDFGRM:
    def __init__(self):
        self.n=None#数据长度
        self.ahead=4 #验证集数据个数
        self.ahead1=4#未来预测数据个数
        self.h=None #分形导数
        self.r=None #分数阶累加生成阶数
   #分数阶累加生成函数
    def fen_add(self,r,x):
        dim=len(x)
        A=np.mat(np.ones((dim,dim)))
        for i in range(len(x)):
            for j in range(len(x)):
                A[i,j]=float(gamma(i+1-(j+1)+self.r)/(gamma(self.r)*gamma(i+1-(j+1)+1)))
        return np.dot(A,x.T),A
    #最小二乘参数估计函数
    def fit(self,h,r): #此处的参数该代码未给出，利用QPSO求beta和gamma
        self.h=h
        self.r=r
        self.n=len(x_d1)
        x1_r=np.array(self.fen_add(self.r,x_d1)[0])[0]
        A0=self.fen_add(self.r,x_d1)[1]
        Y=x1_r[1:]-x1_r[:-1]
        B=np.ones((self.n-1,3))
        for i in range(1,self.n):
            z=(x1_r[i]+x1_r[i-1])/2
            B[i-1,0]=z**2*self.h*((2*(i+1)-1)/2)**(self.h-1) #B矩阵，行列
            B[i-1,1]=-z*self.h*((2*(i+1)-1)/2)**(self.h-1)
            B[i-1,2]=self.h*((2*(i+1)-1)/2)**(self.h-1)
        beta=np.dot(np.dot(np.linalg.inv(np.dot(B.T,B)),B.T),Y.T) #最小二乘估计参数
        return beta
   #拟合，测试以及未来预测结果输出
    def result(self):
        beta=self.fit(self.h,self.r)
        b=beta[0]
        a=beta[1]
        c=beta[2]
        p=a/(2*b)
        q=(a**2-4*b*c)/(4*b**2)
        x1_pred=np.array([1.0]*(self.n+self.ahead+self.ahead1))
        if q>0:
            ABS=(x_d1[0]-p-np.sqrt(q))/(x_d1[0]-p+np.sqrt(q))
            C1=1/(2*np.sqrt(q))*np.log(np.abs((x_d1[0]-p-np.sqrt(q))/(x_d1[0]-p+np.sqrt(q))))-b
            if ABS<0:
                for t in range(1,self.n+1+self.ahead+self.ahead1):
                    x1_pred[t-1]=(p+np.sqrt(q)-np.exp(2*np.sqrt(q)*(b*t**self.h+C1))*(np.sqrt(q)-p))/(1+np.exp(2*np.sqrt(q)*(b*t**self.h+C1)))
            else:
                for t in range(1,self.n+1+self.ahead+self.ahead1):
                    x1_pred[t-1]=(p+np.sqrt(q)+np.exp(2*np.sqrt(q)*(b*t**self.h+C1))*(np.sqrt(q)-p))/(1-np.exp(2*np.sqrt(q)*(b*t**self.h+C1)))
        elif q==0:
            C2=1/(x_d1[0]-p)-b
            for t in range(1,self.n+self.ahead+1+self.ahead1):
                x_1pred[t-1]=p-1/(b*t**h+C2)
        else:
            C3=1/np.sqrt(-q)*np.arctan((x_d1[0]-p)/np.sqrt(-q))-b
            for t in range(1,self.n+1+self.ahead+self.ahead1):
                x1_pred[t-1]=np.sqrt(-q)*np.tan(np.sqrt(-q)*(b*t**h+C3))+p
        A=self.fen_add(self.r,x1_pred)[1]
        x_pred=np.array(np.dot(np.linalg.inv(A),x1_pred))[0]
        df=pd.DataFrame()
        df['Actual']=x_d1
        df['Prediction']=x_pred[:-(self.ahead+self.ahead1)]
        df1=pd.DataFrame()
        df1['Actual']=x_t
        df1['Prediction']=x_pred[-(self.ahead+self.ahead1):-self.ahead1]
        return df,df1,x_pred[-(self.ahead1):]
    #拟合误差和测试误差
    def error(self):
        df=result(h,r)[0]
        df1=result(h,r)[1]
        error_llM=(np.mean(np.abs((df['Prediction']-x_d1)/x_d1)))*100
        error_llR=( np.sqrt(np.mean((df['Prediction']-x_d1)**2)))
        error_llA=(np.mean(np.abs(df['Prediction']-x_d1)))
        error_p1M=np.mean(np.abs((df1['Prediction']-x_t)/x_t))*100
        error_p1R=( np.sqrt(np.mean((df1['Prediction']-x_t)**2)))
        error_p1A=(np.mean(np.abs(df1['Prediction']-x_t)))
        return [error_llM,error_llA,error_llR],[error_p1M,error_p1R,error_p1A]

if __name__=="__main__":
    x=np.array([926.57,1233.97,1316.54,1288.61,1577.13,1961.39,2314.14,2543.81,2885.79,3141.00,3264.53,3328.96,3683.10,3504.40])
    x_d1=x[:-4]
    x_t=x[-4:]
    a=FDFGRM()
    print('模型的参数估计:\n{}'.format(a.fit(0.998,0.2)))
    print('模型拟合结果:\n{}'.format(a.result()[0]))
    print('模型预测结果:\n{}'.format(a.result()[1]))
    print('模型未来预测:\n{}'.format(a.result()[2]))
