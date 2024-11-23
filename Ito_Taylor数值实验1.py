import numpy as np
from sklearn.linear_model import LinearRegression as LR
from Wiener_Process import Wiener_Process as W_P
from Euler_Maruyama_Method import Euler_Maruyama as EM
from Milstein_Method import Milstein as M
from Ito_Taylor_Method import Ito_Taylor15 as IT15
from ex_1 import a,dat,dax,ddax,b,dbt,dbx,ddbx,true_solver1
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False # 修复负号问题
#ex1
t0_1 = 0#初始时间
T_1 = 1#终端时间
y0_1 = 1#初始值
ndt_1 = 2 ** 12#时间步数
ndt_1_= 2 ** np.arange(4,9)#时间步数
M_1 = 10000#模拟路径次数
t_1,Wt_1,dWt_1,Nor_1 = W_P(t0_1,T_1,ndt_1,M_1)#生成Wiener过程
y_e_m_1 = []#Eu-Maruyama法法求解的解
y_m_1 = []#Milstein法求解的解
y_i_t_1 = []#
error_s_e_m_1 = []
error_w_e_m_1 = []
error_s_m_1 = []
error_w_m_1 = []
error_s_i_t_1 = []
error_w_i_t_1= []
for i in range(len(ndt_1_)):
    Wt = Wt_1[:,np.arange(0,ndt_1+1,ndt_1//ndt_1_[i])]
    dWt = np.diff(Wt,1)
    Nor = dWt/(np.sqrt((T_1 - t0_1)/ndt_1_[i]))
    y_e_m_ = EM(t0_1,T_1,y0_1,ndt_1_[i],M_1,a,b,dWt)
    y_e_m_1.append(y_e_m_)
    y_m_ = M(t0_1,T_1,y0_1,ndt_1_[i],M_1,a,b,dbx,dWt)
    y_m_1.append(y_m_)
    y_i_t_ = IT15(t0_1, T_1, y0_1, ndt_1_[i], M_1, a, b, dat, dbt, dax, dbx, ddax, ddbx, dWt,Nor)
    y_i_t_1.append(y_i_t_)
    t = t_1[np.arange(0,ndt_1+1,ndt_1//ndt_1_[i])]
    t_v = true_solver1(t,Wt)
    e_s_i_t = np.mean(abs(y_i_t_[:,-1] - t_v[:,-1]))
    error_s_i_t_1.append(e_s_i_t)
    e_s_e_m = np.mean(abs(y_e_m_[:,-1]-t_v[:,-1]))
    #e_s_e_m = np.mean((abs(y_e_m_[:,-1] - t_v[:,-1])))
    error_s_e_m_1.append(e_s_e_m)
    e_s_m = np.mean(abs(y_m_[:,-1] - t_v[:,-1]))
    error_s_m_1.append(e_s_m)
    e_w_e_m = abs(np.mean(y_e_m_[:,-1]) - np.mean(t_v[:,-1]))
    error_w_e_m_1.append(e_w_e_m)
    e_w_m = abs(np.mean(y_m_[:,-1]) - np.mean(t_v[:,-1]))
    error_w_m_1.append(e_w_m)
    e_w_i_t = abs(np.mean(y_i_t_[:,-1]) - np.mean(t_v[:,-1]))
    error_w_i_t_1.append(e_w_i_t)

x = np.log((T_1-t0_1)/ndt_1_)
y1 = np.log(error_s_e_m_1)
y2 = np.log(error_s_m_1)
y3 = np.log(error_s_i_t_1)
y4 = np.log(error_w_e_m_1)
y5 = np.log(error_w_m_1)
y6 = np.log(error_w_i_t_1)
fig1 = plt.figure()

ax1 = fig1.subplots(1,1)
ax1.plot(x,y1,marker='o',ls=':')
ax1.plot(x,y2,marker='x',ls='--')
ax1.plot(x,y3,marker='v',ls='-.')
ax1.set_xlabel('ln$\Delta$t')
ax1.set_ylabel(r'ln$E|\mathcal{X}_{1}-X_{1}|$')
fig1.legend(['$Euler-Maruyama$法','$Milstein$法','强收敛阶为1.5的$It\^o-Taylor$法'],bbox_to_anchor=(1, 1),prop={"size":20})
#plt.axis('equal')


fig2 = plt.figure()

ax2 = fig2.subplots(1,1)
ax2.plot(x,y4,marker='o',ls=':')
ax2.plot(x,y5,marker='x',ls='--')
ax2.plot(x,y6,marker='v',ls='-.')
ax2.set_xlabel('ln$\Delta$t')
ax2.set_ylabel(r'ln$|E(\mathcal{X}_{1}-X_{1})|$')
#ax.set_ylabel(r'ln$|E\mathcal{X}_{1}-EX_{1}|$')
fig2.legend(['$Euler-Maruyama$法','$Milstein$法','强收敛阶为1.5的$It\^o-Taylor$法'],bbox_to_anchor=(1, 1),prop={"size":20})
plt.show(block=True)

model1 = LR()
model1.fit(x.reshape(-1,1),y1.reshape(-1,1))
model2 = LR()
model2.fit(x.reshape(-1,1),y2.reshape(-1,1))
model3 = LR()
model3.fit(x.reshape(-1,1),y3.reshape(-1,1))
model4 = LR()
model4.fit(x.reshape(-1,1),y4.reshape(-1,1))
model5 = LR()
model5.fit(x.reshape(-1,1),y5.reshape(-1,1))
model6 = LR()
model6.fit(x.reshape(-1,1),y6.reshape(-1,1))
print(model1.coef_)
print(model2.coef_)
print(model3.coef_)
print(model4.coef_)
print(model5.coef_)
print(model6.coef_)