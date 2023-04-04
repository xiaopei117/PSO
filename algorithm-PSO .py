# 基础版本——PSO
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
output_file = "PSO_algorithm.xlsx"

class PSO(object):
    # 1.读取数据，得到参数
    def __init__(self):
        self.v_max1 = 1
        self.v_min1 = -1                         # 粒子第一段速度取值
        self.v_max2 = 5
        self.v_min2 = -5                        # 粒子第二段速度取值
        self.T = 15                               # 迭代次数
        self.S = 50                               # 每代粒子群的粒子数
        self.r = 2                               # 变异数量
        self.dim1 = 252                          # 需求点为252个
        self.dim2 = 70                           # 自提点备选点有70个
        self.dim3 = 4                            # 网格仓备选点有4个
        self.dim4 = 40                           # 自提点枢纽
        self.dim5 = 2                            # 网格仓枢纽
        self.dim = self.dim2 + self.dim3         # 每个粒子的维度为74
        self.x_bound1 = [0, self.dim5]           # 粒子第1-4个位置的取值范围
        self.x_bound2 = [0, self.dim4]           # 粒子第4-70个位置的取值范围
        self.r1 = random.random()                # r1、r2取0-1之间的随机值
        self.r2 = random.random()
        self.b = 5000
        self.c = 280
        # (1) 获取自提点坐标
        df_ztd_axis = pd.read_excel("data.xlsx", sheet_name="自提点备选点")
        self.h_ff = np.array(df_ztd_axis.iloc[:, 2:4])  # (70,2)
        # (2）获取网格仓坐标
        df_wgc_axis = pd.read_excel("data.xlsx", sheet_name="网格仓备选点")
        self.h_gg = np.array(df_wgc_axis.iloc[:, 2:4])  # (4,2)
        # (1)取【中心仓-网格仓-自提点-需求点】距离矩阵
        df_zxc_wgc_dis = pd.read_excel("distance_data.xlsx", sheet_name="中心仓-网格仓距离矩阵", header=None)
        df_wgc_ztd_dis = pd.read_excel("distance_data.xlsx", sheet_name="网格仓-自提点距离矩阵")
        df_ztd_xq_dis = pd.read_excel("distance_data.xlsx", sheet_name="自提点-需求点距离矩阵")
        self.h_ab = df_zxc_wgc_dis.iloc[1:, 1:]   # 中心仓-网格仓——距离矩阵(1,4)
        self.h_bc = df_wgc_ztd_dis.iloc[:, 1:]    # 网格仓-自提点——距离矩阵(4,70)
        self.h_cl = df_ztd_xq_dis.iloc[:, 1:]     # 自提点—需求点——距离矩阵(70,252)
        # (2)取需求数据
        xql = pd.read_excel("distance_data.xlsx", sheet_name="需求量")
        self.d_ld = np.array(xql.iloc[:, 3:])     # (252,1)
        self.d_ld_all = xql["总需求量"].sum()      # 需求点的总需求量
        # (3)取中断概率
        q_b0 = pd.read_excel("distance_data.xlsx", sheet_name="网格仓中断概率")
        self.q_b = np.array(q_b0.iloc[:, 1:2])    # q_b,4*1矩阵
        self.q_b1 = np.array(q_b0.iloc[:, 2:])    # (1-q_b),4*1矩阵
        q_c0 = pd.read_excel("distance_data.xlsx", sheet_name="自提点中断概率")
        self.q_c = np.array(q_c0.iloc[:, 1:2])    # q_c,70*1矩阵
        self.q_c1 = np.array(q_c0.iloc[:, 2:])    # (1-q_c),70*1矩阵
        # (4)取时间矩阵
        df_zxc_wgc_dura = pd.read_excel("duration_data.xlsx", sheet_name="中心仓-网格仓时间矩阵")
        df_wgc_ztd_dura = pd.read_excel("duration_data.xlsx", sheet_name="网格仓-自提点时间矩阵")
        df_ztd_xq_dura = pd.read_excel("duration_data.xlsx", sheet_name="自提点-需求点时间矩阵")
        self.t_ab = df_zxc_wgc_dura.iloc[:, 1:]
        self.t_bc = df_wgc_ztd_dura.iloc[:, 1:]
        self.t_cl = df_ztd_xq_dura.iloc[:, 1:]
        # (5)取【网格仓-网格仓】【自提点-自提点】距离矩阵
        df_wgc_wgc_dis = pd.read_excel("distance_data01.xlsx", sheet_name="网格仓-网格仓距离矩阵")
        df_ztd_ztd_dis = pd.read_excel("distance_data01.xlsx", sheet_name="自提点-自提点距离矩阵")
        self.h_bb = np.array(df_wgc_wgc_dis.iloc[:, 1:])
        self.h_cc = np.array(df_ztd_ztd_dis.iloc[:, 1:])

    # 2.粒子转关键参数
    def particle_init(self,e1,e2):
        # (1) choice1、a1
        e11 = np.argsort(-e1)[:self.dim4]       # 按从小到大得出排序,选出最小的20个值对应的位置
        choice1 = e11                           # 0-3 （2，）
        choice1 = choice1.astype(np.int16)
        b1 = np.zeros((self.dim2))              # b1:(70,)
        b1[choice1[:]] = 1
        b1 = b1.astype(np.int16)
        # (2) choice2、b2
        e21 = np.argsort(-e2)[:self.dim5]       # 按从小到大得出排序,选出最小的20个值对应的位置
        choice2 = e21                           # 0-3 （2，）
        choice2 = choice2.astype(np.int16)
        b2 = np.zeros((self.dim3))              # 0/1 (4,)
        b2[choice2[:]] = 1
        b2 = b2.astype(np.int16)
        # (3) a1:自提点-需求点分配结果
        a1 = np.zeros((self.dim1)).astype(np.int16) # a1:(252,)，自提点-需求点分配结果
        a2 = np.zeros((self.dim2))                  # a2:(70,),每个自提点所辖的需求总量
        h_cl0 = np.array(self.h_cl.take(choice1,0)) # (70,252)
        # 按 1.距离 / 2.容量 进行自提点-需求点分配
        for e in range(self.dim4 * self.dim1):
            if np.sum(h_cl0[:,:] == 1000000) != self.dim1 * self.dim4 :
                a11_ = np.where(h_cl0 == np.min(h_cl0))[0][0]# 二维矩阵最小值所在行
                a11 = choice1[a11_]  # (自提点编号)
                a12 = np.where(h_cl0 == np.min(h_cl0))[1][0] # 二维矩阵最小值所在列(需求点编号)
                if a2[a11] + self.d_ld[a12,:] <= self.c - 50:
                    a1[a12] = a11                            # 将需求点a12分配给自提点a11
                    a2[a11] = a2[a11] + self.d_ld[a12,:]     # 自提点加上需求点需求量
                    h_cl0[:, a12] = 1000000
                elif self.c - 50 <= a2[a11] + self.d_ld[a12, :] <= self.c:
                    a1[a12] = a11                            # 将需求点a12分配给自提点a11
                    a2[a11] = a2[a11] + self.d_ld[a12, :]    # 自提点加上需求点需求量
                    h_cl0[:, a12] = 1000000
                    h_cl0[a11_, :] = 1000000
                elif a2[a11] + self.d_ld[a12,:] > self.c:
                    h_cl0[a11_, a12] = 1000000
        a1 = a1.astype(np.int16)
        # (4) a3:网格仓-自提点分配结果
        h_bc0 = np.array(self.h_bc.take(choice2, 0).take(choice1, 1))# 取出网格仓枢纽与自提点枢纽的距离
        a3 = np.zeros((self.dim2)).astype(np.int16)           # a3:(70,)，网格仓自提点分配结果
        a4 = np.zeros((self.dim3))                            # a4:(4,),每个网格仓所辖的需求总量
        # 按 1.距离 / 2.容量 进行自提点-需求点分配
        for e in range(self.dim5 * self.dim2):
            if np.sum(h_bc0[:, :] == 1000000) != self.dim4 * self.dim5:
                a31_ = np.where(h_bc0 == np.min(h_bc0))[0][0]  # 二维矩阵最小值所在行
                a31 = choice2[a31_]                            # (网格仓编号)
                a32_ = np.where(h_bc0 == np.min(h_bc0))[1][0]  # 二维矩阵最小值所在列(自提点编号)
                a32 = choice1[a32_]                            # (自提点编号)
                a33 = np.count_nonzero(a3)
                if a33 < self.dim4 :
                    if a4[a31] + a2[a32] <= self.b - 200:
                        a3[a32] = a31 + 1                      # 将自提点a32分配给网格仓a31
                        a4[a31] = a4[a31] + a2[a32]            # 自提点加上需求点需求量
                        h_bc0[:, a32_] = 1000000
                    elif self.b - 200 <= a4[a31] + a2[a32] <= self.b:
                        a3[a32] = a31 + 1                      # 将自提点a32分配给网格仓a31
                        a4[a31] = a4[a31] + a2[a32]            # 自提点加上需求点需求量
                        h_bc0[:, a32_] = 1000000
                        h_bc0[a31_, :] = 1000000
                    else:
                        h_bc0[a31_, a32_] = 1000000
        a3 = a3.astype(np.int16)
        # (5)Z_lc,Z_cb
        z_lc = np.zeros((self.dim1, self.dim2))      # a5对应Z_lc——252*70
        for a, adx1 in enumerate(a1):                # 把a3矩阵中，位置为i、值为idx的元素
            z_lc[a][adx1] = 1                    # a5:0/1 (252,70)
        z_cb = np.zeros((self.dim2, self.dim3))      # a6:0/1 (70,4)
        for b, bdx2 in enumerate(a3):
            if bdx2 != 0:
                z_cb[b][bdx2 - 1] = 1
        # (3) 计算备用枢纽矩阵
        # 1）计算需要的备选网格仓个数
        a21 = np.ceil((np.dot(np.dot(self.d_ld.T, np.dot(z_lc, z_cb)), self.q_b) / self.b)[0][0]).astype(np.int16)
        # 2）网格仓枢纽按照需要的备选网格仓个数聚类，找到离每簇网格仓聚类中心最近的备选网格仓，完成备选点分配
        a22 = np.zeros((self.dim5,2))
        a22[:] = self.h_gg[choice2[:]]             # 取出已选网格仓枢纽的坐标
        a23 = KMeans(n_clusters=a21, n_init=10, max_iter=300, tol=0.00001).fit(a22) # 聚类
        a24_ = a23.labels_                         # 得出聚类结果
        a25 = a23.cluster_centers_                 # 得出聚类中心
        a26 = np.arange(0,self.dim3)               # 找出未被选中的网格仓
        a26 = np.array(list(set(a26).difference(set(choice2))))
        a27_ = self.h_gg[a26[:]]                   # 未被选中网格仓的坐标
        a28 = np.zeros((a21,self.dim3 - self.dim5))
        a29 = np.zeros((a21)).astype(np.int16)
        for p in range(a21):                       # 求备选点和各枢纽之间的距离
            for q in range(self.dim3 - self.dim5):
                a28[p,q] = np.linalg.norm(a27_[q] - a25[p])
        for p in range(a21):                       # 选出离聚类中心最近的a41个网格仓备选点
            a29[p] = np.where(a28[p,] == np.min(a28[p,]))[0][0]
            a28[:,a29[p]] = 1000000
        a29_ = np.zeros((self.dim5))               # 转成每个网格仓枢纽的备选点(self.dim5,)
        for p in range(a21):
            a29_[np.where(a24_[:] == p)] = (a26[a29])[p] + 1
        a29_ = a29_.astype(np.int16)
        a27 = np.zeros((self.dim2))                # (self.dim2,)
        for w in range(self.dim5):
            a27[np.where(a3[:]-1 == choice2[w])[0]] = a29_[w]
        a27 = a27.astype(np.int16)

        # 3）计算需要的备选自提点个数
        a41 = np.ceil((np.dot(np.dot(self.d_ld.T, z_lc),self.q_c) / self.c)[0][0]).astype(np.int16)
        # 4）自提点枢纽按照需要的备选自提点个数聚类，找到离每簇自提点聚类中心最近的备选自提点，完成备选点分配
        a42 = np.zeros((self.dim4, 2))              # (35,2)
        a42[:] = self.h_ff[choice1[:]]
        a43 = KMeans(n_clusters= a41, n_init=10, max_iter=300, tol=0.00001).fit(a42)  # 聚类
        a44 = a43.labels_                           # 得出聚类结果
        a45 = a43.cluster_centers_                  # 得出聚类中心
        # 找出未被选中的自提点
        a46 = np.arange(0, self.dim2)
        a46 = np.array(list(set(a46).difference(set(choice1))))
        a47 = self.h_ff[a46[:]]
        a48 = np.zeros((a41, self.dim2 - self.dim4))
        a49 = np.zeros((a41)).astype(np.int16)
        for s in range(a41):                        # 选出离聚类中心最近的a41个
            for t in range(self.dim2 - self.dim4):
                a48[s,t] = np.linalg.norm(a47[t] - a45[s])
        for s in range(a41):
            a49[s] = np.where(a48[s,] == np.min(a48[s,]))[0][0]
            a48[:,a49[s]] = 1000000
        a49_ = np.zeros((self.dim4))
        for s in range(a41):
             a49_[np.where(a44[:] == s)] = (a46[a49])[s]
        a49_ = a49_.astype(np.int16)
        a24 = np.zeros((self.dim1))                 # (self.dim1,),每个需求点应该被分配给哪些备选自提点
        for o in range(self.dim4):
            a24[np.where(a1[:]  == choice1[o])[0]] = a49_[o]
        a24 = a24.astype(np.int16)
        # (6) 将a24转化为2维矩阵u_cs,将a27转化为2维矩阵u_bs
        u_bs = np.zeros((self.dim2, self.dim3))     # u_bs:(70,4)
        for n, ndx2 in enumerate(a27):
            if a27[n] != 0:
                u_bs[n][ndx2 - 1] = 1
        u_cs = np.zeros((self.dim1, self.dim2))     # u_cs:(252,70)
        for m, mdx1 in enumerate(a24):
            u_cs[m][mdx1] = 1

        # (7) 自提点中断时(u_cs) - a3 依据(choice1 - a49_ )变形 - z_cb 变形为z_cb_
        a3_ = np.zeros((self.dim2))
        a3_[a49_[:]] = a3[choice1[:]]
        a3_ = a3_.astype(np.int16)
        z_cb_ = np.zeros((self.dim2, self.dim3))  # a6:0/1 (70,4)
        for z, zdx2 in enumerate(a3_):
            if zdx2 != 0:
                z_cb_[z][zdx2 - 1] = 1

        # (8) 自提点网格仓都中断时，依据a27得来的u_bs需变形为u_bs_
        a490 = a46[a49]
        a27_ = np.zeros((self.dim2))
        c2 = np.zeros((len(a490)))
        for x in range(len(a490)):
            c1 = a3[choice1[np.where( a49_[:] == a490[x])[0]][0]]
            c2[x] = a29_[np.where(choice2[:] == c1-1)[0][0]]
        a27_[a490[:]] = c2[:]
        a27_ = a27_.astype(np.int16)
        u_bs_ = np.zeros((self.dim2, self.dim3))  # u_bs:(70,4)
        for n, ndx2 in enumerate(a27_):
            if a27_[n] != 0:
                u_bs_[n][ndx2 - 1] = 1
        return b1, b2, choice1, choice2, z_lc, z_cb, u_bs, u_cs, a24, a27, a1, a3, a21, a41, z_cb_, u_bs_, a27_

    # 3 适应度函数
    # 3.1fun1指稳定运营情形下的成本目标
    def fun1(self, z_cb, z_lc, u_bs, u_cs, a24, a27, a3, a21, a41, z_cb_, u_bs_, a27_):
        g_a = 23333                              # 租金是35元/平*月；共2万平，每月30天
        g_b = 5000                               # 网格仓日租金5000
        g_c = 20                                 # 自提点日租金20
        g_b_ = 500                               # 备选网格仓日征用成本
        g_c_ = 10                                # 备选自提点日征用成本
        e_ad = 0.09                              # 中心仓非生鲜品类/生鲜品类分拣单价
        e_bd = 0.1                               # 网格仓非生鲜品类/生鲜品类分拣单价
        e_cd = 0.05                              # 自提点非生鲜品类/生鲜品类分拣单价
        n_b = 500                                # 网格仓寻源成本；元/个
        g_t = 300                                # 司机费用-300元/趟
        g_0 = 30                                 # 车辆折旧费1——30元/趟
        g_r = 20                                 # 车辆折旧费2——0.01
        Q_0 = 4000                               # 4.2米大货车最大装载量/件
        Q_k = 900                                # 小货车最大装载量/件
        alpha = 0.6                              # 运费折扣因子
        m_d1 = 0.0003                            # 中心仓-网格仓段：非生鲜/生鲜品类的单位运输成本；元/件·米
        m_d2 = 0.0005                            # 网格仓-自提点段：非生鲜/生鲜品类的单位运输成本；元/件·米
        m_l = 0.001                              # 用户自提成本；元/件·米
        # 3.2 目标函数
        # 目标函数1——稳定运营下的成本
        # 1.C11-中心仓运营成本
        C11 = g_a + self.d_ld_all * e_ad
        # 2.C12-中心仓运输成本
        z_cb01 = np.zeros((self.dim2, self.dim3))# self.z_cb01是70*4的矩阵
        for i in range(z_cb.shape[0]):           # z_cb是70*4的矩阵
            for j in range(z_cb.shape[1]):
                if z_cb[i, j] == 1:
                    z_cb01[i, j] = z_cb[i, j] * self.q_c1[i, 0] * self.q_b1[j, 0]
        C12 = np.ceil(self.d_ld_all / Q_0) * (g_t + g_0)+ alpha * m_d1 * np.dot(np.dot(self.d_ld.T, np.dot(z_lc ,z_cb01)) ,(self.h_ab.T))[0,0]
        # 3.C13-网格仓招标建设成本
        C13 = n_b * self.dim5
        # 4.C14-网格仓运营成本+备选网格仓征用成本
        C14 = g_b * self.dim5 + e_bd * self.d_ld_all + a21 * g_b_
        # 5.C15-网格仓运输成本
        C150 = np.sum(np.ceil( np.dot(self.d_ld.T, np.dot(z_lc ,z_cb))/Q_k )* (g_t + g_r))
        C151 = (alpha * m_d2 * np.dot(np.dot((np.dot(self.d_ld.T ,z_lc) * self.q_c1.T) , self.h_bc.T) , self.q_b1))[0,0]
        C15 = C150 + C151
        # 6.C16-自提点运营成本+备选自提点征用成本
        C16 = g_c * self.dim4 + e_cd * self.d_ld_all  + a41 * g_c_
        # 7.C7-用户的自提成本
        C17 = m_l * np.dot(np.dot(self.d_ld.T , (z_lc * self.h_cl.T)) , self.q_c1)[0][0]
        C1 = C11 + C12 + C13 + C14 + C15 + C16 + C17
        # 目标函数2——中断情形下的成本
        # 1.第一种中断情形下的成本-仅网格仓中断
        # (1)C2_11-中心仓运输成本
        u_bs01 = np.zeros((self.dim2, self.dim3))     # u_bs01是70*4矩阵;self.u_bs01 = self.u_bs * q_c * q_b * p_b1
        u_bs02 = np.zeros((self.dim2, self.dim3))     # u_bs02是70*4矩阵;self.u_bs02 = self.u_bs * q_b * p_b1
        z_lc01 = np.zeros((self.dim1, self.dim2))     # self.z_lc01是252*70的矩阵
        for i in range(0,self.dim2):                  # i:0-69
            if a3[i] != 0:                            # a4和a30都是(70,)的矩阵
                for j in range(0,self.dim3):          # j:0-3
                    for m in range(0,self.dim1):      # m:0-251
                        C21 = a27[i]                  # 求出p_b1/1-p_b;找出a4中对应索引的a30中的网格仓备选点序号
                        u_bs01[i,j] = u_bs[i,j] * self.q_c1[i,0] * self.q_b[j,0] * self.q_b1[C21-1,0]
                        u_bs02[i,j] = u_bs[i,j] * self.q_b[j,0] * self.q_b1[C21-1,0]
                        z_lc01[m,i] = z_lc[m,i] * self.q_b[j,0] * self.q_b1[C21-1,0]
        C2_11 = (alpha * m_d1 * np.dot(np.dot(np.dot(self.d_ld.T, z_lc) ,u_bs01) ,(self.h_ab.T)))[0][0]
        # (2)C2-12-网格仓运输成本
        C2_12 = (alpha * m_d2 * np.dot(np.dot(np.dot(np.dot(self.d_ld.T, z_lc) ,u_bs02) ,(self.h_bc)) ,self.q_c1))[0][0]
        # (3)C2-13-用户自提成本
        C2_13 = (m_l * np.dot(np.dot(self.d_ld.T , (z_lc01 * self.h_cl.T)) , self.q_c1))[0][0]
        # 2.第二种中断情形下的成本-仅自提点中断
        # (1)C2_21-中心仓运输成本
        z_cb02 = np.zeros((self.dim2, self.dim3))     # self.z_cb02是70*4矩阵;self.z_cb02 = self.z_cb * q_c * q_b * p_b1
        u_cs01 = np.zeros((self.dim1, self.dim2))     # self.u_cs01是252*70矩阵;self.z_cb03 = self.z_cb * q_b * p_b1
        u_cs02 = np.zeros((self.dim1, self.dim2))     # self.z_lc02是252*70的矩阵
        for m in range(0, self.dim1):                 # m:0-251
            for i in range(0, self.dim2):             # i:0-69
                for j in range(0, self.dim3):         # j:0-3
                    C22 = a24[m,]                     # 求出p_b1/1-p_b;找出a4中对应索引的a30中的网格仓备选点序号
                    z_cb02[i, j] = z_cb_[i, j] * self.q_c[i, 0] * self.q_b1[j, 0] * self.q_c1[C22, 0]
                    u_cs01[m, i] = u_cs[m, i] * self.q_b1[j, 0] * self.q_c1[C22, 0]
                    u_cs02[m, i] = u_cs[m, i] * self.q_b1[j, 0] * self.q_c1[C22, 0]
        C2_21 = (alpha * m_d1 * np.dot(np.dot(np.dot(self.d_ld.T, u_cs), z_cb02), (self.h_ab.T)))[0][0]
        # (2)C2-22-网格仓运输成本
        C2_22 = (alpha * m_d2 * np.dot(np.dot(np.dot(np.dot(self.d_ld.T, u_cs01), z_cb_), (self.h_bc)), self.q_c))[0][0]
        # (3)C2-23-用户自提成本
        C2_23 = (m_l * np.dot(np.dot(self.d_ld.T, (u_cs02 * self.h_cl.T)),self. q_c))[0][0]
        # 3.第三种中断情形下的成本-仅初始网格仓及初始自提点中断
        # (1)C2_31-中心仓运输成本
        u_bs03 = np.zeros((self.dim2, self.dim3))     # self.u_bs03是70*4矩阵
        u_cs03 = np.zeros((self.dim1, self.dim2))     # self.u_cs03是252*70矩阵
        u_cs04 = np.zeros((self.dim1, self.dim2))     # self.u_cs04是252*70矩阵
        for i in range(0, self.dim2):                 # i:0-69
            if a27_[i,] != 0:
                for j in range(0, self.dim3):         # j:0-3
                    for m in range(0, self.dim1):     # m:0-251
                        C21 = a27_[i]
                        C22 = a24[m]
                        u_bs03[i, j] = u_bs_[i, j] * self.q_c[i, 0] * self.q_b[j, 0] * self.q_b1[C21-1,0] * self.q_c1[C22, 0]
                        u_cs03[m, i] = u_cs[m, i] * self.q_b[j, 0] * self.q_b1[C21-1,0] * self.q_c1[C22, 0]
                        u_cs04[m, i] = u_cs[m, i] * self.q_b[j, 0] * self.q_b1[C21-1,0] * self.q_c1[C22, 0]
        C2_31 = (alpha * m_d1 * np.dot(np.dot(np.dot(self.d_ld.T, u_cs), u_bs03), (self.h_ab.T)))[0][0]
        # (2)C2-32-网格仓运输成本
        C2_32 = (alpha * m_d2 * np.dot(np.dot(np.dot(np.dot(self.d_ld.T, u_cs03), u_bs_), (self.h_bc)), self.q_c))[0][0]
        # (3)C2-33-用户自提成本
        C2_33 = (m_l * np.dot(np.dot(self.d_ld.T, (u_cs04 * self.h_cl.T)), self.q_c))[0][0]
        # 4.第四种中断情形下的成本-需求损失成本
        z_cb03 = np.zeros((self.dim2, self.dim3))
        for i in range(0, self.dim2):                # i:0-69
            for j in range(0, self.dim3):            # j:0-3
                for m in range(0, self.dim1):        # m:0-251
                    if a3[i,] != 0:                  # a4和a30都是(70,)的矩阵
                        C21 = a27[i]
                        C22 = a24[m,]
                        z_cb03[i,j] = z_cb[i,j] * (self.q_b[C21-1,0] + self.q_c[i, 0] * self.q_c1[C22, 0]/self.q_b[j, 0] + self.q_b[C21-1, 0] * self.q_c[i, 0] * self.q_c[C22, 0])
        C24 = (200 * m_d2 * np.dot(np.dot(np.dot(self.d_ld.T, z_lc), z_cb03), self.q_b))[0][0]
        C2 = C2_11 + C2_12 + C2_13 + C2_21 + C2_22 + C2_23 + C2_31 + C2_32 + C2_33 + C24
        C = C1 + C2                                   # 总成本
        return C

    # 4.主函数>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def pso_main(self):
        w = 0.6                                                    # w
        c1 = 1.5                                                   # C1自学习因子
        c2 = 1.5                                                   # C2社会学习因子
        # (1) 接下来把关键变量求出来
        N = []                                                     # N: (S,144);存每一代粒子的位置x
        num_N = []                                                 # 存每一代每个粒子的适应度值
        for o in range(self.S):
            # (1 先随机初始化,运行second函数更新目标函数相关变量，得初始化下的目标函数关键变量
            e2 = np.random.uniform(0, self.dim5, (self.dim3))      # (4, ),取值为0-2之间
            e1 = np.random.uniform(0, self.dim4, (self.dim2))      # (70,),取值在0-30之间
            b1, b2, choice1, choice2, z_lc, z_cb, u_bs, u_cs, a24, a27, a1, a3, a21, a41, z_cb_, u_bs_, a27_ = self.particle_init(e1,e2)
            # (2 拼接粒子
            x = np.concatenate((e2, e1), axis=0)           # 将e2(4,)、a2(70,)横向拼接;(74,)
            N.append(x)
            # (3 求初代的适应度
            fitness1 = self.fun1(z_cb, z_lc, u_bs, u_cs, a24, a27, a3, a21, a41, z_cb_, u_bs_, a27_)
            num_N.append(fitness1.squeeze())
        num_N = np.array(num_N)                                     # 将self.num_N、N转成array;sum_N —— 存fitness适应度值
        print("最初的num_N",num_N)
        N = np.array(N)
        # (2)空列表存初代的局部最优值和全局最优值
        g_best_num = []                                             # 建立[局部最优值列表]
        pg_best_num = []                                            # 建立[全局最优值列表]
        g_best_state= []                                            # 建立[局部最优粒子位置列表]
        pg_best_state = []                                          # 建立[全局最优粒子位置列表]
        g_best_index = []
        g_best_num1 = np.min(num_N)                                 # 选出粒子群此次迭代中最小的值(局部最优值)
        g_best_num.append(g_best_num1.copy())                              # 把初代的最优值存入【局部最优值列表】
        pg_best_num.append(g_best_num1.copy())                             # 把初代的最优值存入【全局最优值列表】
        g_best_index1 = np.where(num_N == g_best_num1)[0][0]        # 局部最优粒子的索引(list用。index；array用np.where)
        g_best_index.append(g_best_index1.copy())
        g_best_state1 = N[g_best_index1]
        g_best_state.append(g_best_state1.copy())
        pg_best_state1 = N[g_best_index1]
        pg_best_state.append(pg_best_state1.copy())
        pg = N[g_best_index1]                                       # 全局最优粒子pg(74，)
        # (3)算迭代初代的自适应参数——此版本不变
        # (4)迭代+求最优
        N2 = N                                                      # 迭代时用N2
        num_N2 = num_N                                              # 存迭代过程中的适应度值
        # 0) 随机初始化粒子的速度
        v = np.zeros((self.S, self.dim))                            # v:(S,144)
        for tt in tqdm(range(0,self.T)):                            # 每一次迭代都有进度条显示
            # 1）迭代过程
            for jj in range(0, self.S):                             # 每代j个粒子
                # 1.第一段(0,4)和第二段(4,74)进行粒子群迭代操作
                # 1.1 更新速度
                v[jj][:self.dim3] = w * v[jj][:self.dim3] + c1 * random.random() * (N[jj][:self.dim3] - N2[jj][:self.dim3]) + c2 * random.random() * (pg[:self.dim3] - N2[jj][:self.dim3])
                v[jj][self.dim3:self.dim3 + self.dim2] = w * v[jj][self.dim3:self.dim3 + self.dim2] + c1 * random.random() * (N[jj][self.dim3:self.dim3 + self.dim2] - N2[jj][self.dim3:self.dim3 + self.dim2]) + c2 * random.random() * (pg[self.dim3:self.dim3 + self.dim2] - N2[jj][self.dim3:self.dim3 + self.dim2])
                # 1.2 限制速度最大值最小值
                v[jj][:self.dim3] = np.clip(v[jj][:self.dim3], self.v_min1, self.v_max1)
                v[jj][self.dim3:self.dim3 + self.dim2] = np.clip(v[jj][self.dim3:self.dim3 + self.dim2], self.v_min2, self.v_max2)
                # 1.3 更新位置
                N2[jj][:self.dim3] = N2[jj][:self.dim3] + v[jj][:self.dim3]
                N2[jj][self.dim3:self.dim3 + self.dim2] = N2[jj][self.dim3:self.dim3 + self.dim2] + v[jj][self.dim3:self.dim3 + self.dim2]
                # 1.4 限制更新位置的最大/最小值
                N2[jj][:self.dim3] = np.where(N2[jj][:self.dim3] > self.x_bound1[1], 2 * self.x_bound1[1] - N2[jj][:self.dim3] , N2[jj][:self.dim3])
                N2[jj][:self.dim3] = np.where(N2[jj][:self.dim3] < self.x_bound1[0], - N2[jj][:self.dim3] , N2[jj][:self.dim3])
                N2[jj][:self.dim3:self.dim3 + self.dim2] = np.where(N2[jj][:self.dim3:self.dim3 + self.dim2] > self.x_bound2[1], 2 * self.x_bound2[1] - N2[jj][:self.dim3:self.dim3 + self.dim2] , N2[jj][:self.dim3:self.dim3 + self.dim2])
                N2[jj][:self.dim3:self.dim3 + self.dim2] = np.where(N2[jj][:self.dim3:self.dim3 + self.dim2] < self.x_bound2[0], - N2[jj][:self.dim3:self.dim3 + self.dim2] , N2[jj][:self.dim3:self.dim3 + self.dim2])
            # 2）得迭代后的关键参数
                e2 = N2[jj][ :self.dim3]
                e1 = N2[jj][self.dim3 : self.dim3 + self.dim2]
                e2 = e2.astype(np.int16)
                b1, b2, choice1, choice2, z_lc, z_cb, u_bs, u_cs, a24, a27, a1, a3, a21, a41, z_cb_, u_bs_, a27_ = self.particle_init(e1,e2)
            # 3）得迭代后的适应度
                fitness2 = self.fun1(z_cb, z_lc, u_bs, u_cs, a24, a27, a3, a21, a41, z_cb_, u_bs_, a27_)
                fitness2 = fitness2.squeeze().item()
                num_N2[jj] = fitness2
            print("num_N2", num_N2)

            # (5) 个体最优值列表替换——将个体最优值存在self.num_N里
            for pp in range(self.S):
                if num_N2[pp] < num_N[pp]:
                    num_N[pp] = num_N2[pp]
                    N[pp] = N2[pp]
            # (6) 求出局部最优粒子及其适应度值
            g_best_num2 = np.min(np.array(num_N2))                        # 选出粒子群此次迭代中最小的值(局部最优值)
            g_best_num.append(g_best_num2.copy())                                # 各局部最优值存入列表
            g_best_index2 = np.where(np.array(num_N2) == g_best_num2)[0][0]# 局部最优粒子的索引
            g_best_index.append(g_best_index2.copy())
            g_best_state2 = N2[g_best_index2]                             # 取出局部最优的粒子位置
            g_best_state.append(g_best_state2.copy())                     # 局部最优粒子位置存入列表
            # (7) 求全局最优值——拿这一代的最优值跟全局最优做比较,若这一代最优值更优，则成为全局最优值
            pg_best_num2 = np.min(g_best_num)                             # 全局最优粒子适应度
            pg_best_num.append(pg_best_num2.copy())                       # 全局最优粒子适应度
            # print(np.where(np.array(g_best_num) == pg_best_num2)[0])
            pg = g_best_state[np.where(np.array(g_best_num) == pg_best_num2)[0][0]]
            pg_best_state.append(pg)                                      # 全局最优粒子位置列表
            print("g_best_num", g_best_num)
            print("pg_best_num", pg_best_num)

            print('第' + str(tt) + '次迭代：局部最优解位置在' + str(g_best_index2) + '，全局最优解的适应度值为：' + str(pg_best_num2))  # 输出指示
            print("...............")
            # (9) 画分配图
            if tt == self.T - 1:
                e2 = pg[: self.dim3]
                e1 = pg[self.dim3: self.dim3 + self.dim2]
                e1 = e1.astype(np.int16)
                b1, b2, choice1, choice2, z_lc, z_cb, u_bs, u_cs, a24, a27, a1, a3, a21, a41, z_cb_, u_bs_, a27_ = self.particle_init(e1, e2)
                a1 = pd.DataFrame(a1)
                a3 = pd.DataFrame(a3)
                choice1 = pd.DataFrame(choice1)
                choice2 = pd.DataFrame(choice2)
                a24 = pd.DataFrame(a24)
                a27 = pd.DataFrame(a27)
                # (6) 数据存表
                dt = [r for r in range(0, self.T + 1)]
                dt = pd.DataFrame(dt)
                best = pd.DataFrame(pg_best_num)
                df = pd.concat([dt, best, a1, a3, choice1, choice2, a24, a27], axis=1)
                print(df)
                writer = pd.ExcelWriter(output_file)
                df.to_excel(writer, sheet_name="algorithm", encoding="utf-8")
                writer.close()
        # (10) 画图
        plt.plot([t for t in range(self.T+1)], pg_best_num)               # 画每一代的最优值
        plt.ylabel('fitness')                                             # y轴是每代的适应度值
        plt.xlabel('iter nums')                                           # x轴是迭代次数
        plt.title('PSO')                                                  # 标题是粒子群适应度趋势
        plt.show()                                                        # 画图

pso = PSO()
pso.pso_main()