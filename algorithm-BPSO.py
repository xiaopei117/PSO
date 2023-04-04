# 优化版本2—需求点/自提点聚类+参数自适应+交叉变异——BPSO
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
output_file = "BPSO_algorithm.xlsx"

class PSO(object):
    # 1.读取数据，得到参数
    def __init__(self):
        self.v_max1 = 1
        self.v_min1 = -1                         # 粒子第一段速度取值
        self.v_max2 = 1
        self.v_min2 = -1                         # 粒子第二段速度取值
        self.T = 15                              # 迭代次数
        self.S = 50                              # 每代粒子群的粒子数
        self.dim1 = 252                          # 需求点为252个
        self.dim2 = 70                           # 自提点备选点
        self.dim3 = 4                            # 网格仓备选点有4个
        self.dim4 = 40                           # 自提点枢纽数量
        self.dim5 = 2                            # 网格仓枢纽数量
        self.dim = self.dim2 + self.dim3         # 每个粒子的维度
        self.w_max = 0.7
        self.w_min = 0.4
        self.P_max = 1
        self.P_min = 0.1                         # 迭代中每代粒子第三段的最大/最小变异概率
        self.x_bound1 = [0, self.dim5]           # 粒子第1-4个位置的取值范围
        self.x_bound2 = [1, self.dim5]           # 粒子第4-70个位置的取值范围
        self.r1 = random.random()                # r1、r2取0-1之间的随机值
        self.r2 = random.random()
        self.b = 5000                            # 网格仓容量限制
        self.c = 280                             # 自提点容量限制
         # (1) 获取需求点坐标
        df_xqd_axis = pd.read_excel("data.xlsx", sheet_name="需求点")
        self.h_ee = np.array(df_xqd_axis.iloc[:, 2:4])  # (252,2)
        # (2) 获取自提点坐标
        df_ztd_axis = pd.read_excel("data.xlsx", sheet_name="自提点备选点")
        self.h_ff = np.array(df_ztd_axis.iloc[:, 2:4])  # (70,2)
        # (3) 获取网格仓坐标
        df_wgc_axis = pd.read_excel("data.xlsx", sheet_name="网格仓备选点")
        self.h_gg = np.array(df_wgc_axis.iloc[:, 2:4])  # (4,2)
        # (4)取【中心仓-网格仓-自提点-需求点】距离矩阵
        df_zxc_wgc_dis = pd.read_excel("distance_data.xlsx", sheet_name="中心仓-网格仓距离矩阵", header=None)
        df_wgc_ztd_dis = pd.read_excel("distance_data.xlsx", sheet_name="网格仓-自提点距离矩阵")
        df_ztd_xq_dis = pd.read_excel("distance_data.xlsx", sheet_name="自提点-需求点距离矩阵")
        self.h_ab = df_zxc_wgc_dis.iloc[1:, 1:]    # 中心仓-网格仓——距离矩阵
        self.h_bc = df_wgc_ztd_dis.iloc[:, 1:]     # 网格仓-自提点——距离矩阵
        self.h_cl = df_ztd_xq_dis.iloc[:, 1:]      # 自提点—需求点——距离矩阵
        # (5)取需求数据
        xql = pd.read_excel("distance_data.xlsx", sheet_name="需求量")
        self.d_ld = xql.iloc[:, 3:]                # 需求点的总需求量
        self.d_ld_all = xql["总需求量"].sum()
        # (6)取中断概率
        q_b0 = pd.read_excel("distance_data.xlsx", sheet_name="网格仓中断概率")
        self.q_b = np.array(q_b0.iloc[:, 1:2])      # q_b,4*1矩阵
        self.q_b1 = np.array(q_b0.iloc[:, 2:])     # (1-q_b),4*1矩阵
        q_c0 = pd.read_excel("distance_data.xlsx", sheet_name="自提点中断概率")
        self.q_c = np.array(q_c0.iloc[:, 1:2])      # q_c,70*1矩阵
        self.q_c1 = np.array(q_c0.iloc[:, 2:])     # (1-q_c),70*1矩阵
        # (7)取时间矩阵
        df_zxc_wgc_dura = pd.read_excel("duration_data.xlsx", sheet_name="中心仓-网格仓时间矩阵")
        df_wgc_ztd_dura = pd.read_excel("duration_data.xlsx", sheet_name="网格仓-自提点时间矩阵")
        df_ztd_xq_dura = pd.read_excel("duration_data.xlsx", sheet_name="自提点-需求点时间矩阵")
        self.t_ab = df_zxc_wgc_dura.iloc[:, 1:]
        self.t_bc = df_wgc_ztd_dura.iloc[:, 1:]
        self.t_cl = df_ztd_xq_dura.iloc[:, 1:]
        # (8)取【网格仓-网格仓】【自提点-自提点】距离矩阵
        df_wgc_wgc_dis = pd.read_excel("distance_data01.xlsx", sheet_name="网格仓-网格仓距离矩阵")
        df_ztd_ztd_dis = pd.read_excel("distance_data01.xlsx", sheet_name="自提点-自提点距离矩阵")
        self.h_bb = np.array(df_wgc_wgc_dis.iloc[:, 1:])
        self.h_cc = np.array(df_ztd_ztd_dis.iloc[:, 1:])

    # 2.粒子转关键参数
    def particle_init_0(self):
        # 0.需求点聚类\a1、cluster_center
        kmeans = KMeans(n_clusters=self.dim4, n_init=10, max_iter=300, tol=0.00001).fit(self.h_ee)
        a1 = kmeans.labels_  # a1:0-29  (252,)
        a1 = a1.astype(np.int16)
        cluster_center1 = kmeans.cluster_centers_   # 返回聚类中心的坐标(20,2)
        return a1,cluster_center1

    def particle_init_1(self,a1,cluster_center1):
        # 1.求a1对应的a3、choice1、b1、z_lc
        a3 = np.zeros((self.dim1))                  # a3:0-69 (252,)
        a3_ = np.zeros((self.dim4, self.dim2))
        for aa in range(self.dim4):                 # 求每个簇的中心点离选中的自提点枢纽的距离矩阵，放入a3_（20，70）
            a31 = cluster_center1[aa, :]
            a33 = np.zeros((self.dim2))
            for bb in range(self.dim2):
                a34 = np.linalg.norm(a31 - self.h_ff[bb])
                a33[bb] = a34
            a3_[aa] = a33
        a36 = np.zeros((self.dim4))
        for dd in range(self.dim4):
            a36[dd] = np.sum(a1[:] == dd)
        a37 = np.argsort(-a36)                      # 求a1中各簇中的需求点数量，并按从大到小排序
        # 2.choice1、a3、b1、z_lc
        choice1 = np.zeros((self.dim4))
        for cc in range(self.dim4):                 # 求a3_中每行最小的值，每行只取一个
            a38 = a37[cc]                           # 取第n簇
            a39 = np.where(a3_[a38, :] == np.min(a3_[a38, :]))[0][0]
            choice1[cc] = a39                       # 第n簇中距离最小值；即第n簇对应choice1中的第m个枢纽号码
            a3[np.where(a1[:] == a38)[0]] = a39
            a3_[a38,:] = 10000000
            a3_[:,a39] = 10000000
        a3 = a3.astype(np.int16)
        choice1 = choice1.astype(np.int16)
        b1 = np.zeros((self.dim2))                   # 给b1抽20个自提点备选点作枢纽 0/1（70，）
        b1[choice1[:]] = 1
        b1 = b1.astype(np.int16)
        z_lc = np.zeros((self.dim1, self.dim2))      # a5对应Z_lc——252*70
        for a, adx1 in enumerate(a3):                # 把a3矩阵中，位置为i、值为idx的元素
            z_lc[a][adx1] = 1                        # a5:0/1 (252,70)
        # 3. 计算备用枢纽矩阵
        # 1）计算需要的备选自提点个数
        a41 = np.ceil((np.dot(np.dot(self.d_ld.T, z_lc), self.q_c) / self.c)[0][0]).astype(np.int16)
        # 2）自提点枢纽按照需要的备选自提点个数聚类，找到离每簇自提点聚类中心最近的备选自提点，完成备选点分配
        a42 = np.zeros((self.dim4, 2))               # (35,2)
        a42[:] = self.h_ff[choice1[:]]
        a43 = KMeans(n_clusters=a41, n_init=10, max_iter=300, tol=0.00001).fit(a42)  # 聚类
        a44 = a43.labels_                            # 得出聚类结果
        a45 = a43.cluster_centers_                   # 得出聚类中心
        # 找出未被选中的自提点
        a46 = np.arange(0, self.dim2)
        a46 = np.array(list(set(a46).difference(set(choice1))))
        a47 = self.h_ff[a46[:]]
        a48 = np.zeros((a41, self.dim2 - self.dim4))
        a49 = np.zeros((a41)).astype(np.int16)
        for s in range(a41):                         # 选出离聚类中心最近的a41个
            for t in range(self.dim2 - self.dim4):
                a48[s, t] = np.linalg.norm(a47[t] - a45[s])
        for s in range(a41):
            a49[s] = np.where(a48[s,] == np.min(a48[s,]))[0][0]
            a48[:, a49[s]] = 100000
        a490 = np.array(a46[a49])
        a49_ = np.zeros((self.dim4))
        for s in range(a41):
            a49_[np.where(a44[:] == s)] = (a46[a49])[s]
        a49_ = a49_.astype(np.int16)
        a24 = np.zeros((self.dim1))                  # (self.dim1,),每个需求点应该被分配给哪些备选自提点
        for o in range(self.dim4):
            a24[np.where(a3[:] == choice1[o])[0]] = a49_[o]
        a24 = a24.astype(np.int16)
        # 4. 将a24转化为2维矩阵u_cs,将a27转化为2维矩阵u_bs
        u_cs = np.zeros((self.dim1, self.dim2))      # u_cs:(252,70)
        for m, mdx1 in enumerate(a24):               # 把a3矩阵中，位置为i、值为idx的元素赋值为1
            u_cs[m][mdx1] = 1
        # 5. 计算自提点枢纽中超载个数所占比例
        a5 = ( np.sum ( np.dot( self.d_ld.T, z_lc ) > self.c )) / self.dim4
        # 6. 计算超出自提点容量的总需求量占比
        a6 = (np.sum(np.where( np.dot( self.d_ld.T, z_lc ) > self.c ,np.dot( self.d_ld.T, z_lc )[:] ,0)) - np.sum ( np.dot( self.d_ld.T, z_lc ) > self.c ) * self.c) / self.d_ld_all
        return a3, choice1, b1, z_lc, a24, u_cs, a41, a5, a6, a49_, a490

    def particle_init_2(self, choice1):
        # 1.初始化时求a2
        a2 = np.zeros((self.dim2))                   # a2:1-2 (70,)
        a41 = self.h_ff.take(choice1, 0)
        kmeans4 = KMeans(n_clusters = self.dim5, n_init=10, max_iter=300, tol=0.00001).fit(a41)
        a42 = kmeans4.labels_                        # a1:0-29  (252,)
        # cluster_center4 = kmeans4.cluster_centers_ # 返回聚类中心的坐标(20,2)
        # print("cluster_center4",cluster_center4)
        for i in range(self.dim5):
            a2[choice1[np.where(a42[:] == i)[0]]] = i + 1
        a2 = a2.astype(np.int16)
        return a2

    def particle_init_3(self, a2, b1 ,e2, z_lc, a49_, choice1, a490):
        # 1. choice2、b2
        e21 = np.argsort(-e2)[:self.dim5]            # 按从小到大得出排序,选出最小的2个值对应的位置
        choice2 = e21                                # 0-3 （2，）
        choice2 = choice2.astype(np.int16)
        b2 = np.zeros((self.dim3))                   # 0/1 (4,)
        b2[choice2[:]] = 1
        b2 = b2.astype(np.int16)
        # 2.a4、z_cb
        a4 = np.zeros((self.dim2))                   # a4:1-4 (70,)
        a4[:] = np.where(a2[:]!= 0, choice2[a2[:] - 1] + 1, 0)
        a4[b1[:] == 0] = 0                           # 即未被选择的备选点，在此赋值为0
        a4 = a4.astype(np.int16)
        # print("a4",a4)
        a6 = np.zeros((self.dim2, self.dim3))        # a6:0/1 (70,4)
        for b, bdx2 in enumerate(a4):
            if bdx2 != 0:
                a6[b][a4[b] - 1] = 1
        z_cb = a6
        # 3. 计算备用枢纽矩阵
        # 1）计算需要的备选网格仓个数
        a21 = np.ceil((np.dot(np.dot(self.d_ld.T, np.dot(z_lc, z_cb)), self.q_b) / self.b)[0][0]).astype(np.int16)
        # 2）网格仓枢纽按照需要的备选网格仓个数聚类，找到离每簇网格仓聚类中心最近的备选网格仓，完成备选点分配
        a22 = np.zeros((self.dim5, 2))
        a22[:] = self.h_gg[choice2[:]]                # 取出已选网格仓枢纽的坐标
        a23 = KMeans(n_clusters=a21, n_init=10, max_iter=300, tol=0.00001).fit(a22)  # 聚类
        a24_ = a23.labels_                            # 得出聚类结果
        a25 = a23.cluster_centers_                    # 得出聚类中心
        a26 = np.arange(0, self.dim3)                 # 找出未被选中的网格仓
        a26 = np.array(list(set(a26).difference(set(choice2))))
        a27_ = self.h_gg[a26[:]]                      # 未被选中网格仓的坐标
        a28 = np.zeros((a21, self.dim3 - self.dim5))
        a29 = np.zeros((a21)).astype(np.int16)
        for p in range(a21):                          # 求备选点和各枢纽之间的距离
            for q in range(self.dim3 - self.dim5):
                a28[p,q] = np.linalg.norm(a27_[q] - a25[p])
        for p in range(a21):                          # 选出离聚类中心最近的a41个网格仓备选点
            a29[p] = np.where(a28[p,] == np.min(a28[p,]))[0][0]
            a28[:,a29[p]] = 1000000
        a29_ = np.zeros((self.dim5))                  # 转成每个网格仓枢纽的备选点(self.dim5,)
        for p in range(a21):
            a29_[np.where(a24_[:] == p)] = (a26[a29])[p] + 1
        a29_ = a29_.astype(np.int16)
        a27 = np.zeros((self.dim2))                   # (self.dim2,)
        for w in range(self.dim5):
            a27[np.where(a4[:] - 1 == choice2[w])[0]] = a29_[w]
        a27 = a27.astype(np.int16)
        # (4) 将a24转化为2维矩阵u_cs,将a30转化为2维矩阵u_bs
        u_bs = np.zeros((self.dim2, self.dim3))
        for n, ndx2 in enumerate(a27):
            if a27[n] != 0:
                u_bs[n][ndx2 - 1] = 1
        # (5) 自提点中断时(u_cs) - a3 依据(choice1 - a49_ )变形 - z_cb 变形为z_cb_
        a3_ = np.zeros((self.dim2))
        a3_[a49_[:]] = a4[choice1[:]]
        a3_ = a3_.astype(np.int16)
        z_cb_ = np.zeros((self.dim2, self.dim3))  # a6:0/1 (70,4)
        for c, cdx2 in enumerate(a3_):
            if cdx2 != 0:
                z_cb_[c][cdx2 - 1] = 1
        # (8) 自提点网格仓都中断时，依据a27得来的u_bs需变形为u_bs_
        a27_ = np.zeros((self.dim2))
        c2 = np.zeros((len(a490)))
        for x in range(len(a490)):
            c1 = a4[choice1[np.where(a49_[:] == a490[x])[0]][0]]
            c2[x] = a29_[np.where(choice2[:] == c1 - 1)[0][0]]
        a27_[a490[:]] = c2[:]
        a27_ = a27_.astype(np.int16)
        u_bs_ = np.zeros((self.dim2, self.dim3))  # u_bs:(70,4)
        for n, ndx2 in enumerate(a27_):
            if a27_[n] != 0:
                u_bs_[n][ndx2 - 1] = 1
        return b2, choice2, z_cb, u_bs, a27, a4, a21, z_cb_, u_bs_, a27_

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
        C11 = g_a +  e_ad * self.d_ld_all
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
        for i in range(0, self.dim2):                 # i:0-69
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
        for i in range(0, self.dim2):                 # i:0-69
            for j in range(0, self.dim3):             # j:0-3
                for m in range(0, self.dim1):         # m:0-251
                    C22 = a24[m]                     # 求出p_b1/1-p_b;找出a4中对应索引的a30中的网格仓备选点序号
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

    # 4. 划分迭代状态
    def pso_third(self, N_, g_best_index, t, f_state):
        # （1）计算每一代中，各粒子跟其他粒子的欧氏距离
        d1 = []
        d2 = 0
        for p_ in range(self.S):
            for q_ in range(self.S):
                d2 += np.linalg.norm(np.array(N_[p_,:]) - np.array(N_[q_,:]))
            d1.append(d2)
        d1 = np.array(d1)                            # d的size是（self.S,）
        # （2）计算进化因子f
        dg = d1[g_best_index]
        d_min = np.min(d1)
        d_max = np.max(d1)
        f_ = (dg - d_min) / (d_max - d_min)
        print("f_", f_)
        # (3) 划分状态，自适应调整w,c1,c2
        # S1、S4状态时，需要较大的w利于全局搜索；在S2、S3状态中，为改善算法的收敛性能，避免粒子在全局最优解附近“振荡”，需要较小的w。且让w随着迭代次数的增加而减少，且随着迭代所至状态的变化而变化
        w = self.w_max - (t / self.T) * (self.w_max - self.w_min) * (np.e ** (-abs(f_)))
        w = np.clip(w, self.w_min, self.w_max)
        print("w", w)
        global c1
        global c2
        if 0.2 < f_ < 0.3:
            if f_state[t + 1] == 1 or f_state[t + 1] == 2 or f_state[t + 1] == 4:
                f_state[t + 2] = 3
                c1 = 1.505  # 轻微增加c_1
                c2 = 1.505  # 轻微增加c_2
            elif f_state[t + 1] == 3:
                f_state[t + 2] = 2
                c1 = 1.505  # 轻微增加c_1
                c2 = 1.495  # 轻微减小c_2
        elif 0.4 < f_ < 0.6:
            if f_state[t + 1] == 2 or f_state[t + 1] == 3 or f_state[t + 1] == 4:
                f_state[t + 2] = 1
                c1 = 1.51  # 增加c_1
                c2 = 1.49  # 减小c_2
            elif f_state[t + 1] == 1:
                f_state[t + 2] = 2
                c1 = 1.505  # 轻微增加c_1
                c2 = 1.495  # 轻微减小c_2
        elif 0.7 < f_ < 0.8:
            if f_state[t + 1] == 1 or f_state[t + 1] == 2 or f_state[t + 1] == 3:
                f_state[t + 2] = 4
                c1 = 1.49  # 减小c_1
                c2 = 1.51  # 增加c_2
            elif f_state[t + 1] == 4:
                f_state[t + 2] = 1
                c1 = 1.51  # 增加c_1
                c2 = 1.49  # 减小c_2
        elif 0 <= f_ <= 0.2:  # S3:收敛状态
            f_state[t + 2] = 3
            c1 = 1.505  # 轻微增加c_1
            c2 = 1.495  # 轻微增加c_2
        elif 0.3 <= f_ <= 0.4:  # S2:开发状态
            f_state[t + 2] = 2
            c1 = 1.505  # 轻微增加c_1
            c2 = 1.495  # 轻微减小c_2
        elif 0.6 <= f_ <= 0.7:  # S1:探索状态
            f_state[t + 2] = 1
            c1 = 1.51  # 增加c_1
            c2 = 1.49  # 减小c_2
        elif 0.8 <= f_ <= 1:  # S4:跳出状态
            f_state[t + 2] = 4
            c1 = 1.49  # 减小c_1
            c2 = 1.51  # 增加c_2
        else:
            c1 = 1.5
            c2 = 1.5
        return w, c1, c2, f_, f_state

    # 5.主函数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def pso_main(self):
        # (1) 接下来把关键变量求出来
        N = []                                    # N: (S,144);存每一代粒子的位置x
        N_1 = []
        num_N = []                                # 存每一代每个粒子的适应度值
        A3 = []
        A5 = []
        A6 = []
        A24 = []
        A41 = []
        U_cs = []
        Z_lc = []
        B1 = []
        Choice1 = []
        A49_ = []
        A490 = []
        for o in range(self.S):
            # (1 先随机初始化,运行second函数更新目标函数相关变量，得初始化下的目标函数关键变量
            e2 = np.random.uniform(0, self.dim5, (self.dim3))  # (4, ),取值为0-2之间
            a1, cluster_center = self.particle_init_0()
            a3, choice1, b1, z_lc, a24, u_cs, a41, a5, a6, a49_, a490 = self.particle_init_1(a1, cluster_center)
            a2 = self.particle_init_2 ( choice1 )
            b2, choice2, z_cb, u_bs, a27, a4, a21, z_cb_, u_bs_, a27_ = self.particle_init_3(a2, b1 ,e2, z_lc, a49_, choice1, a490)
            # (2 拼接粒子
            x = np.concatenate((e2, a2), axis=0) # 将e2(4,)、a2(70,)、choice1(70,)、a3(252,)横向拼接;(396,)
            N.append(x)
            x_1 = np.concatenate((b1, b2, a3, a4), axis=0)
            N_1.append(x_1)
            A3.append(a3)
            A5.append(a5)
            A6.append(a6)
            A24.append(a24)
            A41.append(a41)
            U_cs.append(u_cs)
            Z_lc.append(z_lc)
            B1.append(b1)
            Choice1.append(choice1)
            A49_.append(a49_)
            A490.append(a490)
            # (3 求初代的适应度
            fitness1 = self.fun1(z_cb, z_lc, u_bs, u_cs, a24, a27, a4, a21, a41, z_cb_, u_bs_, a27_)
            num_N.append(fitness1.squeeze())
        num_N = np.array(num_N)                 # 将self.num_N、N转成array;sum_N —— 存fitness适应度值
        print("最初的num_N",num_N)
        N = np.array(N)
        N_1 = np.array(N_1)
        A3 = np.array(A3)
        A5 = np.array(A5)
        A6 = np.array(A6)
        A24 = np.array(A24)
        U_cs = np.array(U_cs)
        Z_lc = np.array(Z_lc)
        B1 = np.array(B1)
        Choice1 = np.array(Choice1)
        A49_ = np.array(A49_)
        A490 = np.array(A490,dtype = object)
        # (3)空列表存初代的局部最优值和全局最优值
        g_best_num = []                         # 建立[局部最优值列表]
        pg_best_num = []                        # 建立[全局最优值列表]
        g_best_state= []                        # 建立[局部最优粒子位置列表]
        pg_best_state = []                      # 建立[全局最优粒子位置列表]
        g_best_index = []
        g_best_num1 = np.min(num_N)             # 选出粒子群此次迭代中最小的值(局部最优值)
        g_best_num.append(g_best_num1)          # 把初代的最优值存入【局部最优值列表】
        pg_best_num.append(g_best_num1)         # 把初代的最优值存入【全局最优值列表】
        g_best_index1 = np.where(num_N == g_best_num1)[0][0]# 局部最优粒子的索引(list用。index；array用np.where)
        g_best_index.append(g_best_index1)
        g_best_state1 = N[g_best_index1]
        g_best_state.append(g_best_state1.copy())
        pg_best_state1 = N[g_best_index1]
        pg_best_state.append(pg_best_state1)
        pg = N[g_best_index1]                   # 全局最优粒子pg(74，)

        # (4)算迭代初代的自适应参数
        tt = -1
        f_state = np.zeros((self.T + 2))
        f_state[0] = 2
        w, c1, c2, f_, f_state = self.pso_third(N_1, g_best_index1, tt, f_state)
        print(f_state)
        # (5)迭代+求最优
        N2 = N                                  # 迭代时用N2
        num_N2 = num_N                          # 存迭代过程中的适应度值
        N_2 = N_1                               # 存迭代过程中的状态粒子
        # 0) 随机初始化粒子的速度
        v = np.zeros((self.S, self.dim))        # v:(S,144)
        for tt in tqdm(range(0,self.T)):        # 每一次迭代都有进度条显示
            # 1）迭代过程
            for jj in range(0, self.S):         # 每代j个粒子
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
                N2[jj][:self.dim3] = np.where(N2[jj][:self.dim3] > self.x_bound1[1], 2*self.x_bound1[1] - N2[jj][:self.dim3] , N2[jj][:self.dim3])
                N2[jj][:self.dim3] = np.where(N2[jj][:self.dim3] < self.x_bound1[0], - N2[jj][:self.dim3] , N2[jj][:self.dim3])
                N2[jj][self.dim3:self.dim3 + self.dim2] = np.clip(np.round(N2[jj][self.dim3:self.dim3 + self.dim2]), self.x_bound2[0], self.x_bound2[1])
            # 2）得迭代后的关键参数
                e2 = N2[jj][ :self.dim3]
                a2 = N2[jj][self.dim3: self.dim3 + self.dim2]
                a2 = a2.astype(np.int16)
                a24 = A24[jj]
                a41 = A41[jj]
                u_cs = U_cs[jj]
                z_lc = Z_lc[jj]
                b1 = B1[jj]
                a3 = A3[jj]
                choice1 = Choice1[jj]
                a49_ = A49_[jj]
                a490 = A490[jj]
                b2, choice2, z_cb, u_bs, a27, a4, a21, z_cb_, u_bs_, a27_ = self.particle_init_3(a2, b1 ,e2, z_lc, a49_, choice1, a490)
                x_2 = np.concatenate((b1, b2, a3, a4), axis=0)
                N_2[jj] = x_2
            # 3）得迭代后的适应度
                fitness2 = self.fun1(z_cb, z_lc, u_bs, u_cs, a24, a27, a4, a21, a41, z_cb_, u_bs_, a27_)
                fitness2 = fitness2.squeeze().item()
                num_N2[jj] = fitness2
            print("num_N2", num_N2)
            # 4) 求出局部最优粒子及其适应度值
            g_best_num2 = np.min(np.array(num_N2))     # 选出粒子群此次迭代中最小的值(局部最优值)
            g_best_index2 = np.where(np.array(num_N2) == g_best_num2)[0][0] # 局部最优粒子的索引
            pg_best_num2 = np.min(g_best_num)          # 全局最优粒子适应度
            pg = g_best_state[np.where(np.array(g_best_num) == pg_best_num2)[0][0]]
            print("g_best_index2",g_best_index2)
            print("pg_best_num2", pg_best_num2)

            # (6) 对第一段[0,4]和第二段[4,74]进行遗传算法的交叉操作
            N3 = N2
            N_3 = N_2
            for ii in range(self.S):
                if np.random.rand() <= 0.9:
                    # 1) 选取parent1和parent2
                    parent1 = N3[ii]
                    rand = random.uniform(0, w + c1 + c2)# 分别按1/3的概率选取粒子本身逆序、当前最优解、全局最优解作为parent2
                    if 0 <= rand <= w:
                        parent2 = N3[-ii - 1]            # 1)parent2选取粒子本身逆序
                    elif w < rand <= w + c1:
                        parent2 = N3[g_best_index2]      # 2)parent2选取全局最优解
                    else:
                        parent2 = pg                     # 3)parent2选取当前最优解
                    # 2) 交叉——将[parent1第一段]+[=parent2第二段]进行交叉
                    son = np.zeros_like(parent1)         # (2,326)
                    son[0 : self.dim3] = parent1[0 : self.dim3]
                    son[self.dim3 : ] = parent2[self.dim3 : ]
                    N3[ii] = son
            for nn in range(self.S):
                # 1.得迭代后的关键参数
                e2 = N3[nn][:self.dim3]
                a2 = N3[nn][self.dim3: self.dim3 + self.dim2]
                a2 = a2.astype(np.int16)
                a24 = A24[nn]
                a41 = A41[nn]
                u_cs = U_cs[nn]
                z_lc = Z_lc[nn]
                b1 = B1[nn]
                a3 = A3[nn]
                choice1 = Choice1[nn]
                a49_ = A49_[nn]
                a490 = A490[nn]
                b2, choice2, z_cb, u_bs, a27, a4, a21, z_cb_, u_bs_, a27_ = self.particle_init_3(a2, b1 ,e2, z_lc, a49_, choice1, a490)
                x_3 = np.concatenate((b1, b2, a3, a4), axis=0)
                N_3[nn] = x_3
                # 3）得迭代后的适应度
                fitness3 = self.fun1(z_cb, z_lc, u_bs, u_cs, a24, a27, a4, a21, a41, z_cb_, u_bs_, a27_)
                fitness3 = fitness3.squeeze().item()
                if fitness3 < num_N2[nn]:
                    num_N2[nn] = fitness3
                    N2[nn] = N3[nn]
                    N_2[nn] = N_3[nn]
            print("num_N2_", num_N2)

            # (8) 个体最优值列表替换——将个体最优值存在self.num_N里
            for pp in range(self.S):
                if num_N2[pp] < num_N[pp]:
                    num_N[pp] = num_N2[pp]
                    N[pp] = N2[pp]
            # (9) 求出局部最优粒子及其适应度值
            g_best_num3 = np.min(np.array(num_N2))     # 选出粒子群此次迭代中最小的值(局部最优值)
            g_best_num.append(g_best_num3.copy())             # 各局部最优值存入列表
            g_best_index3 = np.where(np.array(num_N2) == g_best_num3)[0][0]# 局部最优粒子的索引
            g_best_index.append(g_best_index3.copy())
            # (10) 求全局最优值——拿这一代的最优值跟全局最优做比较,若这一代最优值更优，则成为全局最优值
            g_best_state3 = N2[g_best_index3]          # 取出局部最优的粒子位置
            g_best_state.append(g_best_state3.copy())  # 局部最优粒子位置存入列表
            pg_best_num3 = np.min(g_best_num)          # 全局最优粒子适应度
            pg_best_num.append(pg_best_num3.copy())           # 全局最优粒子适应度
            pg = g_best_state[np.where(np.array(g_best_num) == pg_best_num3)[0][0]]
            pg_best_state.append(pg)                   # 全局最优粒子位置列表
            print("g_best_index3", g_best_index3)
            print("pg_best_num3", pg_best_num3)
            print("g_best_num", g_best_num)
            print("pg_best_num", pg_best_num)
            print("g_best_index", g_best_index)

            # (11) 更新自适应参数
            w, c1, c2, f_, f_state = self.pso_third(N_2, g_best_index2, tt, f_state)
            print("w, c1, c2,f_state", w, c1, c2, f_state)
            print('第' + str(tt) + '次迭代：局部最优解位置在' + str(g_best_index3) + '，全局最优解的适应度值为：' + str(pg_best_num3))
            # 算全局最优粒子pg所在的位置
            if pg_best_num[tt+1] == pg_best_num[tt]:
                g_best_index4 = g_best_index[np.where(np.array(g_best_num) == pg_best_num3)[0][0]]
            else:
                g_best_index4 = g_best_index3
            # (12)变异操作_对最优粒子
            # 1) 求这一代里的全体粒子适应度方差——计算最优粒子的变异概率
            f_min = pg_best_num2
            f_vary = 0
            f_avg = np.average(np.array(num_N2))
            for dd in range(self.S):
                f_i = num_N2[dd]
                f_vary += ((f_i - f_min) / f_avg) ** 2
            f_vary = self.P_min + (self.P_max - self.P_min) * (1 - (f_vary / self.S)) * f_
            print("f_vary", f_vary)
            pg_ = pg
            if np.random.rand() <= f_vary:
                d1 = random.sample(range(0, self.dim3), 2)
                for x in d1:
                    pg_[x] = np.random.uniform(0, self.dim5, (1))
            # 1.得迭代后的关键参数
            e2 = pg_[:self.dim3]
            a2 = pg_[self.dim3: self.dim3 + self.dim2]
            a2 = a2.astype(np.int16)
            a24 = A24[g_best_index4]
            a41 = A41[g_best_index4]
            u_cs = U_cs[g_best_index4]
            z_lc = Z_lc[g_best_index4]
            b1 = B1[g_best_index4]
            choice1 = Choice1[g_best_index4]
            a49_ = A49_[g_best_index4]
            a490 = A490[g_best_index4]
            b2, choice2, z_cb, u_bs, a27, a4, a21, z_cb_, u_bs_, a27_ = self.particle_init_3(a2, b1 ,e2, z_lc, a49_, choice1, a490)
            # 2）得迭代后的适应度
            fitness4 = self.fun1(z_cb, z_lc, u_bs, u_cs, a24, a27, a4, a21, a41, z_cb_, u_bs_, a27_)
            fitness4 = fitness4.squeeze().item()
            if fitness4 < pg_best_num2:
                pg_best_num[tt] = fitness4
                pg = pg_
            # print("变异后pg",pg)
            print("fitness4", fitness4)
            print("...............")

            # (13) 画分配图
            if tt == self.T - 1:
                e2 = pg[: self.dim3]
                a2 = pg[self.dim3: self.dim3 + self.dim2]
                a2 = a2.astype(np.int16)
                a24 = A24[g_best_index3]
                b1 = B1[g_best_index3]
                choice1 = Choice1[g_best_index3]
                a3 = A3[g_best_index3]
                a5 = A5[g_best_index3]
                a6 = A6[g_best_index3]
                b2, choice2, z_cb, u_bs, a27, a4, a21, z_cb_, u_bs_, a27_ = self.particle_init_3(a2, b1 ,e2, z_lc, a49_, choice1, a490)
                a3 = pd.DataFrame(a3)
                a4 = pd.DataFrame(a4)
                choice1 = pd.DataFrame(choice1)
                choice2 = pd.DataFrame(choice2)
                a24 = pd.DataFrame(a24)
                a27 = pd.DataFrame(a27)
                # (13) 数据存表
                dt = [r for r in range(0,self.T+1)]
                dt = pd.DataFrame(dt)
                best = pd.DataFrame(pg_best_num)
                df = pd.concat([dt,best,a3,a4,choice1,choice2,a24,a27],axis = 1)
                print(df)
                print("a5&a6",a5,a6)
                writer = pd.ExcelWriter(output_file)
                df.to_excel(writer, sheet_name="algorithm", encoding="utf-8")
                writer.close()
        # (14) 画图
        plt.plot([t for t in range(self.T+1)], pg_best_num)                   # 画每一代的最优值
        plt.ylabel('fitness')                                             # y轴是每代的适应度值
        plt.xlabel('iter nums')                                           # x轴是迭代次数
        plt.title('BPSO')                                                  # 标题是粒子群适应度趋势
        plt.show()                                                        # 画图

pso = PSO()
pso.pso_main()