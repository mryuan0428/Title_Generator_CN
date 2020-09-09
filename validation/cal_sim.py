#-*- encoding:utf-8 -*-
'''
sim_bilstm: 93.5311353301591%
sim_textrank: 93.66629714960351%
sim_trw2v: 93.92663687443734%
sim_unilm: 95.65124754779288%
'''
import os, sys
import numpy as np
import threading

title_vec_path = sys.path[0] + "/title_vec.npy"
title_bilstm_vec_path = sys.path[0] + "/title_bilstm_vec.npy"
title_textrank_vec_path = sys.path[0] + "/title_textrank_vec.npy"
title_trw2v_vec_path = sys.path[0] + "/title_trw2v_vec.npy"
title_unilm_vec_path = sys.path[0] + "/title_unilm_vec.npy"

class CalSimThread(threading.Thread):
    def __init__(self, func, *args):
        threading.Thread.__init__(self)  #对父类属性初始化
        self.func = func
        self.args = args

    # 重写run方法进行计算相似度
    def run(self):
        self.result = self.func(*self.args)

    # 构造get_result方法传出返回值
    def get_result(self):
        try:
            return self.result  #将结果return
        except Exception:
            return None

def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim * 100

def get_mean_sim(title_vec_path, title_gen_vec_path):
    title_vec = np.load(title_vec_path)
    title_vec = title_vec.tolist()

    title_gen_vec = np.load(title_gen_vec_path)
    title_gen_vec = title_gen_vec.tolist()

    sim = 0.0

    l = len(title_vec)
    for i in range(l):
        s_i = cos_sim(title_vec[i],title_gen_vec[i])
        sim += s_i

        if i % 100 == 0:
            print(f"{i} sims with {title_gen_vec_path.split('/')[-1].split('.')[0]} finished.")
    sim /= l
    return sim

if __name__ == "__main__":
    s_bilstm = CalSimThread(get_mean_sim, title_vec_path, title_bilstm_vec_path)
    s_textrank = CalSimThread(get_mean_sim, title_vec_path, title_textrank_vec_path)
    s_trw2v = CalSimThread(get_mean_sim, title_vec_path, title_trw2v_vec_path)
    s_unilm = CalSimThread(get_mean_sim, title_vec_path, title_unilm_vec_path)

    s_bilstm.start()
    s_textrank.start()
    s_trw2v.start()
    s_unilm.start()

    s_bilstm.join()
    s_textrank.join()
    s_trw2v.join()
    s_unilm.join()

    sim_bilstm = s_bilstm.get_result()
    sim_textrank = s_textrank.get_result()
    sim_trw2v = s_trw2v.get_result()
    sim_unilm = s_unilm.get_result()

    print(f"sim_bilstm: {sim_bilstm}%")
    print(f"sim_textrank: {sim_textrank}%")
    print(f"sim_trw2v: {sim_trw2v}%")
    print(f"sim_unilm: {sim_unilm}%")
