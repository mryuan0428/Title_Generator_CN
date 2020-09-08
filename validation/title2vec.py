import os, sys
import numpy as np
from bert_serving.client import BertClient

title_path = sys.path[0] + "/title.txt"

title_vec_path = sys.path[0] + "/title_vec.npy"

bc = BertClient()

#-----------Title2Vec-----------
cnt = 0
title_vec = []
for line in open(title_path,"r",encoding="utf-8"):
    a = bc.encode([line.rstrip('\n')])
    title_vec.append(a)
    cnt += 1
    if cnt % 100 == 0:
        print(f'{cnt} titles have been encoded.')
    
title_vec = np.array(title_vec)
np.save(title_vec_path, title_vec)
