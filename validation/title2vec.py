import os, sys
import numpy as np
from bert_serving.client import BertClient

title_path = sys.path[0] + "/title.txt"
title_bilstm_path = sys.path[0] + "/title_bilstm.txt"
title_textrank_path = sys.path[0] + "/title_textrank.txt"
title_trw2v_path = sys.path[0] + "/title_trw2v.txt"
title_unilm_path = sys.path[0] + "/title_unilm.txt"

title_vec_path = sys.path[0] + "/title_vec.npy"
title_bilstm_vec_path = sys.path[0] + "/title_bilstm_vec.npy"
title_textrank_vec_path = sys.path[0] + "/title_textrank_vec.npy"
title_trw2v_vec_path = sys.path[0] + "/title_trw2v_vec.npy"
title_unilm_vec_path = sys.path[0] + "/title_unilm_vec.npy"

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

#-----------Title_bilstm2Vec-----------
cnt = 0
title_vec = []
for line in open(title_bilstm_path,"r",encoding="utf-8"):
    a = bc.encode([line.rstrip('\n')])
    title_vec.append(a)
    cnt += 1
    if cnt % 100 == 0:
        print(f'{cnt} titles have been encoded.')
    
title_vec = np.array(title_vec)
np.save(title_bilstm_vec_path, title_vec)

#-----------Title_textrank2Vec-----------
cnt = 0
title_vec = []
for line in open(title_textrank_path,"r",encoding="utf-8"):
    a = bc.encode([line.rstrip('\n')])
    title_vec.append(a)
    cnt += 1
    if cnt % 100 == 0:
        print(f'{cnt} titles have been encoded.')
    
title_vec = np.array(title_vec)
np.save(title_textrank_vec_path, title_vec)

#-----------Title_trw2v2Vec-----------
cnt = 0
title_vec = []
for line in open(title_trw2v_path,"r",encoding="utf-8"):
    a = bc.encode([line.rstrip('\n')])
    title_vec.append(a)
    cnt += 1
    if cnt % 100 == 0:
        print(f'{cnt} titles have been encoded.')
    
title_vec = np.array(title_vec)
np.save(title_trw2v_vec_path, title_vec)

#-----------Title_UniLM2Vec-----------
cnt = 0
title_vec = []
for line in open(title_unilm_path,"r",encoding="utf-8"):
    a = bc.encode([line.rstrip('\n')])
    title_vec.append(a)
    cnt += 1
    if cnt % 100 == 0:
        print(f'{cnt} titles have been encoded.')
    
title_vec = np.array(title_vec)
np.save(title_unilm_vec_path, title_vec)
