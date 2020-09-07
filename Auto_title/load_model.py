#! -*- coding: utf-8 -*-
import sys
import copy
import numpy as np
from tqdm import tqdm
import os,json
from keras.layers import *
from keras_layer_normalization import LayerNormalization
from keras.models import Model,load_model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam


min_count = 32
maxlen = 400
#batch_size = 128
#epochs = 20
char_size = 128
z_dim = 128
train_data_path=sys.path[0]+"/news2016zh_train.json"
    

if os.path.exists(sys.path[0]+'/seq2seq_config.json'):
    chars,id2char,char2id = json.load(open(sys.path[0]+'/seq2seq_config.json'))
    id2char = {int(i):j for i,j in id2char.items()}
else:
    chars = {}
    for line in open(train_data_path,"r",encoding="utf-8"):
        a = json.loads(line)
        for w in a['content']: # 纯文本，不用分词
            chars[w] = chars.get(w,0) + 1
        for w in a['title']: # 纯文本，不用分词
            chars[w] = chars.get(w,0) + 1
    chars = {i:j for i,j in chars.items() if j >= min_count}
    # 0: mask
    # 1: unk
    # 2: start
    # 3: end
    id2char = {i+4:j for i,j in enumerate(chars)}
    char2id = {j:i for i,j in id2char.items()}
    json.dump([chars,id2char,char2id], open(sys.path[0]+'/seq2seq_config.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end: # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen-2]]
        ids = [2] + ids + [3]
    else: # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [0] * (ml-len(i)) for i in x]


#def data_generator():
#    # 数据生成器
#    X,Y = [],[]
#    while True:
#        for line in open(train_data_path,"r",encoding="utf-8"):
#            a = json.loads(line)
#            X.append(str2id(a['content']))
#            Y.append(str2id(a['title'], start_end=True))
#            if len(X) == batch_size:
#                X = np.array(padding(X))
#                Y = np.array(padding(Y))
#                yield [X,Y], None
#                X,Y = [],[]


def to_one_hot(x):
    """输出一个词表大小的向量，来标记该词是否在文章出现过
    """
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars)+4)
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x


class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def build(self, input_shape):
        kernel_shape = (1,) * (len(input_shape)-1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')
    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        return outputs


class OurBidirectional(OurLayer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = copy.deepcopy(layer)
        self.backward_layer = copy.deepcopy(layer)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return K.tf.reverse_sequence(x, seq_len, seq_dim=1)
    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], 2)
        return x * mask
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1], self.forward_layer.units * 2)


def seq_avgpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做avgpooling。
    """
    seq, mask = x
    return K.sum(seq * mask, 1) / (K.sum(mask, 1) + 1e-6)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


class SelfModulatedLayerNormalization(OurLayer):
    """模仿Self-Modulated Batch Normalization，
    只不过降Batch Normalization改为Layer Normalization
    """
    def __init__(self, num_hidden, **kwargs):
        super(SelfModulatedLayerNormalization, self).__init__(**kwargs)
        self.num_hidden = num_hidden
    def build(self, input_shape):
        super(SelfModulatedLayerNormalization, self).build(input_shape)
        output_dim = input_shape[0][-1]
        self.layernorm = LayerNormalization(center=False, scale=False)
        self.beta_dense_1 = Dense(self.num_hidden, activation='relu')
        self.beta_dense_2 = Dense(output_dim)
        self.gamma_dense_1 = Dense(self.num_hidden, activation='relu')
        self.gamma_dense_2 = Dense(output_dim)
    def call(self, inputs):
        inputs, cond = inputs
        inputs = self.reuse(self.layernorm, inputs)
        beta = self.reuse(self.beta_dense_1, cond)
        beta = self.reuse(self.beta_dense_2, beta)
        gamma = self.reuse(self.gamma_dense_1, cond)
        gamma = self.reuse(self.gamma_dense_2, gamma)
        for _ in range(K.ndim(inputs) - K.ndim(cond)):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return inputs * (gamma + 1) + beta
    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Attention(OurLayer):
    """多头注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(a[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


# 搭建seq2seq模型

x_in = Input(shape=(None,))
y_in = Input(shape=(None,))
x, y = x_in, y_in

x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)

x_one_hot = Lambda(to_one_hot)([x, x_mask])
x_prior = ScaleShift()(x_one_hot) # 学习输出的先验分布（标题的字词很可能在文章出现过）

embedding = Embedding(len(chars)+4, char_size)
x = embedding(x)
y = embedding(y)

# encoder，双层双向LSTM
x = LayerNormalization()(x)
x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
x = LayerNormalization()(x)
x = OurBidirectional(CuDNNLSTM(z_dim // 2, return_sequences=True))([x, x_mask])
x_max = Lambda(seq_maxpool)([x, x_mask])

# decoder，双层单向LSTM
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])
y = CuDNNLSTM(z_dim, return_sequences=True)(y)
y = SelfModulatedLayerNormalization(z_dim // 4)([y, x_max])

# attention交互
xy = Attention(8, 16)([y, x, x, x_mask])
xy = Concatenate()([y, xy])

# 输出分类
xy = Dense(char_size)(xy)
xy = LeakyReLU(0.2)(xy)
xy = Dense(len(chars)+4)(xy)
xy = Lambda(lambda x: (x[0]+x[1])/2)([xy, x_prior]) # 与先验结果平均
xy = Activation('softmax')(xy)

# 交叉熵作为loss，但mask掉padding部分
cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
cross_entropy = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

model = Model([x_in,y_in],xy)
model.load_weights(sys.path[0]+'/best_model.weights')


def clean_punct(s):
    """清除候选序列的最后不规则标点符号"""
    if len(s) == 0:
        return "最新新闻报道"
    if s[-1] in ["，", ",", "。", ".", "<", ">", "《", "》", "！", "!", '“', "'", '"', "？", "?", "：", ":", "[", "【", "】", "]","{", "}", " "]:
        s = s[:-1]
        return clean_punct(s)
    return s


def gen_sent(s, topk=7, maxlen=64):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s)] * topk) # 输入转id
    yid = np.array([[2]] * topk) # 解码均以<start>开头，这里<start>的id为2
    scores = [0] * topk # 候选答案分数
    for i in range(maxlen): # 强制要求输出不超过maxlen字
        proba = model.predict([xid, yid])[:, i, 3:] # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(proba + 1e-6) # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:,-topk:] # 每一项选出topk
        _yid = [] # 暂存的候选目标序列
        _scores = [] # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]+3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(topk):
                for k in range(topk): # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k]+3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:] # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = np.array(_yid)
        scores = np.array(_scores)
        ends = np.where(yid[:, -1] == 3)[0]
        if len(ends) > 0:
           # for t in range(topk):
           #    if id2str(yid[t+1]) != id2str(yid[t])
            k = ends[scores[ends].argmax()]
            return clean_punct(id2str(yid[k]))
    # 如果maxlen字都找不到<end>，直接返回
    return clean_punct(id2str(yid[np.argmax(scores)]))

'''
cnt = 0
title = []
for line in open(train_data_path,"r",encoding="utf-8"):
    a = json.loads(line)
    Y = a['content']
    if len(Y)== 0:
        Y = "最新新闻报道"
    title_gen = gen_sent(Y)
    title.append(title_gen)

    cnt += 1

    if cnt % 500 == 0:
        print(f"{cnt} / 76500")
        f = open(sys.path[0] + "/title_gen.txt",'a+',encoding='utf-8')
        for item in title:
            f.write(item+'\n')
        f.close()
        title = []
'''



while True:
    s = input("请输入新闻：\n")
    if s == 'exit':
       print('已退出')
       break
    else:
       a = gen_sent(s)
       print("标题:《%s》"% a)
       #print(gen_sent(s))
       continue
