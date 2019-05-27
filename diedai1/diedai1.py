

import json
from tqdm import tqdm
import os
import numpy as np
from random import choice
from itertools import groupby


mode = 0
min_count = 1
char_size = 128
negative=5


id2kb = {}
with open(r'kb_data',encoding='utf-8') as f: #知识库中的所有属性串联起来 alias 和object各自串联起来
    for l in tqdm(f):
        _ = json.loads(l)
        subject_id = _['subject_id']
        # if subject_id=='310293':
        #     print(_)
        subject_alias = list(set([_['subject']] + _.get('alias', [])))
        # if 'lol'in subject_alias:
        #     print(_)
        # if '英雄联盟' in subject_alias:
        #     print(_)
        subject_alias = [alias.lower() for alias in subject_alias]
        subject_desc = '\n'.join(u'%s：%s' % (i['predicate'], i['object']) for i in _['data'])
        subject_desc = subject_desc.lower()
        if subject_desc:

            id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}


kb2id = {}
for i,j in zip(id2kb.keys(),id2kb.values()): #i为序号，j为每一个存储在id2kb中的字典
    for k in j['subject_alias']:# 所有alias都加入kb2id，并且字典对应的值为在知识库中的序号
        if k not in kb2id:
            kb2id[k] = []
for i1, j1 in zip(id2kb.keys(), id2kb.values()):  # i为序号，j为每一个存储在id2kb中的字典
    for k1 in j1['subject_alias']:  # 所有alias都加入kb2id，并且字典对应的值为在知识库中的序号
        kb2id[k1].append(i1)
    # for k in j['subject_alias']:
    #

# print(id2kb['391539'])
# print(kb2id['lol'])
# print(id2kb['161540']['subject_alias'])
# for i in (id2kb['161540']['subject_alias']):
#     print(kb2id[i])

train_data = []
with open(r'train.json',encoding='utf-8') as f:
    for l in tqdm(f):
        _ = json.loads(l)
        train_data.append({
            'text': _['text'].lower(),
            'mention_data': [(x['mention'].lower(), int(x['offset']), x['kb_id'])
                for x in _['mention_data'] if x['kb_id'] != 'NIL'
            ]
        })


if not os.path.exists(r'all_chars_me.json'):
    chars = {} #字典 存储实体的属性在材料中出现的次数
    for d in tqdm(iter(id2kb.values())):
        for c in d['subject_desc']:
            chars[c] = chars.get(c, 0) + 1
    for d in tqdm(iter(train_data)):
        for c in d['text']:
            chars[c] = chars.get(c, 0) + 1
    chars = {i:j for i,j in chars.items() if j >= min_count} #出现过的全保留下来 因为小于二就说明在训练集和知识库至少有一次没出现过 这种数据没意义
    id2char = {i+2:j for i,j in enumerate(chars)} # 0: mask, 1: padding
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], open(r'all_chars_me.json', 'w',encoding='utf-8'))
else:
    id2char, char2id = json.load(open(r'all_chars_me.json'))


if not os.path.exists(r'random_order_train.json'): #乱序的训练材料
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open(r'random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open(r'random_order_train.json'))


dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode]#打乱后重新组织成traindata,在train_json里面划分出验证集
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator: #针对训练数据的处理
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, S1, S2, Y, T = [], [], [], [], [], [],
            count=0
            for i in idxs:
                # print(count)
                # print(i)
                # count += 1
                d = self.data[i]
                text = d['text'] #数据中的文本部分
                x1 = [char2id.get(c, 1) for c in text] # 从字符表中寻找到text中每个字
                s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                mds = {} #存储mentiondata的序列
                for md in d['mention_data']: #如果mentiondata中出现在了对应知识库中
                    if md[0] in kb2id: #md[0]=mentiondata md[1]=offset md[2]=kbid。
                        j1 = md[1]
                        j2 = j1 + len(md[0])
                        s1[j1] = 1       #实体位置标1 说明这是边界 分别规定了实体在text位置上的左边界与右边界
                        s2[j2 - 1] = 1
                        mds[(j1, j2)] = (md[0], md[2]) # 存储每个mention及其在kb中的id
                if mds:
                    for j1,j2 in mds.keys():
                        y = np.zeros(len(text))
                        y[j1: j2] = 1 #mention中的实体在 text长度的列表中标记
                        # print(mds[(j1, j2)][0])
                        # print(mds[(j1, j2)][1])
                        x2 = kb2id[mds[(j1, j2)][0]] #mention中的实体在kb中的id（一个实体对应多个id，取随机一个）
                        if mds[(j1, j2)][1] not in x2:
                            continue
                        # print(d)
                        h = x2.index(mds[(j1, j2)][1])
                        if (h > negative-1):
                            x2[0] = mds[(j1, j2)][1]
                        if (len(x2) < negative):
                            for i in range(negative - len(x2)):
                                lift = choice(choice(list(kb2id.values())))
                                while (lift == mds[(j1, j2)][1]):
                                    lift = choice(choice(list(kb2id.values())))
                                x2.append(str(lift))
                        else:

                            x2 = x2[0:negative]
                        for i in range(negative):
                            if x2[i] == mds[(j1, j2)][1]: #mention中的实体是否与kb同名实体的随机抽取的id相一致
                                t=[1]
                            else:
                                t=[0]
                            x2change = id2kb[x2[i]]['subject_desc']  # 与x2为id的实体相联系的属性
                            x2change = [char2id.get(c, 1) for c in x2change]  # 转化为字符编码
                            X1.append(x1)
                            X2.append(x2change)
                            S1.append(s1)
                            S2.append(s2)
                            Y.append(y)
                            T.append(t)
                            if len(X1)==self.batch_size or i == idxs[-1]:
                                X1 = seq_padding(X1)  # 填充达到一致长度
                                X2 = seq_padding(X2)
                                S1 = seq_padding(S1)
                                S2 = seq_padding(S2)
                                Y = seq_padding(Y)
                                T = seq_padding(T)
                                yield [X1, X2, S1, S2, Y, T], None
                                X1, X2, S1, S2, Y, T = [], [], [], [], [], []


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import initializers
from keras.layers.normalization import BatchNormalization
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                2 * K.arange(self.size / 2, dtype='float32' \
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, mask_right=False, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.mask_right = mask_right
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        if self.mask_right:
            ones = K.ones_like(A[:1, :1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e12
            A = A - mask
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


# class Interact(Layer):
#     """交互层，负责融合encoder和decoder的信息
#     """
#     def __init__(self, **kwargs):
#         super(Interact, self).__init__(**kwargs)
#     def build(self, input_shape):
#         in_dim = input_shape[0][-1]
#         out_dim = input_shape[1][-1]
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(in_dim, out_dim),
#                                       initializer='glorot_normal')
#     def call(self, inputs):
#         q, v, v_mask = inputs
#         k = v
#         mv = K.max(v - (1. - v_mask) * 1e10, axis=1, keepdims=True) # maxpooling1d
#         mv = mv + K.zeros_like(q[:,:,:1]) # 将mv重复至“q的timesteps”份
#         # 下面几步只是实现了一个乘性attention
#         qw = K.dot(q, self.kernel)
#         a = K.batch_dot(qw, k, [2, 2]) / 10.
#         a -= (1. - K.permute_dimensions(v_mask, [0, 2, 1])) * 1e10
#         a = K.softmax(a)
#         o = K.batch_dot(a, v, [2, 1])
#         # 将各步结果拼接
#         return K.concatenate([o, q, mv], 2)
#     def compute_output_shape(self, input_shape):
#         return (None, input_shape[0][1],
#                 input_shape[0][2]+input_shape[1][2]*2)

def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


x1_in = Input(shape=(None,)) # 待识别句子输入
x2_in = Input(shape=(None,)) # 实体语义表达输入 kb中相关属性的连接
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）
y_in = Input(shape=(None,)) # 实体标记 text中 mention的位置标记为1
t_in = Input(shape=(1,)) # 是否有关联（标签）


x1, x2, s1, s2, y, t = x1_in, x2_in, s1_in, s2_in, y_in, t_in
x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
x2_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x2)

embedding = Embedding(len(id2char)+2, char_size)
position = Position_Embedding()
Batch=BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')

x1 = embedding(x1) #char embedding
x1 = position(x1)
x1 =Attention(8,16)([x1,x1,x1])#self attention
x1 = Dense(8*16,activation='relu')(x1)
x1 = Dense(8*16,activation='linear')(x1)
x1 = Batch(x1)
x1 =Attention(8,16)([x1,x1,x1])#self attention
x1 = Dense(8*16,activation='relu')(x1)
x1 = Dense(8*16,activation='linear')(x1)
x1 = Batch(x1)
x1 =Attention(8,16)([x1,x1,x1])#self attention
x1 = Dense(8*16,activation='relu')(x1)
x1 = Dense(8*16,activation='linear')(x1)
x1 = Batch(x1)
x1 =Attention(8,16)([x1,x1,x1])#self attention
x1 = Dense(8*16,activation='relu')(x1)
x1 = Dense(8*16,activation='linear')(x1)
x1 = Batch(x1)
x1 =Attention(8,16)([x1,x1,x1])#self attention
x1 = Dense(8*16,activation='relu')(x1)
x1 = Dense(8*16,activation='linear')(x1)
x1 = Batch(x1)
x1 =Attention(8,16)([x1,x1,x1])#self attention
x1 = Dense(8*16,activation='relu')(x1)
x1 = Dense(8*16,activation='linear')(x1)
x1 = Batch(x1)
x1 = Dropout(0.2)(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
# x1 = AttLayer()(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
h = Conv1D(char_size, 3, activation='relu', padding='same')(x1)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)

s_model = Model(x1_in, [ps1, ps2]) #识别实体，输入句子，输出实体识别的左右边界（如是s1为句子字符长度，实体左右边界标1）


y = Lambda(lambda x: K.expand_dims(x, 2))(y)
x1 = Concatenate()([x1, y])
x1 = Conv1D(char_size, 3, padding='same')(x1)
x2 = embedding(x2)
x2 = position(x2)
x2 = Dropout(0.2)(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x2 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x2 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x1 = Lambda(seq_maxpool)([x1, x1_mask])
x2 = Lambda(seq_maxpool)([x2, x2_mask])
# x2 =Attention(8,16)([x2,x2,x2])
x12 = Multiply()([x1, x2])
# x23 = Interact()([x12,x3,x2_mask])
x = Concatenate()([x1,x12])
x = Dense(char_size, activation='relu')(x)
pt = Dense(1, activation='sigmoid')(x)
t_model = Model([x1_in, x2_in, y_in], pt)  #输出识别实体的对应在知识库中的编号
                                         #模型主要是训练实体属性到这个实体的映射，判断是否有关联，同名实体是否符合


train_model = Model([x1_in, x2_in, s1_in, s2_in, y_in, t_in],
                    [ps1, ps2, pt])

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * x1_mask) / K.sum(x1_mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * x1_mask) / K.sum(x1_mask)
pt_loss = K.mean(K.binary_crossentropy(t, pt))

loss = s1_loss + s2_loss + pt_loss

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()


def extract_items(text_in):  #验证函数
    _x1 = [char2id.get(c, 1) for c in text_in]
    _x1 = np.array([_x1])
    _k1, _k2 = s_model.predict(_x1)
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.5)[0] #大于0.5的识别成真实的位置
    _subjects = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i: j+1]
            _subjects.append((_subject, i, j)) #列表加入实体和左右边界
    if _subjects:
        R = []
        _X2, _Y = [], []
        _S, _IDXS = [], {}
        for _s in _subjects:
            _y = np.zeros(len(text_in))
            _y[_s[1]: _s[2]] = 1
            _IDXS[_s] = kb2id.get(_s[0], [])  #找出知识库中与预测实体同名的实体集合
            for i in _IDXS[_s]:
                _x2 = id2kb[i]['subject_desc']
                _x2 = [char2id.get(c, 1) for c in _x2]
                _X2.append(_x2)
                _Y.append(_y)
                _S.append(_s)
        if _X2:
            _X2 = seq_padding(_X2)
            _Y = seq_padding(_Y)
            _X1 = np.repeat(_x1, len(_X2), 0)
            scores = t_model.predict([_X1, _X2, _Y])[:, 0]
            for k, v in groupby(zip(_S, scores), key=lambda s: s[0]): #每一个预测出来的实体以及分数
                v = np.array([j[1] for j in v])
                kbid = _IDXS[k][np.argmax(v)]  #选择分数最高的
                R.append((k[0], k[1], kbid))  #输出相关的位置和分数最高的实体的编码
        return R
    else:
        return []


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.65
    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            train_model.save_weights(r'diedai1\best_model1.weights')
            s_model.save_weights(r'diedai1\ner_model1.weights')
            t_model.save_weights(r'diedai1\el_model1.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        for d in tqdm(iter(dev_data)):
            R = set(extract_items(d['text']))
            T = set(d['mention_data'])
            A += len(R & T)
            B += len(R)
            C += len(T)
        return 2 * A / (B + C), A / B, A / C


evaluator = Evaluate()
train_D = data_generator(train_data)
if os.path.exists(r'diedai1\best_model1.weights'):
    train_model.load_weights(r'diedai1\best_model1.weights')

train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=3125,
                          epochs=2000,
                          callbacks=[evaluator],
                         )
