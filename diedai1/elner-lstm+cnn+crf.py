
import json
from tqdm import tqdm
import os
import numpy as np
from random import choice
from itertools import groupby
from kashgari.embeddings import BERTEmbedding
from keras.utils.np_utils import *
mode = 0
min_count = 1
char_size = 128
negative =5
embedding1 = BERTEmbedding('bert-base-chinese', sequence_length=30)
id2tag = {0:'s', 1:'b', 2:'m'} # 标签（sbme）与id之间的映射
tag2id = {j:i for i,j in id2tag.items()}
id2kb = {}
with open(r'kb_data' ,encoding='utf-8') as f:  # 知识库中的所有属性串联起来 alias 和object各自串联起来
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
for i ,j in zip(id2kb.keys() ,id2kb.values()):  # i为序号，j为每一个存储在id2kb中的字典
    for k in j['subject_alias']  :# 所有alias都加入kb2id，并且字典对应的值为在知识库中的序号
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
with open(r'train.json' ,encoding='utf-8') as f:
    for l in tqdm(f):
        _ = json.loads(l)
        train_data.append({
            'text': _['text'].lower(),
            'mention_data': [(x['mention'].lower(), int(x['offset']), x['kb_id'])
                             for x in _['mention_data'] if x['kb_id'] != 'NIL'
                             ]
        })


if not os.path.exists(r'all_chars_me.json'):
    chars = {}  # 字典 存储实体的属性在材料中出现的次数
    for d in tqdm(iter(id2kb.values())):
        for c in d['subject_desc']:
            chars[c] = chars.get(c, 0) + 1
    for d in tqdm(iter(train_data)):
        for c in d['text']:
            chars[c] = chars.get(c, 0) + 1
    chars = {i :j for i ,j in chars.items() if j >= min_count}  # 出现过的全保留下来 因为小于二就说明在训练集和知识库至少有一次没出现过 这种数据没意义
    id2char = { i +2 :j for i ,j in enumerate(chars)} # 0: mask, 1: padding
    char2id = {j :i for i ,j in id2char.items()}
    json.dump([id2char, char2id], open(r'all_chars_me.json', 'w'))
else:
    id2char, char2id = json.load(open(r'all_chars_me.json'))


if not os.path.exists(r'random_order_train.json'):  # 乱序的训练材料
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open(r'random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open(r'random_order_train.json'))


dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == mode  ]  # 打乱后重新组织成traindata,在train_json里面划分出验证集
train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != mode]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

def seq_padding2(X, padding=0):
    L = [x.shape[1] for x in X]
    ML = max(L)
    return np.array([
        np.concatenate(np.concatenate(x,np.zeros([128,ML-x.shape[1]])),axis=1) if x.shape[1] < ML else x for x in X
    ])


class data_generator:  # 针对训练数据的处理
    def __init__(self, data, batch_size=64):
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
            X1, X2, S1, S2, Y, T,Label = [], [], [], [], [], [],[]
            for i in idxs:
                d = self.data[i]
                text = d['text']  # 数据中的文本部分
                x1 = [char2id.get(c, 1) for c in text]# 从字符表中寻找到text中每个字
                s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                label = np.zeros(len(text))
                mds = {}  # 存储mentiondata的序列
                for md in d['mention_data']:  # 如果mentiondata中出现在了对应知识库中
                    if md[0] in kb2id:  # md[0]=mentiondata md[1]=offset md[2]=kbid。
                        j1 = md[1]
                        j2 = j1 + len(md[0])
                        s1[j1] = 1  # 实体位置标1 说明这是边界 分别规定了实体在text位置上的左边界与右边界
                        s2[j2 - 1] = 1
                        label[j1] = 1
                        label[j2-1] = 1
                        label[j1+1:j2-2] = 2
                        mds[(j1, j2)] = (md[0], md[2]) # 存储每个mention及其在kb中的id
                if mds:
                    for j1 ,j2 in mds.keys():
                        y = np.zeros(len(text))
                        y[j1: j2] = 1  # mention中的实体在 text长度的列表中标
                        x2 = kb2id[mds[(j1, j2)][0]]  # mention中的实体在kb中的id（一个实体对应多个id，取随机一个）
                        if mds[(j1, j2)][1] not in x2:
                            continue
                        h = x2.index(mds[(j1, j2)][1])
                        if (h > negative -1):
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
                            if x2[i] == mds[(j1, j2)][1]:  # mention中的实体是否与kb同名实体的随机抽取的id相一致
                                t=[1]
                            else:
                                t=[0]
                            x2change = id2kb[x2[i]]['subject_desc']  # 与x2为id的实体相联系的属性
                            x2change = [char2id.get(c, 1) for c in x2change]  # 转化为字符编码
                            X1.append(x1)
                            # X1bert.append(x1bert)
                            X2.append(x2change)
                            S1.append(s1)
                            S2.append(s2)
                            Y.append(y)
                            Label.append(label)
                            T.append(t)
                            if len(X1)==self.batch_size or i == idxs[-1]:
                                X1 = seq_padding(X1)  # 填充达到一致长度
                                # X1bert = np.array(X1bert)
                                X2 = seq_padding(X2)
                                S1 = seq_padding(S1)
                                S2 = seq_padding(S2)
                                Y = seq_padding(Y)
                                Label = seq_padding(Label)
                                T = seq_padding(T)
                                yield X1, to_categorical(Label, 4)
                                X1, X2, S1, S2, Y, Label,T = [], [], [], [], [], [],[]


# 模型定义
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。x1_bert=Input(shape=(None,None,))
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)

class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=True, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)
    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=tf.TensorShape((self.num_labels, self.num_labels)).as_list(),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        states = K.expand_dims(states[0], 2) # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0) # (1, output_dim, output_dim)
        output = K.logsumexp(states+trans, 1) # (batch_size, output_dim)
        return output+inputs, [output+inputs]
    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True) # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans*labels, [2,3]), 1, keepdims=True)
        return point_score+trans_score # 两部分得分之和
    def call(self, inputs): # CRF本身不改变输出，它只是一个loss
        return inputs
    def loss(self, y_true, y_pred): # 目标y_pred需要是one hot形式
        mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_states = [y_pred[:,0]] # 初始状态
        log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        return log_norm - path_score # 即log(分子/分母)
    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)

x1_in = Input(shape=(None,),dtype='int32') # 待识别句子输入
# x2_in = Input(shape=(None,)) # 实体语义表达输入 kb中相关属性的连接
# s1_in = Input(shape=(None,)) # 实体左边界（标签）
# s2_in = Input(shape=(None,)) # 实体右边界（标签）
# y_in = Input(shape=(None,)) # 实体标记 text中 mention的位置标记为1
# label= Input(shape=(None,))
# t_in = Input(shape=(1,)) # 是否有关联（标签）


x1 = x1_in# x2_in, s1_in, s2_in, y_in, t_in
x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
# x2_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x2)
embedding = Embedding(len(id2char)+2, char_size)


x1 = embedding(x1) #char embedding
x1 = Dropout(0.2)(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
h = Conv1D(char_size, 3, activation='relu', padding='same')(x1)
print(h)
crf = CRF(True)
tag_score = Dense(4,activation='softmax')(h) # 变成了5分类，第五个标签用来mask掉
tag_score = crf(tag_score)
s_model = Model(x1_in, tag_score) #识别实体，输入句子，输出实体识别的左右边界（如是s1为句子字符长度，实体左右边界标1）



s_model.compile(loss=crf.loss, # 用crf自带的loss
              optimizer='adam',
              metrics=[crf.accuracy] # 用crf自带的accuracy
             )
s_model.summary()


# def extract_items(text_in):  # 验证函数
#     _x1 = [char2id.get(c, 1) for c in text_in]
#     _x1 = np.array([_x1])
#     _k1, _k2 = s_model.predict(_x1)
#     _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
#     _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.5)[0]  # 大于0.5的识别成真实的位置
#     _subjects = []
#     for i in  _k1:
#         j = _k2[_k2 >= i]
#         if len(j) > 0:
#             j = j[0]
#             _subject = text_in[i: j+1]
#             _subjects.append((_subject, i, j) )  # 列表加入实体和左右边界
#     if _subjects:
#         R = []
#         _X2, _Y = [], []
#         _S, _IDXS = [], {}
#         for _s in _subjects:
#             _y = np.zeros(len(text_in))
#             _y[_s[1]: _s[2]] = 1
#             _IDXS[_s] = kb2id.get(_s[0], [])  # 找出知识库中与预测实体同名的实体集合
#             for i in _IDXS[_s]:
#                 _x2 = id2kb[i]['subject_desc']
#                 _x2 = [char2id.get(c, 1) for c in _x2]
#                 _X2.append(_x2)
#                 _Y.append(_y)
#                 _S.append(_s)
#         if _X2:
#             _X2 = seq_padding(_X2)
#             _Y = seq_padding(_Y)
#             _X1 = np.repeat(_x1, len(_X2), 0)
#             scores = t_model.predict([_X1, _X2, _Y])[:, 0]
#             for k, v in groupby(zip(_S, scores), key=lambda s: s[0]):  # 每一个预测出来的实体以及分数
#                 v = np.array([j[  1] for j in v])
#                 kbid = _IDXS[k][np.argmax(v)]  # 选择分数最高的
#                 R.append((k[0], k[1], kbid))  # 输出相关的位置和分数最高的实体的编码
#         return R
#     else:
#         return []
def max_in_dict(d): # 定义一个求字典中最大值的函数
    value=0
    key=0
    for i,j in zip(d.keys(),d.values()):
        if j > value:
            key,value = i,j
    return key,value

def viterbi(nodes, trans): # viterbi算法，跟前面的HMM一致
    paths = nodes[0] # 初始化起始路径
    for l in range(1, len(nodes)): # 遍历后面的节点
        paths_old,paths = paths,{}
        for n,ns in nodes[l].items(): # 当前时刻的所有节点
            max_path,max_score = '',-1e10
            for p,ps in paths_old.items(): # 截止至前一时刻的最优路径集合
                score = ns + ps + trans[p[-1]+n] # 计算新分数
                if score > max_score: # 如果新分数大于已有的最大分
                    max_path,max_score = p+n, score # 更新路径
            paths[max_path] = max_score # 储存到当前时刻所有节点的最优路径
    print(paths)
    return max_in_dict(paths)


# def cut(s, trans): # 分词函数，也跟前面的HMM基本一致
#     if not s: # 空字符直接返回
#         return []
#     # 字序列转化为id序列。注意，经过我们前面对语料的预处理，字符集是没有空格的，
#     # 所以这里简单将空格的id跟句号的id等同起来
#     sent_ids = np.array([[char2id.get(c, 0) if c != ' ' else char2id[u'。']
#                           for c in s]])
#     probas = s_model.predict(sent_ids)[0] # 模型预测
#     nodes = [dict(zip('012', i)) for i in probas[:, :4]] # 只取前3个
#     nodes[0] = {i:j for i,j in nodes[0].items() if i in 'bs'} # 首字标签只能是b或s
#     nodes[-1] = {i:j for i,j in nodes[-1].items() if i in 'es'} # 末字标签只能是e或s
#     tags = viterbi(nodes, trans)[0]
#     result = [s[0]]
#     for i,j in zip(s[1:], tags[1:]):
#         if j in 'bs': # 词的开始
#             result.append(i)
#         else: # 接着原来的词
#             result[-1] += i
#     return result

class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.6903
    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            s_model.save_weights(r'nercrf/ner_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        _ = s_model.get_weights()[-1][:3, :3]# 从训练模型中取出最新得到的转移矩阵
        # print(_)
        trans = {}
        for i in 'sbm':
            for j in 'sbm':
                trans[i + j] = _[tag2id[i], tag2id[j]]
        for d in tqdm(iter(dev_data)):
            _x1 = [char2id.get(c, 1) for c in d['text']]
            _x1 = np.array([_x1])
            # print(_x1)
            probas = s_model.predict(_x1)[0]  # 模型预测
            # print(probas)
            nodes = [dict(zip('sbm', i)) for i in probas[:, :3]]  # 只取前4个
            nodes[0] = {i: j for i, j in nodes[0].items() if i in 'bs'}  # 首字标签只能是b或s
            nodes[-1] = {i: j for i, j in nodes[-1].items() if i in 'bs'}  # 末字标签只能是e或s
            # print(len(nodes))
            tags = viterbi(nodes, trans)[0]
            ad = [0]*len(tags)
            num=0
            for i in range(len(tags)):
                if (tags[i]=='b'):
                    ad[i]=1
                    num+=1
                else:
                    ad[i]=0
            mentionall = [] #预测出来的实体对
            dictpre= []#实体写成训练集的json形式
            dictall=[] # 标签中含有的全部实体
            if num!=0 and num%2==0:
                count=[i for i,j in enumerate(ad) if j==1]
                while(len(count)!=0):
                    a=count[0]
                    del count[0]
                    b=count[0]
                    del count[0]
                    mentionall.append(d['text'][a:b+1])

            for mention_data in d['mention_data']: #d['mention_data']早已经被处理成元组了，只含有三个值 实体 位置 和kb编号
                dictin = (mention_data[0],)
                dictall.append(dictin)
            if mentionall:
                for mention in mentionall:
                    dictin = (str(mention),)
                    dictpre.append(dictin)
            # R=set(dictall)
            # T = set(d['mention_data'])
            R = set(dictpre)
            T = set(dictall)
            A += len(R & T)
            B += len(R)
            C += len(T)
        return 2 * A / (B + C), A / B, A / C



evaluator = Evaluate()
train_D = data_generator(train_data)
print(len(train_data))
if os.path.exists(r'nercrf/ner_model.weights'):
    s_model.load_weights(r'nercrf/ner_model.weights')
s_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=3125,
                          epochs=200,
                          callbacks=[evaluator],

                          )