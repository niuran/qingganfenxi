import jieba
import numpy as np
import pandas as pd
import jieba.analyse
import re
from sklearn.model_selection import train_test_split

import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from collections import Counter
from torch.utils.data import TensorDataset,DataLoader
from torch.optim.lr_scheduler import *

df1 = pd.read_csv('./train/waimai_train.csv')
df2 = pd.read_csv('./train/shopping_train.csv')
df3 = pd.read_csv('./train/movie_train.csv')

with open('./stop_words.txt', 'r', encoding='utf8') as f:
    stopwords = f.readlines()
for i in range(len(stopwords)):
    stopwords[i] = stopwords[i].replace('\n', '')
stopwords

alist = []
for s in df1['review']:
    s = re.sub('[^\u4e00-\u9fa5]+', '', s)
    result = list(jieba.cut(s))
    new_result = []
    for word in result:
        if word in stopwords:
            pass
        else:
            if word.isdigit():
                pass
            else:
                new_result.append(word)

    alist.append(' '.join(new_result))
df1['fenci'] = pd.Series(data=alist)
df1.drop(columns=['review'],inplace=True)

alist = []
for s in df2['review']:
    s = re.sub('[^\u4e00-\u9fa5]+', '', s)
    result = list(jieba.cut(s))
    new_result = []
    for word in result:
        if word in stopwords:
            pass
        else:
            if word.isdigit():
                pass
            else:
                new_result.append(word)

    alist.append(' '.join(new_result))
df2['fenci'] = pd.Series(data=alist)
df2.drop(columns=['review','cat'],inplace=True)

alist = []
for s in df3['comment']:
    s = re.sub('[^\u4e00-\u9fa5]+', '', s)
    result = list(jieba.cut(s))
    new_result = []
    for word in result:
        if word in stopwords:
            pass
        else:
            if word.isdigit():
                pass
            else:
                new_result.append(word)

    alist.append(' '.join(new_result))
df3['fenci'] = pd.Series(data=alist)
df3.drop(columns=['userId','movieId','timestamp','like','comment'],inplace=True)
df3['rating'].value_counts()
df3['rating'] = df3['rating'].map({1:0,5:1})
df3.rename(columns={'rating':'label'},inplace=True)

x_train1, x_test1, y_train1, y_test1 = train_test_split(df1['fenci'], df1['label'], test_size=0.2, random_state=0)
x_train2, x_test2, y_train2, y_test2 = train_test_split(df2['fenci'], df2['label'], test_size=0.2, random_state=0)
x_train3, x_test3, y_train3, y_test3 = train_test_split(df3['fenci'], df3['label'], test_size=0.2, random_state=0)

x_train = pd.concat((x_train1,x_train2,x_train3),axis=0)
y_train = pd.concat((y_train1,y_train2,y_train3),axis=0)

x_validation1, x_test_test1, y_validation1, y_test_test1 = train_test_split(x_test1, y_test1, test_size=0.06, random_state=0)
x_validation2, x_test_test2, y_validation2, y_test_test2 = train_test_split(x_test2, y_test2, test_size=0.06, random_state=0)
x_validation3, x_test_test3, y_validation3, y_test_test3 = train_test_split(x_test3, y_test3, test_size=0.06, random_state=0)

x_validation = pd.concat((x_validation1,x_validation2,x_validation3),axis=0)
y_validation = pd.concat((y_validation1,y_validation2,y_validation3),axis=0)

x_test = pd.concat((x_test_test1,x_test_test2,x_test_test3),axis=0)
y_test = pd.concat((y_test_test1,y_test_test2,y_test_test3),axis=0)

train = pd.DataFrame(data=y_train,columns=['label'])
train['content'] = x_train

validation = pd.DataFrame(data=y_validation,columns=['label'])
validation['content'] = x_validation

test = pd.DataFrame(data=y_test,columns=['label'])
test['content'] = x_test

train.to_csv('./train.txt',sep='\t',columns=None,index=False)
validation.to_csv('./validation.txt',sep='\t',columns=None,index=False)
test.to_csv('./test.txt',sep='\t',columns=None,index=False)

dft = pd.read_csv('./test/test_set_data.csv')
alist = []
for s in dft['review']:
    s = re.sub('[^\u4e00-\u9fa5]+', '', str(s))
    result = list(jieba.cut(s))
    new_result = []
    for word in result:
        if word in stopwords:
            pass
        else:
            if word.isdigit():
                pass
            else:
                new_result.append(word)

    alist.append(' '.join(new_result))
dft['fenci'] = pd.Series(data=alist)
dft.drop(columns=['review'],inplace=True)
df_p = pd.DataFrame(data={},columns=['label'])
df_p['fenci'] = dft['fenci']
df_p['label'] = 1

df_p.to_csv('./predict.txt',sep='\t',columns=None,index=False)


# 构建 word to id 词汇表并存储，形如 word: id。file: word2id 保存地址，save_to_path: 保存训 练语料库中的词组对应的 word2vec 到本地
def build_word2id(file, save_to_path=None):
    """
    :param file: word2id保存地址
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = ['./Dataset/train.txt', './Dataset/validation.txt']

    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    if save_to_path:
        with open(file, 'w', encoding='utf-8') as f:
            for w in word2id:
                f.write(w + '\t')
                f.write(str(word2id[w]))
                f.write('\n')

    return word2id


# 基于预训练的 word2vec 构建训练语料中所含词向量，fname: 预训练的 word2vec，word2id: 语 料文本中包含的词汇集，save_to_path: 保存训练语料库中的词组对应的 word2vec 到本地，语料文本 中词汇集对应的 word2vec 向量 id: word2vec。

def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs


# 分类类别以及 id 对应词典 pos:0, neg:1，classes: 分类标签;默认为 0:pos, 1:neg，返回分类标 签:id
def cat_to_id(classes=None):
    """
    :param classes: 分类标签；默认为0:pos, 1:neg
    :return: {分类标签：id}
    """
    if not classes:
        classes = ['0', '1']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
    return classes, cat2id


# 加载语料库，path: 样本语料库的文件，返回文本内容 contents，以及分类标签 labels(onehot 形式
def load_corpus(path, word2id, max_sen_len=50):
    """
    :param path: 样本语料库的文件
    :return: 文本内容contents，以及分类标签labels(onehot形式)
    """
    _, cat2id = cat_to_id()
    contents, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            label = sp[0]
            content = [word2id.get(w, 0) for w in sp[1:]]
            content = content[:max_sen_len]
            if len(content) < max_sen_len:
                content += [word2id['_PAD_']] * (max_sen_len - len(content))
            labels.append(label)
            contents.append(content)
    counter = Counter(labels)
    print('Total sample num：%d' % (len(labels)))
    print('class num：')
    for w in counter:
        print(w, counter[w])

    contents = np.asarray(contents)
    print(cat2id)
    #     labels = np.array([cat2id[l] for l in labels])
    labels = []
    for l in labels:
        if l in cat2id:
            labels.append(cat2id[l])

    return contents, labels

word2id = build_word2id('./Dataset/word2id.txt')
word2vec = build_word2vec('./Dataset/wiki_word2vec_50.bin', word2id)
print('train set: ')
train_contents, train_labels = load_corpus('./Dataset/train.txt', word2id, max_sen_len=50)
print('\nvalidation set: ')
val_contents, val_labels = load_corpus('./Dataset/validation.txt', word2id, max_sen_len=50)
print('\ntest set: ')
test_contents, test_labels = load_corpus('./Dataset/test.txt', word2id, max_sen_len=50)


class CONFIG():
    update_w2v = True  # 是否在训练中更新w2v
    vocab_size = 197806  # 词汇量，与word2id中的词汇量一致
    n_class = 2  # 分类数：分别为pos和neg
    embedding_dim = 50  # 词向量维度
    drop_keep_prob = 0.5  # dropout层，参数keep的比例
    kernel_num = 64  # 卷积层filter的数量
    kernel_size = [3, 4, 5]  # 卷积核的尺寸
    pretrained_embed = word2vec  # 预训练的词嵌入模型


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        update_w2v = config.update_w2v
        vocab_size = config.vocab_size
        n_class = config.n_class
        embedding_dim = config.embedding_dim
        kernel_num = config.kernel_num
        kernel_size = config.kernel_size
        drop_keep_prob = config.drop_keep_prob
        pretrained_embed = config.pretrained_embed

        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.embedding.weight.requires_grad = update_w2v
        # 卷积层
        self.conv1 = nn.Conv2d(1, kernel_num, (kernel_size[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, kernel_num, (kernel_size[1], embedding_dim))
        self.conv3 = nn.Conv2d(1, kernel_num, (kernel_size[2], embedding_dim))
        # Dropout
        self.dropout = nn.Dropout(drop_keep_prob)
        # 全连接层
        self.fc = nn.Linear(len(kernel_size) * kernel_num, n_class)

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        x = x.to(torch.int64)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv2)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv3)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

config = CONFIG()          # 配置模型参数
learning_rate = 0.001      # 学习率
BATCH_SIZE = 50            # 训练批量
EPOCHS = 10                 # 训练轮数
model_path = None          # 预训练模型路径
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_dataset = TensorDataset(torch.from_numpy(train_contents).type(torch.float),
                              torch.from_numpy(train_labels).type(torch.long))
train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE,
                              shuffle = True, num_workers = 2)

val_dataset = TensorDataset(torch.from_numpy(val_contents).type(torch.float),
                              torch.from_numpy(val_labels).type(torch.long))
val_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE,
                              shuffle = True, num_workers = 2)

# 配置模型，是否继续上一次的训练
model = TextCNN(config)
if model_path:
    model.load_state_dict(torch.load(model_path))
model.to(DEVICE)

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 设置损失函数
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5)


def train(dataloader, epoch):
    # 定义训练过程
    train_loss, train_acc = 0.0, 0.0
    count, correct = 0, 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

        if (batch_idx + 1) % 100 == 0:
            print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))

    train_loss *= BATCH_SIZE
    train_loss /= len(dataloader.dataset)
    train_acc = correct / count
    print('\ntrain epoch: {}\taverage loss: {:.6f}\taccuracy:{:.4f}%\n'.format(epoch, train_loss, 100. * train_acc))
    scheduler.step()

    return train_loss, train_acc


def validation(dataloader, epoch):
    model.eval()
    # 验证过程
    val_loss, val_acc = 0.0, 0.0
    count, correct = 0, 0
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item()
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

    val_loss *= BATCH_SIZE
    val_loss /= len(dataloader.dataset)
    val_acc = correct / count
    # 打印准确率
    print(
        'validation:train epoch: {}\taverage loss: {:.6f}\t accuracy:{:.2f}%\n'.format(epoch, val_loss, 100 * val_acc))

    return val_loss, val_acc


train_losses = []
train_acces = []
val_losses = []
val_acces = []

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train(train_dataloader, epoch)
    val_loss, val_acc = validation(val_dataloader, epoch)
    train_losses.append(tr_loss)
    train_acces.append(tr_acc)
    val_losses.append(val_loss)
    val_acces.append(val_acc)

model_pth = 'model_' + str(time.time()) + '.pth'
torch.save(model.state_dict(), model_pth)

# 设置超参数
model_path = model_pth    # 模型路径
# 加载测试集
test_dataset = TensorDataset(torch.from_numpy(test_contents).type(torch.float),
                            torch.from_numpy(test_labels).type(torch.long))
test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE,
                            shuffle = False, num_workers = 2)
# 读取模型
model = TextCNN(config)
model.load_state_dict(torch.load(model_path))


def test(dataloader):
    model.eval()
    model.to(DEVICE)

    # 测试过程
    count, correct = 0, 0
    olist = []
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        olist.append(output.argmax(1))
        correct += (output.argmax(1) == y).float().sum().item()
        count += len(x)

    # 打印准确率
    print('test accuracy:{:.2f}%.'.format(100 * correct / count))
    return olist


olist = test(test_dataloader)
# olist

print('\npredict set: ')
predict_contents, predict_labels = load_corpus('./Dataset/predict.txt', word2id, max_sen_len=50)
predict_labels = np.array([1 for i in range(22833)])
predict_labels.shape

# 设置超参数
model_path = model_pth  # 模型路径
# 加载测试集
predict_dataset = TensorDataset(torch.from_numpy(predict_contents).type(torch.float),
                                torch.from_numpy(predict_labels).type(torch.long))
predict_dataloader = DataLoader(dataset=predict_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=2)
# 读取模型
model = TextCNN(config)
model.load_state_dict(torch.load(model_path))


def predict(dataloader):
    model.eval()
    model.to(DEVICE)

    # 测试过程
    count, correct = 0, 0
    output_list = []
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        output_list.append(output.argmax(1))  # .argmax(1)

    return output_list


plist = predict(predict_dataloader)
plist

r = []
for i in plist:
    if i.float().sum().item()/len(i) >= 0.5:
        r.append(1)
    else:
        r.append(0)
#     r.append(i.float().sum().item()/len(i))
r