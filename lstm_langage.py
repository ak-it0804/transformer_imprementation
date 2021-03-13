import os
from glob import glob
import pandas as pd
import linecache
import pickle

import torch
import torch.nn as nn

import MeCab
import re
import torch

from sklearn.model_selection import train_test_split
import torch.optim as optim

import matplotlib.pyplot as plt
%matplotlib inline

import collections


categories = [name for name in os.listdir("./Data/text") if os.path.isdir("./Data/text/" + name)]
print(categories)

#カテゴリを配列で取得
def gen_text_data(categories):
    datasets = pd.DataFrame(columns = ["title","category"])
    for cat in categories:
        path = "./Data/text/" + cat +  "/*.txt"
        files = glob(path)
        for text_name in files:
            title = linecache.getline(text_name,3)
            s = pd.Series([title,cat],index = datasets.columns)
            datasets = datasets.append(s, ignore_index = True)

    #データフレームシャッフル
    datasets = datasets.sample(frac = 1).reset_index(drop = True)
    return datasets,categories

#datasets = gen_text_data(categories)


def load_text(categories):
    file_name = "./Data/" + "livedoor_news_sample.pickle"
    if os.path.exists(file_name):
        with open(file_name,"rb") as f:
            datasets = pickle.load(f)
    else:
        datasets = gen_text_data(categories)
        with open(file_name,"wb") as f:
            pickle.dump(datasets,f)
    
    return datasets

datasets = load_text(categories)

embeds = nn.Embedding(10,6) #(Embedding(単語の合計数, ベクトルの次元数)) ←適当なベクトルを作ってくれる

#特定の行を取り出し
w1 = torch.tensor([2])
#print(embeds(w1))

tagger = MeCab.Tagger("-Owakati")

def make_wakati(sentence):
    # MeCabで分かち書き
    sentence = tagger.parse(sentence)
    # 半角全角英数字除去
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # 記号もろもろ除去
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    # スペースで区切って形態素の配列へ
    wakati = sentence.split(" ")
    # 空の要素は削除
    wakati = list(filter(("").__ne__, wakati))
    return wakati

# test
test = "【人工衛星】は人間の「仕事」を奪った"
print(make_wakati(test))

# 単語ID辞書を作成する
word2index = {}
for title in datasets["title"]:
    wakati = make_wakati(title)
    for word in wakati:
        if word in word2index : continue
        word2index[word] = len(word2index)

print("vocab size : ",len(word2index))

#文書を単語IDの系列データに変換
#tensor方に変換
def sentence2index(sentence):
    wakati = make_wakati(sentence)
    return torch.tensor([word2index[w] for w in wakati], dtype = torch.long)

#テスト
test = "例のあのメニューもニコニコ超会議のフードコートメニュー１４種類紹介（前半)"
print(sentence2index(test))

'''
すでに出来上がってる単語群の辞書を利用して、
若きガチした
'''
#　全単語数の取得
VOCAB_SIZE = len(word2index)
#　単語のベクトル数
EMBEDDING_DIM = 10
test = "ユージの前に立ちはだかったjoy「僕はAKBの高橋みなみを守る"
#単語IDの系列データに変換
inputs = sentence2index(test)
#各単語のベクトルをまとめて取得
embeds = nn.Embedding(VOCAB_SIZE,EMBEDDING_DIM)
sentence_matrix = embeds(inputs)
print(sentence_matrix.size())
print(sentence_matrix)

VOCAB_SIZE = len(word2index)
EMBEDDING_DIM = 10
HIDDEN_DIM = 128

embeds = nn.Embedding(VOCAB_SIZE,EMBEDDING_DIM)
lstm = nn.LSTM(EMBEDDING_DIM,HIDDEN_DIM)
s1 = "震災を受けて感じたこと、大切だと思ったこと"
print(make_wakati(s1))

inputs1 = sentence2index(s1)
emb1 = embeds(inputs1)
lstm_inputs1 = emb1.view(len(inputs1),1,-1)
out1, out2 = lstm(lstm_inputs1)
print(out1)
print(out2)

# nn.Moduleを継承して新しいクラスを作る
class LSTMClassifier(nn.Module):
    # モデルで使う各ネットワークをコントラクタで定義
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        # 親クラスのコントラクタ
        super(LSTMClassifier, self).__init__()
        #隠れ層の次元数
        self.hidden_dim = hidden_dim
        # インプットの単語をベクトル化するために使う
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        #LSTMの隠れ層
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # LSTMの出力を受け取って全結合してsoftmaxを導入する
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        #softmax のLog版
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, sentence):
        #文章の各単語をベクトル化して出力。
        embeds = self.word_embeddings(sentence)
        #2次元テンソルをLSTMに食わせられるようにする
        _, lstm_out = self.lstm(embeds.view(len(sentence),1,-1))
        # 
        tag_space = self.hidden2tag(lstm_out[0].view(-1,self.hidden_dim))
        # softmaxに食わせて、確率として表現
        tag_scores = self.softmax(tag_space)
        return tag_scores

category2index = {}
for cat in categories:
    if cat in category2index: continue
    category2index[cat] = len(category2index)
print(category2index)

def category2tensor(cat):
    return torch.tensor([category2index[cat]], dtype = torch.long)

print(category2tensor("it-life-hack"))


traindata, testdata = train_test_split(datasets, train_size = 0.7)

#単語のベクトル次元数
EMBEDDING_DIM = 10

HIDDEN_DIM = 128

VOCAB_SIZE = len(word2index)

#分類さきのカテゴリ数
TAG_SIZE = len(categories)
#モデル宣言
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
#損失関数はNLLOSSを使用する
loss_function = nn.NLLLoss()
#最適化の手法はSGD
optimizer = optim.SGD(model.parameters(),lr = 0.01)

#各エポック数の合計lossを算出する
losses = []
#100エポック回してみる
for epoch in range(5):
    all_loss = 0
    for title, cat in zip(traindata["title"], traindata["category"]):
        #modelが持っている勾配の情報をリセット
        model.zero_grad()
        #文章の単語IDの系列に変換
        inputs = sentence2index(title)
        #順伝搬の結果を受け取る
        out = model(inputs)
        # 正解カテゴリをテンソル化
        answer = category2tensor(cat)
        # 正解とのlossを計算
        loss = loss_function(out,answer)
        #勾配をリセット
        loss.backward()
        #順伝搬でパラメータ更新
        optimizer.step()
        #lossを計算
        all_loss += loss.item()

    losses.append(all_loss)
    print("epoch", epoch, "\t", "loss", all_loss)
print("done")

plt.plot(losses)

#予測精度確認
test_num = len(testdata)
#正解の件数
a = 0
#購買自動計算OFF
with torch.no_grad():
    for title, category in zip(testdata["title"], testdata["category"]):
        #テストデータの予測
        inputs = sentence2index(title)
        out = model(inputs)

        #outの一番大きい要素を予測結果をする
        _, predict = torch.max(out, 1)

        answer = category2tensor(category)
        if predict == answer:
            a += 1
print("predict : ", a /test_num)


# 分析
traindata_num = len(trandata)
a = 0
with torch.no_grad():
    for title, category in zip(traindata["title"], traindata["category"]):
        inputs = sentence2index(title)
        out = model(inputs)
        _, predict = torch.max(out,1)
        answer = category2tensor(category)
        if predict == answer:
            a += 1
print("predict : ", a/traindata_num)


#IDをカテゴリに戻す用
index2category = {}
for cat, idx in category2index.items():
    index2category[idx] = cat

#answer ->正解ラベル　predict->LSTMの予測結果 exact -> 正解してたら０
predict_df = pd.DataFrame(columns = ["answer","predict","exact"])

#　予測して結果をDFに格納
with torch.no_grad():
    for title, catefory in zip(testdata["title"], testdata["category"]):
        out = model(sentence2index(title))
        _, predict = torch.max(out,1)
        answer = category2tensor(category)
        exact = "0" if predict.item() == answer.item() else "X"
        s = pd.Series([answer.item(),predict.item(),exact], index = predict_df.columns)
        predict_df = predict_df.append(s, ignore_index = True)

# Fスコア格納用のDF
fscore_df = pd.DataFrame(columns =["category","all","precision","recall","fscore"])

#分類きが答えた各カテゴリの件数
prediction_count = collections.Counter(predict_df["predict"])
#各カテゴリの総件数
answer_count = collections.Counter(predict_df["answer"])

#Fスコアを求める
for i in range(9):
    all_count = answer_count[i]
    precision = len(predict_df.query("predict == " + str(i) + "and exact == '0'")) / prediction_count[i]
    recall = len(predict_df.query("answer == " + str(i) + "and exact == '0'")) / all_count
    fscore = 2* precision*recall / (precision + recall)
    s = pd.Series([index2category[i], all_count, round(precision,2), round(recall,2), round(fscore,2)],index = fscore_df.columns)
    fscore_df = fscore_df.append(s,ignore_index = True)
print(fscore_df)
