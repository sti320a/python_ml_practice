import glob
import re

import MeCab
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
import pprint

categories = [
    "sports-watch", "topic-news", "dokujo-tsushin", "peachy",
    "movie-enter", "kaden-channel", "livedoor-homme", "smax",
    "it-life-hack",
    ]

docs = []

for category in categories:
    for f in glob.glob(f"./text/{category}/[category]*.txt"):
        with open(f, "r", encoding="utf-8") as fin:
            url = next(fin).strip()
            date = next(fin).strip()
            title = next(fin).strip()
            body = "\n".join([line.strip() for line in fin if line.strip()])
        docs.append((category, url, date, title, body))

df = pd.DataFrame(
    docs,
    columns=["category", "url", "date", "title", "body"],
    dtype ="category"
)

df["date"] = pd.to_datetime(df["date"])

#print(df.head()) 先頭5行を表示

#pprint.pprint(df.category.value_counts())

tagger = MeCab.Tagger("-Owakati")

def parse_to_wakati(text):
    return tagger.parse(text).strip()

df = df.assign(body_wakati=df.body.apply(parse_to_wakati))    

#print(df.head()) 先頭5行を表示

# ラベルを数値に変換する（ここでラベルとはトピック）
le = LabelEncoder()

y = le.fit_transform(df.category)

#print(y[:10]) 最初の10件について、エンコードされたラベル（数値）を表示
#print(le.classes_)　エンコーダがエンコードしたクラスの種類を表示
#print(le.transform(["topic-news"])) topic-newsがどの数値にエンコードされたかを確認する

# 学習とテストに分けて、学習と推定を行う
X_train, X_test, y_train, y_test = train_test_split(
    df.body_wakati, #入力
    y, #正解ラベル
    test_size = 0.2, # テストデータのサイズ（全体の何割をテストデータにするか。20%をテスト用、80%を学習用にする）
    random_state = 42, # データをシャッフルした時の乱数の種を固定する
    shuffle = True # データに偏りができないようにシャッフル
)


from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import classification_report, confusion_matrix


# ナイーブベイズ分類器
# 単語の頻度を数え上げ、それを分類気が学習するパイプライン
text_clf = Pipeline([
    ("count_vec", CountVectorizer()),
    ("clf", MultinomialNB()),
])


# パイプラインに学習データを流し込む
text_clf.fit(X_train, y_train)

# テストデータを入力して分類
pred = text_clf.predict(X_test)

# 分類精度を評価して表示
print(classification_report(y_test, pred, target_names=le.classes_))

#混合行列を表示 間違えた場所を見られる
#print(confusion_matrix(y_test, pred, labels=le.transform(le.classes_))) 

#任意のテキストを入力し、学習したモデルを試す
original_text = "ここに任意のテキストを入力" # ここに任意のテキストを入力

wakati_text = parse_to_wakati(original_text)
print("The category of this documents is ["+le.inverse_transform(text_clf.predict([wakati_text]))[0] + "]")















