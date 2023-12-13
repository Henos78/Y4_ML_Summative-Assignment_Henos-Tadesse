#import the libraries
import numpy as np
import pandas as pd
import json
import re
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

model = load_model("chatbot.h5")

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['intents'])

    dic = {"tag": [], "patterns": [], "responses": []}
    for i in range(len(df)):
        ptrns = df[df.index == i]['patterns'].values[0]
        rspns = df[df.index == i]['responses'].values[0]
        tag = df[df.index == i]['tag'].values[0]
        for j in range(len(ptrns)):
            dic['tag'].append(tag)
            dic['patterns'].append(ptrns[j])
            dic['responses'].append(rspns)

    return pd.DataFrame.from_dict(dic)

def tokenize_and_encode(df):
    tokenizer = Tokenizer(lower=True, split=' ')
    tokenizer.fit_on_texts(df['patterns'])
    ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
    X = pad_sequences(ptrn2seq, padding='post')

    lbl_enc = LabelEncoder()
    y = lbl_enc.fit_transform(df['tag'])

    return tokenizer, X, lbl_enc, y

def generate_answer(pattern, model, tokenizer, X, lbl_enc, df):
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', pattern)
    txt = txt.lower()
    txt = txt.split()
    txt = " ".join(txt)
    text.append(txt)

    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test).squeeze()
    x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]

    return random.choice(responses)
