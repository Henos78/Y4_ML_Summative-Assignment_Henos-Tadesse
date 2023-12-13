# app.py
import glob
import streamlit as st
import numpy as np
import pandas as pd
import json
import re
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the model
model = load_model("chatbot.h5")

# Load the dataset
with open('intents1.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])

# Extract patterns, responses, and tags from the JSON data
dic = {"tag": [], "patterns": [], "responses": []}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

df = pd.DataFrame.from_dict(dic)


# Tokenize the patterns
tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
tokenizer.get_config()


# Encode the tags
ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')
lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])

def generate_answer(pattern):
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

# Streamlit App

# Streamlit App
st.title("Mental Health Chatbot")

st.text("You can type 'bye' to end the conversation.")


count = 0


while  True:
    
    user_input = st.text_input("You: ", key=count)
    
    if user_input.lower() == 'bye':
        st.text("Model: Goodbye! Feel free to return if you need assistance.")
        break
    
    if user_input:
        model_response = generate_answer(user_input)
        st.text(f"Model: {model_response}")
        count+=1
    


    
    

