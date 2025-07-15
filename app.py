import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load the model
model = load_model('next_word_pred.h5')
with open('tokenizer.pickle','rb') as file:
    tokenizer=pickle.load(file)

def predict_function(input_text):
    text= [input_text]
    token_text=tokenizer.texts_to_sequences([text][0])
    print(token_text)
    padded_token=pad_sequences(token_text,maxlen=14,padding='pre')
    #prediction
    pred=model.predict(padded_token)
    # now we will take the max in them
    pos=np.argmax(pred)
    for word,index in tokenizer.word_index.items():
        if index == pos:
            #input_text=input_text+' '+word
            return word
    return word


st.title('Next Word Predictor using LSTM')
input_text = st.text_input("Enter the sentence ")
if st.button('Predict Next Word'):
    st.write(f"Next Word is : {predict_function(input_text)}")


