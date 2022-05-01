from time import time
import random
import pandas as pd
import numpy as np

from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
import os
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import train_test_split
import tensorflow

# The reduce_len parameter will allow a maximum of 3 consecutive repeating characters, while trimming the rest
# For example, it will tranform the word: 'Helloooooooooo' to: 'Hellooo'
tk = TweetTokenizer(reduce_len=True)
data = []

# Stopwords are frequently-used words (such as “the”, “a”, “an”, “in”) that do not hold any meaning useful to extract sentiment.
# If it's your first time ever using nltk, you can download nltk's stopwords using: nltk.download('stopwords')
from nltk.corpus import stopwords
STOP_WORDS = stopwords.words('english')

# Defining a handy function in order to load a given glove file

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


pathh=os.getcwd()+"\\ML Models\\glove.6B.50d.txt"
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(pathh)


# Further data cleaning
# A custom function defined in order to fine-tune the cleaning of the input text.
# This function is being "upgraded" such that it performs a more thourough cleaning of the data
# in order to better fit our words embedding layer
def cleaned(token):
    if token == 'u':
        return 'you'
    if token == 'r':
        return 'are'
    if token == 'some1':
        return 'someone'
    if token == 'yrs':
        return 'years'
    if token == 'hrs':
        return 'hours'
    if token == 'mins':
        return 'minutes'
    if token == 'secs':
        return 'seconds'
    if token == 'pls' or token == 'plz':
        return 'please'
    if token == '2morow' or token == '2moro' or token=='tmrw' or token=='tomorow':
        return 'tomorrow'
    if token == '2day':
        return 'today'
    if token == '4got' or token == '4gotten':
        return 'forget'
    if token in ['hahah', 'hahaha', 'hahahaha', 'hehehe', 'hahahah']:
        return 'haha'
    if token == "mother's":
        return "mother"
    if token == "mom's":
        return "mom"
    if token == "dad's":
        return "dad"
    if token == 'bday' or token == 'b-day':
        return 'birthday'
    if token in ["i'm", "don't", "can't", "couldn't", "aren't", "wouldn't", "isn't", "didn't", "hadn't",
                 "doesn't", "won't", "haven't", "wasn't", "hasn't", "shouldn't", "ain't","weren't", "should've", "would've","could've" ,"here's","where's"]:
        return token.replace("'", "")
    if token in ['lmao', 'lolz', 'rofl']:
        return 'lol'
    if token == '<3':
        return 'love'
    if token == 'thanx' or token == 'thnx':
        return 'thanks'
    if token == 'goood':
        return 'good'
    if token in ['amp', 'quot', 'lt', 'gt', '½25', '..', '. .', '. . .']:
        return ''
    if token in ['awh', 'aw', 'awww']:
        return 'aww'      
    if token=='awsome' or token=='awsm':
        return 'awesome'
    if token in ["g'night","gn","gooodnight"]:
        return 'goodnight'
    if token == '#fb':
        return 'fb'
    if token in ['proly','prolly']:
        return 'probably'
    if token in ['omfg','omgg']:
        return 'omg'
    if token == 'woho':
        return 'woohoo'
    if token == '#folowfriday' or token == 'tweps':
        return ''
    #added later
    if token in ['divorced','divorce', 'parted', 'separated', 'leaving', 'leave']:
        return 'left'
    if token in ['derogatory','depreciative','demeaning']:
        return 'harsh'

    return token


# This function will be our all-in-one noise removal function
def remove_noise(tweet_tokens):

    cleaned_tokens = []

    for token in tweet_tokens:
        # Eliminating the token if it is a link
        token = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", token)
        # Eliminating the token if it is a mention
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        
        cleaned_token = cleaned(token.lower())
        
        if cleaned_token == "idk":
            cleaned_tokens.append('i')
            cleaned_tokens.append('dont')
            cleaned_tokens.append('know')
            continue
        if cleaned_token == "i'll":
            cleaned_tokens.append('i')
            cleaned_tokens.append('will')
            continue
        if cleaned_token == "you'll":
            cleaned_tokens.append('you')
            cleaned_tokens.append('will')
            continue
        if cleaned_token == "we'll":
            cleaned_tokens.append('we')
            cleaned_tokens.append('will')
            continue
        if cleaned_token == "it'll":
            cleaned_tokens.append('it')
            cleaned_tokens.append('will')
            continue
        #added
        if cleaned_token == "they'll" or cleaned_token== "they'l":
            cleaned_tokens.append('they')
            cleaned_tokens.append('will')
            continue
        if cleaned_token == "he'll" or cleaned_token== "he'l":
            cleaned_tokens.append('he')
            cleaned_tokens.append('will')
            continue
        if cleaned_token == "she'll" or cleaned_token== "she'l":
            cleaned_tokens.append('she')
            cleaned_tokens.append('will')
            continue
        
        if cleaned_token == "it's":
            cleaned_tokens.append('it')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "i've":
            cleaned_tokens.append('i')
            cleaned_tokens.append('have')
            continue
        if cleaned_token == "you've":
            cleaned_tokens.append('you')
            cleaned_tokens.append('have')
            continue
        if cleaned_token == "we've":
            cleaned_tokens.append('we')
            cleaned_tokens.append('have')
            continue
        if cleaned_token == "they've":
            cleaned_tokens.append('they')
            cleaned_tokens.append('have')
            continue
        if cleaned_token == "you're":
            cleaned_tokens.append('you')
            cleaned_tokens.append('are')
            continue
        if cleaned_token == "we're":
            cleaned_tokens.append('we')
            cleaned_tokens.append('are')
            continue
        if cleaned_token == "they're":
            cleaned_tokens.append('they')
            cleaned_tokens.append('are')
            continue
        if cleaned_token == "let's":
            cleaned_tokens.append('let')
            cleaned_tokens.append('us')
            continue
        if cleaned_token == "she's":
            cleaned_tokens.append('she')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "he's":
            cleaned_tokens.append('he')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "that's":
            cleaned_tokens.append('that')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "i'd":
            cleaned_tokens.append('i')
            cleaned_tokens.append('would')
            continue
        if cleaned_token == "you'd":
            cleaned_tokens.append('you')
            cleaned_tokens.append('would')
            continue
        if cleaned_token == "there's":
            cleaned_tokens.append('there')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "what's":
            cleaned_tokens.append('what')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "how's":
            cleaned_tokens.append('how')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "who's":
            cleaned_tokens.append('who')
            cleaned_tokens.append('is')
            continue
        if cleaned_token == "y'all" or cleaned_token == "ya'll":
            cleaned_tokens.append('you')
            cleaned_tokens.append('all')
            continue
        #added by me
        if cleaned_token == "hadnt":
            cleaned_tokens.append('had')
            cleaned_tokens.append('not')
            continue
        if cleaned_token == "shouldnt":
            cleaned_tokens.append('should')
            cleaned_tokens.append('not')
            continue
        if cleaned_token == "werent":
            cleaned_tokens.append('were')
            cleaned_tokens.append('not')
            continue
        if cleaned_token == "shouldve":
            cleaned_tokens.append('should')
            cleaned_tokens.append('have')
            continue
        if cleaned_token == "wouldve":
            cleaned_tokens.append('would')
            cleaned_tokens.append('have')
            continue
        if cleaned_token=="tbh":
            cleaned_tokens.append('to')
            cleaned_tokens.append('be')
            cleaned_tokens.append('honest')
            continue
        if cleaned_token == "couldve":
            cleaned_tokens.append('could')
            cleaned_tokens.append('have')
            continue        
        #negation
        if cleaned_token=='dislike':
            cleaned_tokens.append('do')
            cleaned_tokens.append('not')
            cleaned_tokens.append('like')
            continue
        
       
        if cleaned_token.strip() and cleaned_token not in string.punctuation:
            cleaned_tokens.append(cleaned_token)
            
    return cleaned_tokens

# Prevewing the remove_noise() output
# print(remove_noise(data[0][0]))

start_time = time()

unks = []
UNKS = []

def cleared(word):
    res = ""
    prev = None
    for char in word:
        if char == prev: continue
        prev = char
        res += char
    return res

model_path=os.getcwd()+"\\ML Models\\BiLSTM_tune_1_rerun.h5"
built_model=tensorflow.keras.models.load_model(model_path)


def sentence_to_indices(sentence_words, max_len):
    X = np.zeros((max_len))
    print(X.shape)
    sentence_indices = []
    for j, w in enumerate(sentence_words):
        try:
            index = word_to_index[w]
        except:
            w = cleared(w)
            try:
                index = word_to_index[w]
            except:
                index = word_to_index['unk']
        X[j] = index
    return X

def predict_custom_tweet_sentiment(custom_tweet):
    # Convert the tweet such that it can be fed to the model
    x_input = sentence_to_indices(remove_noise(tk.tokenize(custom_tweet)), 162) #max_len=162 for our model final
    
    # Retrun the model's prediction
    return round(built_model.predict(np.array([x_input])).item(),3)

print(predict_custom_tweet_sentiment("I'm glad you're here!"))



