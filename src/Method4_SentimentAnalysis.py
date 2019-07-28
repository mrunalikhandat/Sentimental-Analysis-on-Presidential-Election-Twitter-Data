
# coding: utf-8

# In[1]:


import pandas as pd
import json
import textwrap
import string

tweets = []
for line in open('tweets.json', 'r'):
    tweets.append(json.loads(line))

obama_cnt = 0
romney_cnt = 0
obama_romney_cnt = 0

tweets_df = pd.DataFrame(columns=['id','tweet'],index=None)

# for tweet in tweets[0:10000]:
#     data = pd.DataFrame({"id":[tweet['id_str']],"tweet":[tweet['text']]})
#     tweets_df = tweets_df.append(data, ignore_index = True)

for i in range(0,len(tweets)):
    if(i % 100000 == 0):
        print("i:",i)
    obama = False;
    romney = False;
    for word in tweets[i]['text'].split():
#         print(word)
        if word.lower() in ["obama","barack","barackobama","obamabarack"]:
            obama = True
        if word.lower() in["mitt","romney","mittromney","romneymitt"]:
            romney = True
    if obama == True and romney == False:
        data = pd.DataFrame({"id":[tweets[i]['id_str']],"tweet":[tweets[i]['text']]})
        tweets_df = tweets_df.append(data, ignore_index = True)
        obama_cnt += 1
    elif obama == False and romney == True:
        data = pd.DataFrame({"id":[tweets[i]['id_str']],"tweet":[tweets[i]['text']]})
        tweets_df = tweets_df.append(data, ignore_index = True)
        romney_cnt += 1
#     elif obama == True and romney == True:
#         obama_romney_cnt += 1

print('Tweets about Obama: ' + str(obama_cnt))
print('Tweets about Romney: ' + str(romney_cnt))
# print(obama_romney_cnt)


# In[2]:


tweets_df


# In[3]:


tweets_df.iloc[0]['tweet']


# In[4]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
# print(stopWords)

filteredStopWords = {'y', 'he', 'make', 'doing', 'until', 'your', "it's", 'because', 'am', 'here', 'to', 'didn', "you're", 'doesn', 'are', 'had', 'again', 'aren', 'him', 'did', 'couldn', 'where', "won't", 'being', 'before', 'they', 'was', 'both', "needn't", "that'll", 'whom', 'only', 'too', 'out', 'own', "doesn't", 'ma', 'yours', "she's", 'a', 'ourselves', 'me', "shouldn't", 'under', 'then', 'an', 'can', 'o', 'this', "aren't", 'which', 'mustn', 'or', 'each', 'some', 'be', 'down', 'she', 'few', 'won', 'such', 'himself', 'hasn', 'at', 'shan', 'them', 'will', 'between', 'itself', 'our', 'by', 'we', 'up', 'of', 'it', 'don', 'is', 'while', 'than', 'mightn', 'its', "haven't", "isn't", 'theirs', 'd', 'as', 'shouldn', 'over', 'these', 'those', 'there', 'just', 'hers', 'her', 'after', 'for', "wasn't", "you'll", 'his', 'has', 'who', 'having', 'hadn', 'further', 'isn', "you'd", 'ain', 't', 'now', "shan't", 'into', 'been', 'in', 'same', 'any', 'very', 'do', 'if', 'you', "couldn't", 'll', 's', 'below', 'weren', 'their', 'herself', 'he', 'm', 'my', 'and', "mightn't", 'all', 'through', "weren't", 're', 'once', 'why', "didn't", 'needn', 'i', 'yourself', 'what', 'themselves', 'against', "don't", 'myself', 'more', 'ours', 'no', 'yourselves', 'nor', 'with', 'were', "mustn't", "you've", 'that', "wouldn't", 'most', 'wouldn', 'off', 'on', 'should', 've', 'but', 'haven', 'does', 'the', 'from', "should've", 'how', 'during', "hasn't", 'wasn', "hadn't", 'so', 'other', 'about', 'above', 'have', 'when'}

for i in range (0,len(tweets_df)):
    wordsFiltered = []
    words = word_tokenize(tweets_df.iloc[i]['tweet'])
    for w in words:
        if w not in filteredStopWords:
            wordsFiltered.append(w)
    tweets_df.iloc[i]['tweet'] = wordsFiltered


# In[5]:


tweets_df


# In[6]:


tweets_df.iloc[0]['tweet']


# In[7]:


pos_words = []
neg_words = []

with open('positive.txt') as my_file:
    for line in my_file:
        pos_words.append(line.rstrip())

with open('negative.txt') as my_file:
    for line in my_file:
        neg_words.append(line.rstrip())


# In[8]:


def reviewRating(review, table):
    pos_word_count = 0
    neg_word_count = 0
    words = review

    bo = False
    mr = False
    
    for word in words:
        if word.lower() in ["obama","barack","barackobama","obamabarack"]:
            bo = True
        elif word.lower() in["mitt","romney","mittromney","romneymitt"]:
            mr = True
            
    negate = False
    for word in words:
        w = word.translate(table).lower()
        
        if w is "not":
            # to check if the next word is positive (negative classification) or negative (positive classification)
            negate = True
        elif w in pos_words:
            if (negate):
                neg_word_count = neg_word_count + 1
                negate = False
            else:
                pos_word_count = pos_word_count + 1
        elif w in neg_words:
            if (negate):
                pos_word_count = pos_word_count + 1
                negate = False
            else:
                neg_word_count = neg_word_count + 1
 
    
    if (pos_word_count > neg_word_count and (bo)):
        return "obama positive"
    elif (pos_word_count > neg_word_count and (mr)):
        return "romney positive"
    elif(pos_word_count < neg_word_count and (bo)):
        return "obama negative"
    elif(pos_word_count < neg_word_count and (mr)):
        return "romney negative"
    else:
        return "neutral"


# In[9]:


table = str.maketrans({key: None for key in string.punctuation})
# text = tweets_df.iloc[3]['tweet']

obama_pos = 0
obama_neg = 0
romney_pos = 0
romney_neg = 0
neutral = 0
for i in range (0,len(tweets_df)):
    text = tweets_df.iloc[i]['tweet']
    result = reviewRating(text, table)
    if result == 'obama positive':
        obama_pos = obama_pos + 1
    elif result == 'obama negative':
        obama_neg = obama_neg + 1
    elif result == 'romney positive':
        romney_pos = romney_pos + 1
    elif result == 'romney negative':
        romney_neg = romney_neg + 1
    else:
        neutral = neutral + 1

print('Positive for Obama: ' + str(obama_pos))
print('Negative for Obama: ' + str(obama_neg))
print('Positive for Romney: ' + str(romney_pos))
print('Negative for Romney: ' + str(romney_neg))

total_obama = obama_pos + romney_neg
total_romney = obama_neg + romney_pos

print('Total Votes for Obama: ' + str(total_obama))
print('Total Votes for Romney: ' + str(total_romney))

