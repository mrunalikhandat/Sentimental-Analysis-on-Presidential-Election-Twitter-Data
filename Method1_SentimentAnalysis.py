#!/usr/bin/env python
# coding: utf-8

# ### Reading the json file

# In[1]:


import json
tweets = []
for line in open('tweets.json', 'r'):
    tweets.append(json.loads(line))


# ### Analysing the data and finding the number of Obama and Romney tweets

# In[2]:


obama_cnt = 0
romney_cnt = 0
obama_romney_cnt = 0

import pandas as pd
tweets_df = pd.DataFrame(columns=['id','tweet'],index=None)

for i in range(0,len(tweets)):
    if(i % 10000 == 0):
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
print(obama_cnt)
print(romney_cnt)
print(obama_romney_cnt)


# In[3]:


import re
processed_tweet = []
for i in range (0,len(tweets_df)):
    if(i % 10000 == 0):
        print("i:",i)
    x = tweets_df.iloc[i]['tweet']
#     tweets_df.iloc[i]['tweet'] = ' '.join(re.sub("(RT)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split())
    temp = ' '.join(re.sub("(RT)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split())
    processed_tweet.append(temp)
tweets_df['processed_tweet'] = processed_tweet


# In[27]:


tweets_df.head()


# In[5]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

sid = SentimentIntensityAnalyzer()

tweets_with_score = pd.DataFrame(columns=['processed_tweet','score'])

for i in range (0,len(tweets_df)):
    if(i % 10000 == 0):
        print("i:",i)
    x = tweets_df.iloc[i]['processed_tweet']
    score = sid.polarity_scores(x)['compound']
    data = pd.DataFrame({"processed_tweet":[x],"score":[score]}) 
    tweets_with_score = tweets_with_score.append(data, ignore_index = True)


# In[8]:


print(tweets_with_score.head())
len(tweets_with_score)


# In[14]:


is_obama = []
is_romney = []
for i in range(0,len(tweets_with_score)):
    if(i % 20000 == 0):
        print("i:",i)
    obama = False;
    romney = False;
    for word in tweets_with_score.iloc[i]['processed_tweet'].split():
#         print(word)
        if word.lower() in ["obama","barack","obamabarack","barackobama"]:
            obama = True
            break
        elif word.lower() in["mitt","romney","mittromney","romneymitt"]:
            romney = True
            break
    is_obama.append(obama)
    is_romney.append(romney)
tweets_with_score['is_obama'] = is_obama
tweets_with_score['is_romney'] = is_romney


# In[15]:


tweets_with_score.head()


# In[18]:


from collections import Counter
Counter(tweets_with_score['is_obama'])


# In[20]:


obama_pos = 0
obama_neg = 0
obama_neu = 0
romney_pos = 0
romney_neg = 0
romney_neu = 0
for i in range (0, len(tweets_with_score)):
    if(i % 20000 == 0):
        print("i:",i)
    if tweets_with_score.iloc[i]['is_obama'] == True and tweets_with_score.iloc[i]['score'] > 0:
        obama_pos += 1
    elif tweets_with_score.iloc[i]['is_obama'] == True and tweets_with_score.iloc[i]['score'] < 0:
        obama_neg += 1
    elif tweets_with_score.iloc[i]['is_obama'] == True and tweets_with_score.iloc[i]['score'] == 0:
        obama_neu += 1
    elif tweets_with_score.iloc[i]['is_romney'] == True and tweets_with_score.iloc[i]['score'] > 0:
        romney_pos += 1
    elif tweets_with_score.iloc[i]['is_romney'] == True and tweets_with_score.iloc[i]['score'] < 0:
        romney_neg += 1
    elif tweets_with_score.iloc[i]['is_romney'] == True and tweets_with_score.iloc[i]['score'] == 0:
        romney_neu += 1
        
print("obama_pos",obama_pos)
print("obama_neg",obama_neg)
print("obama_neu",obama_neu)
print("romney_pos",romney_pos)
print("romney_neg",romney_neg)
print("romney_neu",romney_neu)


# In[44]:


# !pip install Pillow
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
from PIL import Image
  
# Reads 'Youtube04-Eminem.csv' file  
# df = pd.read_csv(r"Youtube04-Eminem.csv", encoding ="latin-1") 
  
comment_words = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
# for val in df.CONTENT: 
      
#     # typecaste each val to string 
#     val = str(val) 
  
#     # split the value 
#     tokens = val.split() 

j = 0;
for i in range (0, len(tweets_with_score)):
    if(i%20000 == 0):
        print(i)
    if tweets_with_score.iloc[i]['is_romney'] == True:
        tokens = tweets_with_score.iloc[i]['processed_tweet']
        j += 1
    # Converts each token into lowercase 
#     for i in range(len(tokens)): 
#         tokens[i] = tokens[i].lower() 
          
    for words in tokens.split(): 
        comment_words = comment_words + words.lower() + ' '
    if (j >= 1000):
        break
  
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 


# In[ ]:





# In[ ]:




