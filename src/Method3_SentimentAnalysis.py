#!/usr/bin/env python
# coding: utf-8

# ### Reading the json file

# In[1]:


import json
tweets = []
for line in open('tweets.json', 'r'):
    tweets.append(json.loads(line))


# ### Checking the data

# In[13]:


tweets[0]


# ### Creating Dataframe by extracting the features that we need

# In[3]:


import pandas as pd
tweets_df = pd.DataFrame(columns=['id','tweet'],index=None)
for tweet in tweets[0:1000]:
    data = pd.DataFrame({"id":[tweet['id_str']],"tweet":[tweet['text']]})
    tweets_df = tweets_df.append(data, ignore_index = True)


# In[16]:


cnt = 0
for i in range (0,1000000):
    if len(tweets[i]['entities']['user_mentions']) > 0:
#         print(tweets[i]['text'])
#         print(tweets[i]['entities']['user_mentions'])
        cnt += 1
        
print(cnt)


# In[59]:


# obama_cnt = 0
# romney_cnt = 0
# obama_romney_cnt = 0
# for i in range(0,1000000):
#     obama = False;
#     romney = False;
#     for word in tweets[i]['text'].split():
# #         print(word)
#         if word.lower() in ["obama","barack","barackobama","obamabarack"]:
#             obama = True
#         if word.lower() in["mitt","romney","mittromney","romneymitt"]:
#             romney = True
#     if obama == True and romney == False:
#         obama_cnt += 1
#     elif obama == False and romney == True:
#         romney_cnt += 1
#     elif obama == True and romney == True:
#         obama_romney_cnt += 1
# print(obama_cnt)
# print(romney_cnt)
# print(obama_romney_cnt)


# In[19]:


tweets[0]['text']


# ### Analysing the data and finding the number of Obama and Romney tweets

# In[34]:


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


# In[35]:


print(len(tweets_df))


# ### Removing the unnecessary data from the tweet like RT symbols, hyperlinks usermention symbols, hashtag symbols, etc

# In[44]:


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


# In[45]:


tweets_df.head()


# ### Extract the Adjectives, verbs and adverbs from the tweet

# In[47]:


from nltk.tokenize import word_tokenize
import nltk
tweets_imp_words_df = pd.DataFrame(columns=['id','Adj_Adv_Verb','pos'],index=None)
for i in range (0,len(tweets_df)):
    if(i % 10000 == 0):
        print("i:",i)
    text = word_tokenize(tweets_df.iloc[i]['processed_tweet'])
    pos_tagged_words = nltk.pos_tag(text)
    tempStr = ''
    pos = []
    for i in range(0,len(pos_tagged_words)):
        if pos_tagged_words[i][1] in ['JJ','JJR','JJS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ']:
            tempStr += pos_tagged_words[i][0]+" "
            pos.append(pos_tagged_words[i][1])
    tempData = pd.DataFrame({"id":[tweets_df.iloc[i]['id']],"Adj_Adv_Verb":[tempStr],"pos":[pos]})
    tweets_imp_words_df = tweets_imp_words_df.append(tempData, ignore_index = True)


# In[50]:


tweets_imp_words_df.head()
print(len(tweets_imp_words_df))


# ### Scoring the extracted adjectives, verbs and adverbs

# In[51]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

sid = SentimentIntensityAnalyzer()
individual_score = []
for i in range (0,len(tweets_imp_words_df)):
    if(i % 10000 == 0):
        print("i:",i)
    temp_individual_score = []
    for word in tweets_imp_words_df.iloc[i]['Adj_Adv_Verb'].split():
        temp_individual_score.append(sid.polarity_scores(word)['compound'])
    individual_score.append(temp_individual_score)
tweets_imp_words_df['individual_scores'] = individual_score


# ### Rescoring the adjectives, verbs and adverbs based on the method

# In[52]:


for i in range (0,len(tweets_imp_words_df)):
    if(i % 10000 == 0):
        print("i:",i)
    pos = tweets_imp_words_df.iloc[i]['pos']
    for j in range (1,len(tweets_imp_words_df.iloc[i]['individual_scores'])):
        if(pos[j] in ['JJ','JJR','JJS'] and pos[j-1] not in ['JJ','JJR','JJS']):
            if(tweets_imp_words_df.iloc[i]['individual_scores'][j-1] > 0):
                tweets_imp_words_df.iloc[i]['individual_scores'][j-1] *= tweets_imp_words_df.iloc[i]['individual_scores'][j]
            elif(tweets_imp_words_df.iloc[i]['individual_scores'][j-1] < 0):
                tweets_imp_words_df.iloc[i]['individual_scores'][j] = 5 - tweets_imp_words_df.iloc[i]['individual_scores'][j]


# ### Finding the final score of the tweet

# In[53]:


score = []
for i in range (0,len(tweets_imp_words_df)):
    temp_score = 0;
    no_of_adj = 0;
    pos = tweets_imp_words_df.iloc[i]['pos']
    for j in range (0,len(tweets_imp_words_df.iloc[i]['individual_scores'])):
        temp_score += tweets_imp_words_df.iloc[i]['individual_scores'][j]
        if(pos[j] in ['JJ','JJR','JJS']):
            no_of_adj += 1;
    if no_of_adj > 0:
        temp_score = temp_score/no_of_adj;
    score.append(temp_score)
tweets_imp_words_df['score'] = score


# In[60]:


# pos = 0;
# neg = 0;
# neu = 0;
# for i in range (0,len(tweets_imp_words_df)):
#     if(i % 10000 == 0):
#         print("i:",i)
#     if tweets_imp_words_df.iloc[i]['score'] > 0:
#         pos += 1
#     elif  tweets_imp_words_df.iloc[i]['score'] < 0:
#         neg += 1
#     else:
#         neu += 1


# In[61]:


# print(pos)
# print(neg)
# print(neu)


# In[57]:


is_obama = []
is_romney = []
for i in range(0,len(tweets_imp_words_df)):
    if(i % 20000 == 0):
        print("i:",i)
    obama = False;
    romney = False;
    for word in tweets_df.iloc[i]['tweet'].split():
#         print(word)
        if word.lower() in ["obama","barack","obamabarack","barackobama"]:
            obama = True
        if word.lower() in["mitt","romney","mittromney","romneymitt"]:
            romney = True
    is_obama.append(obama)
    is_romney.append(romney)
tweets_imp_words_df['is_obama'] = is_obama
tweets_imp_words_df['is_romney'] = is_romney


# ### Finding the number of positive and negative tweets for Obama and Romney

# In[58]:


obama_pos = 0
obama_neg = 0
obama_neu = 0
romney_pos = 0
romney_neg = 0
romney_neu = 0
for i in range (0, len(tweets_imp_words_df)):
    if(i % 20000 == 0):
        print("i:",i)
    if tweets_imp_words_df.iloc[i]['is_obama'] == True and tweets_imp_words_df.iloc[i]['score'] > 0:
        obama_pos += 1
    elif tweets_imp_words_df.iloc[i]['is_obama'] == True and tweets_imp_words_df.iloc[i]['score'] < 0:
        obama_neg += 1
    elif tweets_imp_words_df.iloc[i]['is_obama'] == True and tweets_imp_words_df.iloc[i]['score'] == 0:
        obama_neu += 1
    elif tweets_imp_words_df.iloc[i]['is_romney'] == True and tweets_imp_words_df.iloc[i]['score'] > 0:
        romney_pos += 1
    elif tweets_imp_words_df.iloc[i]['is_romney'] == True and tweets_imp_words_df.iloc[i]['score'] < 0:
        romney_neg += 1
    elif tweets_imp_words_df.iloc[i]['is_romney'] == True and tweets_imp_words_df.iloc[i]['score'] == 0:
        romney_neu += 1
        
print("obama_pos",obama_pos)
print("obama_neg",obama_neg)
print("obama_neu",obama_neu)
print("romney_pos",romney_pos)
print("romney_neg",romney_neg)
print("romney_neu",romney_neu)


# ### Creating the results grapph for comparing all the methods that were executed

# In[55]:


import numpy as np
import matplotlib.pyplot as plt

N = 6
ObamaScore = (51.1, 46.8, 66, 53, 41.1, 57.1)

fig, ax = plt.subplots()

ind = np.arange(N)   
width = 0.35
p1 = ax.bar(ind - width/2, ObamaScore, width, color='r', bottom=0)

RomneyScore = (47.2, 53.2, 34, 47, 58.8, 42.9)

p2 = ax.bar(ind + width/2, RomneyScore, width,color='y', bottom=0)

ax.set_title('Vote percentages for Obama and Romney')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Actual', 'Method1', 'Method2', 'Method3', 'Method4', 'Method5'))

ax.legend((p1[0], p2[0]), ('Obama', 'Romney'))
ax.set_ylabel("Vote percentage")

def autolabel(rects, xpos='center'):
    
# for labelling the bar graph with its values

    xpos = xpos.lower()  
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.00, 'left': 0.95}

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


autolabel(p1, "left")
autolabel(p2, "right")

plt.show()


# ### Graph for the analysis of percentage of Obama and Romney tweets

# In[74]:


import matplotlib.pyplot as plt

324805
30092
51443

labels = ['Only Obama Tweets', 'Only Romney Tweets', 'Obama and Romney Tweets', 'Neither Obama nor Romney Tweets']
sizes = [224805/1000000, 130092/1000000, 51443/1000000, (1000000-324805+30092+51443)/1000000]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")

plt.axis('equal')
plt.tight_layout()
plt.show()


# In[ ]:




