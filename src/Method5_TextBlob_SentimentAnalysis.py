#!/usr/bin/env python
# coding: utf-8

# #### Importing libraries

# In[1]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import simplejson as json
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


# #### Reading JSON file in 'tweets'

# In[2]:


tweets = []
for line in open('tweets.json', 'r'):
    tweets.append(json.loads(line))


# #### Print one line of JSON file

# In[3]:


print("\nTraining Doc Sample - ",tweets[:1])


# #### Storing first 1000 tweets and data, to check if correct data gets displayed

# In[4]:


import pandas as pd
tweets_df = pd.DataFrame(columns=['id','tweet'],index=None)
for tweet in tweets[0:1000]:
    data = pd.DataFrame({"id":[tweet['id_str']],"tweet":[tweet['text']]})
    tweets_df = tweets_df.append(data, ignore_index = True)


# ##### Printing first 10 rows of dataframe 'tweets_df'

# In[5]:


tweets_df.head(20)


# #### Keeping tweets with only Obama or Romney mention
# Input- 'tweets' that read JSON file
# Output- 'tweets_df2'- Dataframe containing tweets' id and tweets' text as 'id' and 'tweet'  
# All the instances of tweets_df2['tweet'] contain either Obama or Romney mentioned in them    
# 'i' is printed to see the instances executed so far

# In[10]:


obama_cnt = 0
romney_cnt = 0
obama_romney_cnt = 0

import pandas as pd
tweets_df1 = pd.DataFrame(columns=['id','tweet'],index=None)

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
        tweets_df1 = tweets_df1.append(data, ignore_index = True)
        obama_cnt += 1
    elif obama == False and romney == True:
        data = pd.DataFrame({"id":[tweets[i]['id_str']],"tweet":[tweets[i]['text']]})
        tweets_df1 = tweets_df1.append(data, ignore_index = True)
        romney_cnt += 1
#     elif obama == True and romney == True:
#         obama_romney_cnt += 1
print(obama_cnt)
print(romney_cnt)
#print(obama_romney_cnt)


# #### Function defined to remove patterns in 'tweets_df2['tweet']'

# In[37]:


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 


# #### Defining Patterns to be removed

# In[91]:


tweets_df1['tweet'] = np.vectorize(remove_pattern)(tweets_df1['tweet'], "@[\w]*")


# In[93]:


tweets_df1.head(10)


# In[94]:


tweets_df1['tweet'] = tweets_df1['tweet'].str.replace("[^a-zA-Z#]", " ")


# In[43]:


len(tweets_df1)


# #### Separating each word from the 'tweet'

# In[44]:


tokenized_tweets_df = tweets_df1['tweet'].apply(lambda x: x.split())
tokenized_tweets_df.head()


# #### Stemming individual words
# 
# Stemming returns words in their basic form

# In[45]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

stemmed_tweets_df = tokenized_tweets_df.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
stemmed_tweets_df.head()


# #### Combining stemmed words as a sentence for each tweet

# In[46]:


for i in range(len(stemmed_tweets_df)):
    stemmed_tweets_df[i] = ' '.join(stemmed_tweets_df[i])

tweets_df1['tweet'] = stemmed_tweets_df


# In[23]:


tweets_df1.head()


# #### Converting polarity scores to -1, 0 or 1
# if polarity score is negative, sentiment is assigned to -1  
# if polarity score is positive, sentiment is assigned to +1  
# if polarity score is neutral, sentiment is assigned to 0  
# 

# In[47]:


def sentiment_conversion(sentiment_of_tweet):
    if float(sentiment_of_tweet) > 0.0:
        return 1
    elif float(sentiment_of_tweet) < 0.0:
        return -1
    else:
        return 0


# Install textblob, only needed for the first time

# In[48]:


#! pip install textblob


# importing textblob

# In[49]:


from textblob import TextBlob as tb


# #### Getting polarity scores of each tweet, storing it in the form of array 'pol'
# Polarity Scores are converted to -1,0,1 by caaling 'get_sentiment' function
# Assign new sentiment score to 'sentiment_conv'
# 'tweets_df2' dataframe contains 'tweet' and 'sentiment'

# In[50]:


pol = []
tweets_df2 = pd.DataFrame(columns=['tweet', 'sentiment'],index=None)

for twee in tweets_df1['tweet']:
    testimonial = tb(twee)
    sentiment_of_tweet = testimonial.sentiment.polarity
    pol.append(sentiment_of_tweet)
    sentiment_conv = sentiment_conversion(sentiment_of_tweet)
    data = pd.DataFrame({"tweet":[twee[0:]], "sentiment":[sentiment_conv]})
    tweets_df2 = tweets_df2.append(data, ignore_index = True)
    


# In[51]:


tweets_df2.head(10)


# In[52]:


pol


# #### Calculating overall sentiments of tweets as Positive, Neutral and Negative

# In[53]:


pos = []
neg = []
neu = []
pol_ap = []


for p in pol:
    if float(p) > 0.0:
        pos.append(p)
        pol_ap.append(1)
        
    if float(p) < 0.0:
        neg.append(p)
        pol_ap.append(-1)

        
    if p not in pos:
        if p not in neg:
            neu.append(p)
            pol_ap.append(0)
            
            
print(len(pos))
print(len(neg))
print(len(neu))


# #### Plotting Pie Chart to depict overall sentiments of tweets as Positive, Negative and Neutral

# In[75]:


plot_df = pd.DataFrame({'sentiment': [len(pos), len(neg), len(neu)]}, index=['Positive_Sentiments', 'Negative_Sentiments', 'Neutral_Sentiments'])
plot = plot_df.plot.pie(y='sentiment', figsize=(5, 5))


# In[80]:


#plot = plot_df.plot.bar(y='sentiment', figsize=(5, 5))


# In[54]:


len(pol_ap)


# In[55]:


tweets_df2.head(10)


# In[56]:


len(tweets_df2)


# #### Getting Positive, Negative and Neutral Sentiments towards Obama and Romney

# In[57]:


obama_sentiment_positive = 0
obama_sentiment_negative = 0
obama_sentiment_neutral = 0
romney_sentiment_positive = 0
romney_sentiment_negative = 0
romney_sentiment_neutral = 0

i = 0
#tweets_df2[0]['tweet']
for twee in tweets_df2['tweet']:
    
    for word in twee.split():
        #print(word)
        if word.lower() in ["obama","barack","barackobama","obamabarack"]:
            if pol_ap[i] == -1:
                obama_sentiment_negative += 1
            elif pol_ap[i] == 1:
                obama_sentiment_positive += 1
            else:
                obama_sentiment_neutral += 1
            i += 1
            break
        if word.lower() in["mitt","romney","mittromney","romneymitt"]:
            if pol_ap[i] == -1:
                romney_sentiment_negative += 1
            elif pol_ap[i] == 1:
                romney_sentiment_positive += 1
            else:
                romney_sentiment_neutral += 1            
            i += 1
            break
print(obama_sentiment_positive)
print(obama_sentiment_negative)
print(obama_sentiment_neutral)
print(romney_sentiment_positive)
print(romney_sentiment_negative)
print(romney_sentiment_neutral)
        


# #### Printing Result-

# In[59]:


print("obama_sentiment_positive",obama_sentiment_positive)
print("obama_sentiment_negative",obama_sentiment_negative)
print("obama_sentiment_neutral",obama_sentiment_neutral)
print("romney_sentiment_positive",romney_sentiment_positive)
print("romney_sentiment_negative",romney_sentiment_negative)
print("romney_sentiment_neutral",romney_sentiment_neutral)


# #### Plotting Sentiments of tweets towards Obama and Romney 

# In[79]:


plot_df = pd.DataFrame({'sentiment': [obama_sentiment_positive, obama_sentiment_negative, obama_sentiment_neutral,romney_sentiment_positive,romney_sentiment_negative,romney_sentiment_neutral]}, index=['Obama Positive Sentiments', 'Obama Negative Sentiments', 'Obama Neutral Sentiments','Romney Positive Sentiments', 'Romney Negative Sentiments', 'Romney Neutral Sentiments'])
plot = plot_df.plot.bar(y='sentiment', figsize=(15, 10))


# #### Calculating number of mentions for each Obama and Romney

# In[58]:


obama_mention = 0
romney_mention = 0
obama_mention = obama_sentiment_positive + obama_sentiment_negative + obama_sentiment_neutral
romney_mention = romney_sentiment_positive + romney_sentiment_negative + romney_sentiment_neutral

print(romney_mention)              
print(obama_mention)


# #### Calculating total number of tweets favoring Obama and for Romney, percentage of votes calculation
# 
# Assumption: Negative # of sentiments for Obama are considered as Positive # of sentiments for Romney

# In[60]:


votes_for_obama = 0
votes_for_romney = 0
Total_votes = 0
percentage_votes_obama = 0.0
percentage_votes_romney = 0.0
votes_for_obama = obama_sentiment_positive + romney_sentiment_negative
votes_for_romney = obama_sentiment_negative + romney_sentiment_negative
Total_votes = votes_for_obama + votes_for_romney
percentage_votes_obama = (votes_for_obama*100)/Total_votes
percentage_votes_romney = (votes_for_romney*100)/Total_votes


# #### Printing Result

# In[61]:


print("Percentage Votes for Obama:", percentage_votes_obama)
print("Percentage Votes for Romney:", percentage_votes_romney)


# #### Plotting Predicted Popular Vote Percentage

# In[86]:


plot_df = pd.DataFrame({'percentage_votes': [percentage_votes_obama, percentage_votes_romney]}, index=['Votes for Obama', 'Votes for Romney'])
plot = plot_df.plot.pie(y='percentage_votes', figsize=(5, 5), title = 'Predicted Vote Percentage')


# #### Plotting Actual Popular Vote Percentage in 2012 USA Presidential Election

# In[88]:


plot_df = pd.DataFrame({'percentage_votes': [51.1, 47.2]}, index=['Votes for Obama', 'Votes for Romney'])
plot = plot_df.plot.pie(y='percentage_votes', figsize=(5, 5), title = 'Actual Vote Percentage')


# In[68]:


write_file = tweets_df2.to_csv(index=False)


# In[71]:


with open('predictions.csv','w+') as file:
    file.write(write_file) 

file.close()

