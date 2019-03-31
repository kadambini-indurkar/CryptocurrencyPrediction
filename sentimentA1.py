# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:25:58 2019

@author: Hp
"""


import numpy as np
import pandas as pd
import twitter
from twitter import Twitter
from twitter import OAuth

import json 
from pandas.io.json import json_normalize

ck = '' #consumer key
cs = '' #consumer key secret
at = '' 
ats = ''

oauth = OAuth(at,ats,ck,cs)

api = Twitter(auth=oauth)

df = pd.DataFrame()
mid = 0
for i in range(100):
    if i ==0:
        search_tw = api.search.tweets(q="Bitcoin", count = 100)
    else:
        search_tw = api.search.tweets(q="Bitcoin", count=100, max_id=mid)
    
    dftemp = json_normalize(search_tw,'statuses')
    mid = dftemp['id'].min()
    mid=mid-1
    df = df.append(dftemp,ignore_index=True)
df.shape

tweet = df['text']
df_u = json_normalize(df['user'])
df_u.head()
df_s = df_u['screen_name']
df_s.head()
df['screenname']=df_u['screen_name']

from textblob import TextBlob as tb

pol = []
sub = []

for j in tweet:
    tx = tb(j)
    pol.append(tx.sentiment.polarity)
    sub.append(tx.sentiment.subjectivity)
df_pols = pd.DataFrame({"polarity":pol,"subjectivity":sub})
df['polarity']=df_pols['polarity']
df['subjectivity']=df_pols['subjectivity']

df_sup = pd.DataFrame()
df3 = pd.DataFrame()

df3 = df[['polarity','subjectivity','screenname']]
df3.head()


negative = pd.DataFrame()
positive= pd.DataFrame()
neutral = pd.DataFrame()

negative = df3[df3['polarity']<=-0.4]
positive = df3[df3['polarity']>=0.4]
neutral = df3[(df3['polarity'] > -0.4) & (df3['polarity'] < 0.4)]

print((len(negative)/df.shape[0])*100 ,"percent of negative tweets ")
negative.head()

print((len(positive)/df.shape[0])*100 ,"percent of positive tweets ")
positive.head()

print((len(neutral)/df.shape[0])*100 ,"percent of neutral tweets ")

neutral.head()





















