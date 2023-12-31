# -*- coding: utf-8 -*-
"""biologie_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q7z-FszG9B6sVK11jdojgHPwhzhFtCPl
"""

api_key = "AIzaSyBD-gtIyQ4K4iq5--bxinO-2tCcmByXKrY"

from googleapiclient.discovery import build

youtube = build('youtube', 'v3', developerKey=api_key)

query_results = youtube.search().list( part = 'snippet', q = 'biologie', order = 'relevance',  maxResults = 20, type = 'video', relevanceLanguage = 'en', safeSearch = 'moderate', ).execute()

query_results['items'][0]

liste=[]
for i in query_results['items']:
  id_v=i['id']['videoId']
  description_v=i['snippet']['description']
  print(id_v,  " ",description_v)
  liste.append([id_v,  description_v])

import pandas as pd
df= pd.DataFrame(data=liste, columns=['id_v', 'description_v'] )
df.to_csv('da.csv')

df.head()

l=[]
for i in df.index:
  data = youtube.commentThreads().list(part='snippet', videoId=df["id_v"][i], maxResults='100', textFormat="plainText").execute()
  for j in data['items']:
    comment = j['snippet']['topLevelComment']['snippet']['textDisplay']
    l.append([df["id_v"][i], df["description_v"][i], comment])

dataf1= pd.DataFrame(data=l, columns=['id_v', 'description_v', 'comment'] )
print(dataf1)
dataf1.to_csv('df_biologie.csv')

dataf1.shape