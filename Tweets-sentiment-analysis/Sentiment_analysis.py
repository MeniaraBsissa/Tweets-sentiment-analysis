#!/usr/bin/env python
# coding: utf-8

# In[795]:


#imporation de data
import pandas as pd
df1 = pd.read_csv('df_technologie.csv')
df2=pd.read_csv('df_sport.csv')
df3=pd.read_csv('df_biologie.csv')


# In[796]:


df1.head(1)


# In[797]:


df1.drop(['Unnamed: 0', 'id_v', 'comment' ], axis=1, inplace=True)
df1['target']='tech'
df1.head(1)


# In[798]:


df2.head(1)


# In[799]:


df2.drop(['Unnamed: 0', 'id_v', 'comment' ], axis=1, inplace=True)
df2['target']='sport'
df2.head(1)


# In[800]:


df3.head(1)


# In[801]:


df3.drop(['Unnamed: 0', 'id_v', 'comment' ], axis=1, inplace=True)
df3['target']='biologie'
df3.head(1)


# In[802]:


data=pd.concat([df1, df2, df3], axis=0)


# In[803]:


data.shape


# In[804]:


data.head(1500)


# In[805]:


data.to_csv('data.csv')


# In[806]:


# vusialisation de classes
fig, ax = plt.subplots()
fig.suptitle("target", fontsize=12)
data["target"].reset_index().groupby("target").count().sort_values(by= 
       "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
plt.show()


# In[807]:


#nombre d'instances dans chaque classe 
data['target'].value_counts()


# In[808]:


data.head(1500)


# In[817]:


#changer le type de colonne  'description_v'
import numpy
data['description_v'].astype(numpy.str) 


# In[818]:


#etude exploratoire 1
text = " ".join(review for review in data.description_v)
words = nltk.tokenize.word_tokenize(text)
print ("nbre total de mots : ",len (words))
word_dist = nltk.FreqDist(words)
print ("vocabulaire :", len (word_dist))


# In[819]:


#etude exploratoire 2
import matplotlib.pyplot as plt
from seaborn import distplot
from wordcloud import WordCloud
wordcloud = WordCloud().generate_from_frequencies(word_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()


# In[820]:


# 
def to_lower(input_text: str) -> str:
    return input_text.lower()


# In[821]:


data['description_v']= data['description_v'].apply(to_lower)


# In[822]:


text = " ".join(review for review in data.description_v)
words = nltk.tokenize.word_tokenize(text)
print ("nbre total de mots : ",len (words))
word_dist = nltk.FreqDist(words)
print ("vocabulaire :", len (word_dist))
wordcloud = WordCloud().generate_from_frequencies(word_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()


# In[823]:


#remove les url
def remove_url(input_text: str) -> str:
    
    return re.sub('(www|http)\S+', '', input_text)


# In[824]:


data['description_v']= data['description_v'].apply(remove_url)


# In[825]:


text = " ".join(review for review in data.description_v)
words = nltk.tokenize.word_tokenize(text)
print ("nbre total de mots : ",len (words))
word_dist = nltk.FreqDist(words)
print ("vocabulaire :", len (word_dist))
wordcloud = WordCloud().generate_from_frequencies(word_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()


# In[826]:


#remove les ponctuation
def remove_punctuation(input_text: str) -> str:
     
    punctuations = string.punctuation
    processed_text = input_text.translate(str.maketrans('', '', punctuations))
    return processed_text


# In[827]:


data['description_v']= data['description_v'].apply(remove_punctuation)


# In[828]:


text = " ".join(review for review in data.description_v)
words = nltk.tokenize.word_tokenize(text)
print ("nbre total de mots : ",len (words))
word_dist = nltk.FreqDist(words)
print ("vocabulaire :", len (word_dist))
wordcloud = WordCloud().generate_from_frequencies(word_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()


# In[829]:


#reùove les chractères speciaux 
def remove_special_character(input_text: str) -> str:

   special_characters = 'å¼«¥ª°©ð±§µæ¹¢³¿®ä£'
   processed_text = input_text.translate(str.maketrans('', '', special_characters))
   return processed_text


# In[830]:


data['description_v']= data['description_v'].apply(remove_special_character)


# In[831]:


text = " ".join(review for review in data.description_v)
words = nltk.tokenize.word_tokenize(text)
print ("nbre total de mots : ",len (words))
word_dist = nltk.FreqDist(words)
print ("vocabulaire :", len (word_dist))
wordcloud = WordCloud().generate_from_frequencies(word_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()


# In[833]:


#remove les chiffres
def remove_number(text):
    result = ''.join([i for i in text if not i.isdigit()])
    return result


# In[834]:


data['description_v']= data['description_v'].apply(remove_number)


# In[835]:


#remove les stop words
from nltk.corpus import stopwords
def remove_stop_en(text):
    no_stop=" ".join([c for c in text.split() if c not in stopwords.words('english')])
    return no_stop


# In[836]:


data['description_v']= data['description_v'].apply(remove_stop_en)


# In[837]:


text = " ".join(review for review in data.description_v)
words = nltk.tokenize.word_tokenize(text)
print ("nbre total de mots : ",len (words))
word_dist = nltk.FreqDist(words)
print ("vocabulaire :", len (word_dist))
wordcloud = WordCloud().generate_from_frequencies(word_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()


# In[838]:


data.head(1500)


# In[842]:


data.head(1500)


# In[ ]:





# In[858]:


# split le data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['description_v'], data['target'], test_size=0.3, random_state=0)


# In[859]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()


# In[860]:


train = vectorizer.fit(X_train)
train = vectorizer.transform(X_train)


# In[861]:


from sklearn import feature_selection


# In[862]:


print(train.toarray())


# In[863]:


pd.DataFrame(train.toarray() , columns = vectorizer.get_feature_names())


# In[864]:


from sklearn.naive_bayes import MultinomialNB
model_nb = MultinomialNB()
model_nb.fit(train, y_train)


# In[865]:


test = vectorizer.transform(X_test)
from  sklearn.metrics  import accuracy_score
y_pred_nb = model_nb.predict(test)
print(accuracy_score(y_test,y_pred_nb))


# In[866]:


from sklearn.metrics import  confusion_matrix
conf_nb = confusion_matrix(y_test, y_pred_nb)
conf_nb


# In[867]:


## Plot confusion matrix
import seaborn as sns
cm = metrics.confusion_matrix(y_test, y_pred_nb)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, 
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0)


# In[868]:


## Accuracy, Precision, Recall
import sklearn.metrics as metrics
predicted_prob = model_nb.predict_proba(test)
accuracy = metrics.accuracy_score(y_test, y_pred_nb)
auc = metrics.roc_auc_score(y_test, predicted_prob, 
                            multi_class="ovr")
print("Accuracy:",  round(accuracy,2))

print("Detail:")
print(metrics.classification_report(y_test, y_pred_nb))


# In[869]:


from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier()
DT_model.fit(train, y_train)


# In[870]:


y_pred_dt = DT_model.predict(test)
print(accuracy_score(y_test,y_pred_dt))


# In[856]:


import pickle
pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
vectorizer = pickle.load(open("vectorizer.pickle", "rb")) 


# In[857]:


import pickle
pickle.dump(vectorizer, open("model_nb.pickle", "wb"))
vectorizer = pickle.load(open("model_nb.pickle", "rb")) 


# In[874]:


model_nb.predict(vectorizer.transform(['coupe du monde']))


# In[875]:


model_nb.predict(vectorizer.transform(['machine learning']))


# In[ ]:




