#!/usr/bin/env python
# coding: utf-8

#imporation de data
import pandas as pd
df1 = pd.read_csv('df_technologie.csv')
df2=pd.read_csv('df_sport.csv')
df3=pd.read_csv('df_biologie.csv')

df1.head(1)
df1.drop(['Unnamed: 0', 'id_v', 'comment' ], axis=1, inplace=True)
df1['target']='tech'
df1.head(1)

df2.head(1)
2.drop(['Unnamed: 0', 'id_v', 'comment' ], axis=1, inplace=True)
df2['target']='sport'
df2.head(1)


df3.head(1)
df3.drop(['Unnamed: 0', 'id_v', 'comment' ], axis=1, inplace=True)
df3['target']='biologie'
df3.head(1)

data=pd.concat([df1, df2, df3], axis=0)


data.shape


data.head(1500)


data.to_csv('data.csv')

# vusialisation de classes
fig, ax = plt.subplots()
fig.suptitle("target", fontsize=12)
data["target"].reset_index().groupby("target").count().sort_values(by= 
       "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
plt.show()


#nombre d'instances dans chaque classe 
data['target'].value_counts()


data.head(1500)


#changer le type de colonne  'description_v'
import numpy
data['description_v'].astype(numpy.str) 


#etude exploratoire 1
text = " ".join(review for review in data.description_v)
words = nltk.tokenize.word_tokenize(text)
print ("nbre total de mots : ",len (words))
word_dist = nltk.FreqDist(words)
print ("vocabulaire :", len (word_dist))


#etude exploratoire 2
import matplotlib.pyplot as plt
from seaborn import distplot
from wordcloud import WordCloud
wordcloud = WordCloud().generate_from_frequencies(word_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()



def to_lower(input_text: str) -> str:
    return input_text.lower()


data['description_v']= data['description_v'].apply(to_lower)


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


#remove les url
def remove_url(input_text: str) -> str:
    
    return re.sub('(www|http)\S+', '', input_text)


data['description_v']= data['description_v'].apply(remove_url)


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


#remove les ponctuation
def remove_punctuation(input_text: str) -> str:
     
    punctuations = string.punctuation
    processed_text = input_text.translate(str.maketrans('', '', punctuations))
    return processed_text


data['description_v']= data['description_v'].apply(remove_punctuation)


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


#reùove les chractères speciaux 
def remove_special_character(input_text: str) -> str:

   special_characters = 'å¼«¥ª°©ð±§µæ¹¢³¿®ä£'
   processed_text = input_text.translate(str.maketrans('', '', special_characters))
   return processed_text


data['description_v']= data['description_v'].apply(remove_special_character)


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


#remove les chiffres
def remove_number(text):
    result = ''.join([i for i in text if not i.isdigit()])
    return result

data['description_v']= data['description_v'].apply(remove_number)

#remove les stop words
from nltk.corpus import stopwords
def remove_stop_en(text):
    no_stop=" ".join([c for c in text.split() if c not in stopwords.words('english')])
    return no_stop


data['description_v']= data['description_v'].apply(remove_stop_en)


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


data.head(1500)


data.head(1500)


# split le data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['description_v'], data['target'], test_size=0.3, random_state=0)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()


train = vectorizer.fit(X_train)
train = vectorizer.transform(X_train)


from sklearn import feature_selection

print(train.toarray())


pd.DataFrame(train.toarray() , columns = vectorizer.get_feature_names())


from sklearn.naive_bayes import MultinomialNB
model_nb = MultinomialNB()
model_nb.fit(train, y_train)


test = vectorizer.transform(X_test)
from  sklearn.metrics  import accuracy_score
y_pred_nb = model_nb.predict(test)
print(accuracy_score(y_test,y_pred_nb))


from sklearn.metrics import  confusion_matrix
conf_nb = confusion_matrix(y_test, y_pred_nb)
conf_nb

## Plot confusion matrix
import seaborn as sns
cm = metrics.confusion_matrix(y_test, y_pred_nb)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, 
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0)


## Accuracy, Precision, Recall
import sklearn.metrics as metrics
predicted_prob = model_nb.predict_proba(test)
accuracy = metrics.accuracy_score(y_test, y_pred_nb)
auc = metrics.roc_auc_score(y_test, predicted_prob, 
                            multi_class="ovr")
print("Accuracy:",  round(accuracy,2))

print("Detail:")
print(metrics.classification_report(y_test, y_pred_nb))


from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier()
DT_model.fit(train, y_train)


y_pred_dt = DT_model.predict(test)
print(accuracy_score(y_test,y_pred_dt))


import pickle
pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
vectorizer = pickle.load(open("vectorizer.pickle", "rb")) 


import pickle
pickle.dump(vectorizer, open("model_nb.pickle", "wb"))
vectorizer = pickle.load(open("model_nb.pickle", "rb")) 


model_nb.predict(vectorizer.transform(['coupe du monde']))


model_nb.predict(vectorizer.transform(['machine learning']))


# In[ ]:




