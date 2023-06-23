
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
import os
import string
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score, confusion_matrix, classification_report

df=pd.read_csv("Tweets_Airline.csv", encoding = 'utf8')

df.head()

df.tail()

def exploration(df):
    qnt=0
    quant=[]
    qlt=0
    qualt=[]
    print("=================================================================================")
    type=[]
    print("\n\t\t\tEXAMEN DE LA STRUCTURE DU DATAFRAME\n")
    j="*"*80
    k="-"*80
    row, col=df.shape
    print(j)
    print(f"La dimension de notre dataframe est : {df.shape}")
    print(j)
    print(f"Le nombre d'observation est : {row}")
    print(j)
    print(f"Le nombre de variable est : {col}")
    print(j)
    print("Le nom des variables et leur type de notre dataframe est :\n")
    ''''for x in df.columns:
        print(k)
        print(f" La variable {x} a pour type {df[x].dtypes}")
        type=type+[df[x].dtypes]'''
    df.info()
    print(f"\n{j}")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            qlt += 1
            qualt.append(col)
            
        else:
            qnt += 1
            quant.append(col)
           
    print("\n")
    print("Sur les 17 variables, {} sont quantitatives {} et {} sont qualitatives {}".format(qnt,quant, qlt, qualt))
    print("=================================================================================") 

exploration(df)


def detectValeurManquante(df):
    print(f"\nLe nombre de valeurs manquantes")
    print("*"*60)
    print(df.isna().sum())
    ##print("*"*30)
    print(f"\nLe pourcentage de valeurs manquantes")
    print("*"*60)
    print((df.isna().sum()/df.shape[0])*100)


detectValeurManquante(df)


df.airline_sentiment.value_counts()

df.airline_sentiment.describe().T

def DiagrammeCirculaire(df,vars):    
    #fig = plt.figure(figsize=(15,20))
    #_,axes = plt.subplots(1, k, sharey=True, figsize=(40,10))
    i=441
    for x in vars:
        df1_effectif=df[x].value_counts()
        j=len(df[x].unique())
        explode = [0.03]*j
        df1_effectif.plot(kind="pie",stacked=True, figsize=(10,8), shadow=True, autopct='%1.1f%%', explode=explode)
        plt.legend(loc="lower left",bbox_to_anchor=(0.8,1.0))
        plt.show()

DiagrammeCirculaire(df,['airline_sentiment'])

sn.countplot(x = df['airline_sentiment'],palette='plasma')
plt.xticks(rotation = 'horizontal')

df_Pre=df[['airline_sentiment','text']]

df_Pre.head()

df_Pre['text_len'] = [len(t) for t in df_Pre.text]

df_Pre

def preprocess(text):
    
    #Definition du dictionnaire de emojis et leur signification
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
              ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
              ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
              ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
              ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

    stopwordlist = ['Utilisateur']

        
    # Definition des regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    # alphaPattern      = "[^a-zA-Z0-9]"
    alphaPattern      = "[^a-zA-Z]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    text = text.lower()
        
    # Remplacement de tous les URls par 'URL'
    text = re.sub(urlPattern,' URL',text)
        
    # Remplacement des emojis.
    for emoji in emojis.keys():
        text = text.replace(emoji, "EMOJI" + emojis[emoji])    
            
    # Remplacement des @USERNAME to 'Utilisateur'.
    text = re.sub(userPattern,' Utilisateur', text)  
        
    
    text = re.sub(alphaPattern, " ", text)
        
    # Replacement de 3 ou plus de lettres consecutives par 2 .
    text = re.sub(sequencePattern, seqReplacePattern, text)
        
    #Suppression des ponctuations
    all_char_list = []
    all_char_list = [char for char in text if char not in string.punctuation]
    text = ''.join(all_char_list)
    
    #Retrait du mot 'Utilisateur'
    tweetwords = ''
    for word in text.split():
        if word not in (stopwordlist):
            if len(word)>1:
                tweetwords += (word+' ')
        
    return tweetwords


# In[ ]:





# In[22]:


df_Pre['Text_clean'] = df_Pre['text'].apply(lambda x: preprocess(x))


# In[ ]:





# <ul style="font-family: times, serif; font-size:14pt; color:red;">
#     
# __c.Suppression des mots vides__
#     
# </ul>

# In[23]:


nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# removing the stopwords

def removestopwords(text):
    
    removedstopword = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopword)


# In[ ]:





# In[24]:


df_Pre['Text_clean'] = df_Pre['Text_clean'].apply(lambda text:removestopwords(text))


# In[ ]:





# <ul style="font-family: times, serif; font-size:14pt; color:red;">
#     
# __c.Lemmatisation__
#     
# </ul>

# In[25]:


import nltk
from nltk.stem import WordNetLemmatizer

lemma=WordNetLemmatizer()

def lematizing(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


# In[26]:


df_Pre['Text_clean'] = df_Pre['Text_clean'].apply(lambda text:lematizing(text))


# In[ ]:





# ## <a name="p11">5.Etude Comparative</a>

# In[ ]:






def showmostfrequentwords(text,no_of_words):
    
    allwords = ' '.join([char for char in text])
    allwords = allwords.split()
    fdist = nltk.FreqDist(allwords)
    
    wordsdf = pd.DataFrame({'word':list(fdist.keys()),'count':list(fdist.values())})
    
    df = wordsdf.nlargest(columns="count",n = no_of_words)
    
    plt.figure(figsize=(7,5))
    ax = sn.barplot(data=df,x = 'count',y = 'word')
    ax.set(ylabel = 'Word')
    plt.show()
    
    return wordsdf


# In[ ]:





# In[31]:


motsdf = showmostfrequentwords(df_Pre['Text_clean'],25)


# In[ ]:





# In[32]:


motsdf.sort_values('count',ascending=False).head(10).style.background_gradient(cmap = 'plasma')


# In[ ]:





# <ul style="font-family: times, serif; font-size:14pt; color:darkgreen;">
#     
# __Balancement des donnees__
#     
# </ul>

# In[ ]:





# In[33]:


df_grouped_by = df_Pre.groupby(['airline_sentiment'])
df_balanced = df_grouped_by.apply(lambda x: x.sample(df_grouped_by.size().min()).reset_index(drop=True))
df_balanced = df_balanced.droplevel(['airline_sentiment'])
df_balanced


# In[34]:


df_balanced.airline_sentiment.value_counts()


# **Nos donnees sont maintenant equilibrees**

# ## <a name="p10">6.Encodage</a>

# In[35]:


sentiment = list(df_balanced['airline_sentiment'].unique())
encode = [i for i in range(len(sentiment))]
mapper = dict(zip(sentiment,encode))
print(mapper)


# In[ ]:





# ## <a name="p4">7.Decoupage du jeu de donnees</a>

# In[36]:


le = LabelEncoder()
y=le.fit_transform(df_balanced['airline_sentiment'])

X_train,X_test,y_train,y_test = train_test_split(df_balanced['Text_clean']
                                                ,y,test_size=0.2,
                                                random_state=557)

X_train.shape,X_test.shape


# ## <a name="p8">8.TF-IDF Vectoriser</a>

# Le **TF-IDF** pour **« Term Frequency - Inverse Document Frequency »** est une méthode de calcul générée par un algorithme de moteur de recherche et qui permet de déterminer la pertinence d'un document ou d'un site internet par rapport à un terme. Pour atteindre son objectif, cette formule mathématique prend en compte deux facteurs principaux : la fréquence du terme étudié dans le texte (TF) et le nombre de documents contenant ce terme (IDF).
# 
# **TF-IDF Vectoriser** convertit une collection de documents bruts en une **matrice de fonctionnalités TF-IDF**. Le **Vectoriser** est généralement entraîné uniquement sur le jeu de données **train**.
# 
# **max_features** spécifie le nombre de fonctionnalités à prendre en compte.

# In[37]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
xtrain_tfidf = tfidf_vectorizer.fit_transform(X_train.values.astype('U'))
xtest_tfidf = tfidf_vectorizer.transform(X_test.values.astype('U'))


# In[ ]:





# In[38]:


def print_metrics(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[ ]:





# In[39]:


def ApprentissageModele(models, x_train, y_train, x_test, y_test):
    for x in models:
        print(f"\nCLASSIFICATION {x}\n")
        model=x
        model.fit(x_train,y_train)
        print_metrics(model, x_train, y_train, x_test, y_test, train=True)
        print_metrics(model, x_train, y_train, x_test, y_test, train=False)


# In[ ]:





# In[40]:


def ApprentissageMetrics(models, x_train, y_train, x_test, y_test):
    results_df = pd.DataFrame()
    result_df = pd.DataFrame()
    for x in models:
        model=x
        model.fit(x_train,y_train)
        #print_metrics(model, x_train, y_train, x_test, y_test, train=True)
        #print_metrics(model, x_train, y_train, x_test, y_test, train=False)
        test_score = accuracy_score(y_test, model.predict(x_test)) * 100
        train_score = accuracy_score(y_train, model.predict(x_train)) * 100
        train_recall = recall_score(y_train, model.predict(x_train),average='macro') * 100
        train_precision = precision_score(y_train, model.predict(x_train),average='macro') * 100
        train_f1 = f1_score(y_train, model.predict(x_train),average='macro') * 100
        test_recall = recall_score(y_test, model.predict(x_test),average='macro') * 100
        test_precision = precision_score(y_test, model.predict(x_test),average='macro') * 100
        test_f1 = f1_score(y_test, model.predict(x_test),average='macro') * 100
        results_df_test = pd.DataFrame(data=[[str(x), test_score, test_recall, test_precision, test_f1]], 
                              columns=['Model Test', 'Accuracy %', 'Recall %','Precision %','f1 %'])
        results_df = results_df.append(results_df_test, ignore_index=True)
        results_df_training = pd.DataFrame(data=[[str(x), train_score, train_recall, train_precision, train_f1]], 
                              columns=['Model Training', 'Accuracy %', 'Recall %','Precision %','f1 %'])
        result_df = result_df.append(results_df_training, ignore_index=True)
    display(result_df)
    display(results_df)


# In[ ]:





# In[41]:


models=[SVC(),MultinomialNB(),RandomForestClassifier()]


# In[ ]:





# In[42]:


ApprentissageModele(models, xtrain_tfidf, y_train, xtest_tfidf, y_test)


# In[ ]:





# In[43]:


ApprentissageMetrics(models, xtrain_tfidf, y_train, xtest_tfidf, y_test)


# In[ ]:





# In[44]:


def createModel(predictors, target, model):
    model = model
    model.fit(predictors, np.ravel(target))
    return model


# In[ ]:





# In[45]:


model= createModel(xtrain_tfidf, y_train, RandomForestClassifier())


# In[ ]:





# In[46]:


models=[model]
for model in models:
    y_pred= model.predict(xtest_tfidf)
    accuracy=metrics.accuracy_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred,average='macro')
    precision_score = metrics.precision_score(y_test, y_pred, average='macro')
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')


# In[ ]:





# In[47]:


with open("Metric_Classification.txt", 'w') as outfile:
        outfile.write("Metriques du modele {}: ".format(model))
        outfile.write("\n")
        outfile.write("Accuracy:  {0:2.1f} \n".format(accuracy*100))
        outfile.write("Recall: {0:2.1f} \n".format(recall_score*100))
        outfile.write("f1_score: {0:2.1f} \n".format(f1_score*100))
        outfile.write("Precision: {0:2.1f} \n".format(precision_score*100))
        outfile.write("La matrice de confusion {}: \n".format(confusion_matrix(y_test, y_pred)))


# In[ ]:




