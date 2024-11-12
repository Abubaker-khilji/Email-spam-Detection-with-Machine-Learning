# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
import string
from collections import Counter

# %%
file = pd.read_csv("dataset\spam.csv", encoding='ISO-8859-1')

# %%
file.head()

# %%
file.shape

# %% [markdown]
# Steps to be performed
# 1. Data Cleaning
# 2. EDA 
# 3. text Preprocessing 
# 4. Model building 
# 5. Evaluation
# 6. Improvement
# 7. Website building
# 8. Deployement
# 

# %% [markdown]
# ## 1.Data cleaning

# %%
file.info( )

# %% [markdown]
# we will drop last three columns because of low value

# %%
file.drop(columns=["Unnamed: 2","Unnamed: 3", "Unnamed: 4"] ,inplace=True)

# %%
file.sample(5)

# %%
 #renamr the cols
file.rename(columns={"v1":"target","v2":"text"},inplace=True)
file.sample(5)

# %%
encoder = LabelEncoder()

# %%
#converting text of ham and spam into 0 and 1
file["target"]= encoder.fit_transform(file['target'])

# %%
file.head()

# %%
#check missing values
file.isnull().sum()

# %%
#check for duplicated values
file.duplicated().sum()

# %%
#droping duplicated values
file=file.drop_duplicates(keep="first")

# %%
file.shape

# %% [markdown]
# ## 2. EDA 

# %% [markdown]
# First we need to check the percentage sms are ham or spam 

# %%
file["target"].value_counts()

# %%
plt.pie(file["target"].value_counts(), labels=['ham', 'spam'],autopct='%0.2f')

# %% [markdown]
# Above from pie we can conclude our data is imbalance 

# %%
nltk.download('punkt')
nltk.download('punkt_tab')


# %%
#counting number of characters
file['num_characters'] = file['text'].apply(len)


# %%
file.head()

# %%
file['num_words']=file['text'].apply(lambda x:len(nltk.word_tokenize(x)))

# %%
file.head()

# %%
file['num_sentences']=file['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

# %%
file.head()

# %%
file[['num_characters','num_words','num_sentences']].describe()

# %%
#output for ham messages
file[file['target'] == 0][['num_characters','num_words','num_sentences']].describe()

# %%
#output for spam messages
file[file['target'] == 1][['num_characters','num_words','num_sentences']].describe()

# %% [markdown]
# ## Plotting 

# %%
plt.figure(figsize=(12,6))
sns.histplot(file[file['target'] == 0]['num_characters'])
sns.histplot(file[file['target'] == 1]['num_characters'],color='red')

# %%
plt.figure(figsize=(12,6))
sns.histplot(file[file['target'] == 0]['num_words'])
sns.histplot(file[file['target'] == 1]['num_words'],color='red')

# %%
plt.figure(figsize=(12,6))
sns.histplot(file[file['target'] == 0]['num_sentences'])
sns.histplot(file[file['target'] == 1]['num_sentences'],color='red')

# %%
#we need to  understand coorelation between coloumns
sns.pairplot(file,hue='target')

# %%
sns.heatmap(file.corr(numeric_only=True),annot=True)

# %% [markdown]
# ## 3. Data Preprocessing
# * Lower case
# * Tokenization
# * Remove special characters
# * Remove stop words and punctuation
# * Stemming

# %%
ps= PorterStemmer()


# %%
def transfrom_text(text):
    text=text.lower()
    text= nltk.word_tokenize(text)

    y =[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i) 

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i) )             
    
    return " ".join(y)

# %%
transfrom_text('hi how are  %% you abubaker did you like machine learning!!')

# %%
file['transformed_text'] = file['text'].apply(transfrom_text) 

# %%
file.head()

# %%
wc = WordCloud(width=500,height=500,min_font_size=10,background_color="white")

# %%
spam_wc = wc.generate(file[file["target"]==1]['transformed_text'].str.cat(sep=" "))

# %%
plt.figure(figsize=(10,5))
plt.imshow(spam_wc)

# %%
ham_wc = wc.generate(file[file["target"]==0]['transformed_text'].str.cat(sep=" "))

# %%
plt.figure(figsize=(10,5))
plt.imshow(spam_wc)

# %%
file.head()

# %%
spam_corpus = []
for msg in file[file["target"]==1]['transformed_text'].to_list():
    for words in msg.split():
        spam_corpus.append(words)

# %%
len(spam_corpus)

# %%
#sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
#plt.xticks(rotation='vertical')
#plt.show()

# %% [markdown]
# ## Model Building
# # we start builidng with naive bayes mode

# %%


# %%


# %%



