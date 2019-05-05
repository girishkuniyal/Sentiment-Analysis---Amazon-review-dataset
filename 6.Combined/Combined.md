
# Combined Approch
## Bag of words with Lemmatization and Bi-gram features


```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```
# load preprocessed text
review_data = pd.read_pickle("review_data.pkl")
```


```
# nltk installation

!pip install nltk

import nltk
nltk.download('wordnet')
```


```
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
```


```
def tokenize_lemmatize(review_data):
  """Takes input as pandas data frame with field name Text and return tokenize 
  and lemmatized list"""
  X_data = []
  for i in range(review_data.shape[0]):
      token = word_tokenize(review_data.Text.iloc[i])
      for i in range(len(token)):
          token[i] = lemmatizer.lemmatize(token[i])
      token = ' '.join(token)
      X_data.append(token)
  return X_data
 
X_data = tokenize_lemmatize(review_data)

```


```
# Stratified Test Train Spilt

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_data,review_data.Sentiment,test_size=0.3,
                                                 stratify=review_data.Sentiment,random_state=42)
len(X_train)
```




    254914




```
#Creating BOW with Bi gram for our dataset

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(ngram_range=(1,2))
final_counts = count_vector.fit_transform(X_train)
print(final_counts.shape)
# here final_counts is sparse representation of document
```

    (254914, 2193130)



```
print("dimention of single document is :",len(count_vector.get_feature_names()))
```

    dimention of single document is : 2193130



```
# Naive Bayes Classifer
from sklearn.naive_bayes import MultinomialNB

clf =  MultinomialNB()
clf.fit(final_counts,y_train)
print(clf.score(final_counts,y_train))
X_test_bow = count_vector.transform(X_test)
print(clf.score(X_test_bow,y_test))
```

    0.9593470739151243
    0.9121189931350114



```
from sklearn.metrics import confusion_matrix

cnf = confusion_matrix(y_test,clf.predict(X_test_bow))
sns.heatmap(cnf,annot=True,fmt='g',cmap="YlGnBu");
plt.title("BOW(count)with Lemmatization and Bi-gram Performace");
```


![png](resources/output_10_0.png)


## Binary Bag of Words with lemmatization and Bi-gram features


```
#Creating Binary BOW with Lemmatization and Bi-gram features

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(ngram_range=(1,2),binary=True)
final_counts = count_vector.fit_transform(X_train)
print(final_counts.shape)
# here final_counts is sparse representation of document
```

    (254914, 2193130)



```
print("dimention of single document is :",len(count_vector.get_feature_names()))
```

    dimention of single document is : 2193130



```
# Naive Bayes Classifer
from sklearn.naive_bayes import MultinomialNB

clf =  MultinomialNB()
clf.fit(final_counts,y_train)
print(clf.score(final_counts,y_train))
X_test_bow = count_vector.transform(X_test)
print(clf.score(X_test_bow,y_test))
```

    0.9590960088500435
    0.9082288329519451



```
from sklearn.metrics import confusion_matrix

cnf = confusion_matrix(y_test,clf.predict(X_test_bow))
sns.heatmap(cnf,annot=True,fmt='g',cmap="YlGnBu");
plt.title("BOW(binary) with Lemmatization and Bi-gram Performace");
```


![png](resources/output_15_0.png)


