
# Bag of Words with Lemmatization


```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```
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

X_train,X_test,y_train,y_test = train_test_split(X_data,review_data.Sentiment,
                                                 test_size=0.3,stratify=review_data.Sentiment,
                                                 random_state=42)
len(X_train)
```




    254914




```
#Creating Count BOW for our dataset

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
final_counts = count_vector.fit_transform(X_train)
print(final_counts.shape)
# here final_counts is sparse representation of document
```

    (254914, 93479)



```
print("dimention of single document is :",len(count_vector.get_feature_names()))
```

    dimention of single document is : 93479


Observation : In this dataset Stemming gives less dimension as compare to Lemmatization. It may be due to overstemming.


```
# Naive Bayes Classifer
from sklearn.naive_bayes import MultinomialNB

clf =  MultinomialNB()
clf.fit(final_counts,y_train)
print(clf.score(final_counts,y_train))
X_test_bow = count_vector.transform(X_test)
print(clf.score(X_test_bow,y_test))
```

    0.9164267164612379
    0.9086590389016018



```
from sklearn.metrics import confusion_matrix

cnf = confusion_matrix(y_test,clf.predict(X_test_bow))
sns.heatmap(cnf,annot=True,fmt='g',cmap="YlGnBu");
plt.title("BOW with Lemmatization Performace");
```


![png](resources/output_11_0.png)


**Conclusion** : In this dataset both Lemmatization and Stemming gives similar performance.

Lemmatization and Stemming is also known as Text Normalization step.
