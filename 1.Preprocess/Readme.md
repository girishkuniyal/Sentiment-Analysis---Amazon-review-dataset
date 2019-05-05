
## Preprocessing of Raw Data



```
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

```


```
review_data = pd.read_csv("Reviews.csv")
review_data.shape
```




    (568454, 10)




```
review_data.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>B001E4KFG0</td>
      <td>A3SGXH7AUHU8GW</td>
      <td>delmartian</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1303862400</td>
      <td>Good Quality Dog Food</td>
      <td>I have bought several of the Vitality canned d...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>B00813GRG4</td>
      <td>A1D87F6ZCVE5NK</td>
      <td>dll pa</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1346976000</td>
      <td>Not as Advertised</td>
      <td>Product arrived labeled as Jumbo Salted Peanut...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>B000LQOCH0</td>
      <td>ABXLMWJIXXAIN</td>
      <td>Natalia Corres "Natalia Corres"</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1219017600</td>
      <td>"Delight" says it all</td>
      <td>This is a confection that has been around a fe...</td>
    </tr>
  </tbody>
</table>
</div>




```
review_data.Score.value_counts().sort_index().plot(kind='bar');
```


![png](resources/output_4_0.png)


**Conclusion** : We have very much High rating as compare to Low rating.
so much less Low rating reviews as compare to High rating reviews.


```
#Review with Score Greater or equal to four

temp = review_data[review_data.Score >=4]
for i in range(5):
    print("Rating: ",temp.Score.iloc[i])
    print("Review: ",temp.Text.iloc[i],'\n')
```

    Rating:  5
    Review:  I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most. 
    
    Rating:  4
    Review:  This is a confection that has been around a few centuries.  It is a light, pillowy citrus gelatin with nuts - in this case Filberts. And it is cut into tiny squares and then liberally coated with powdered sugar.  And it is a tiny mouthful of heaven.  Not too chewy, and very flavorful.  I highly recommend this yummy treat.  If you are familiar with the story of C.S. Lewis' "The Lion, The Witch, and The Wardrobe" - this is the treat that seduces Edmund into selling out his Brother and Sisters to the Witch. 
    
    Rating:  5
    Review:  Great taffy at a great price.  There was a wide assortment of yummy taffy.  Delivery was very quick.  If your a taffy lover, this is a deal. 
    
    Rating:  4
    Review:  I got a wild hair for taffy and ordered this five pound bag. The taffy was all very enjoyable with many flavors: watermelon, root beer, melon, peppermint, grape, etc. My only complaint is there was a bit too much red/black licorice-flavored pieces (just not my particular favorites). Between me, my kids, and my husband, this lasted only two weeks! I would recommend this brand of taffy -- it was a delightful treat. 
    
    Rating:  5
    Review:  This saltwater taffy had great flavors and was very soft and chewy.  Each candy was individually wrapped well.  None of the candies were stuck together, which did happen in the expensive version, Fralinger's.  Would highly recommend this candy!  I served it at a beach-themed party and everyone loved it! 
    



```
#Review with Score Smaller or equal to two

temp = review_data[review_data.Score <=2]
for i in range(5):
    print("Rating: ",temp.Score.iloc[i])
    print("Review: ",temp.Text.iloc[i],'\n')
```

    Rating:  1
    Review:  Product arrived labeled as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. Not sure if this was an error or if the vendor intended to represent the product as "Jumbo". 
    
    Rating:  2
    Review:  If you are looking for the secret ingredient in Robitussin I believe I have found it.  I got this in addition to the Root Beer Extract I ordered (which was good) and made some cherry soda.  The flavor is very medicinal. 
    
    Rating:  1
    Review:  My cats have been happily eating Felidae Platinum for more than two years. I just got a new bag and the shape of the food is different. They tried the new food when I first put it in their bowls and now the bowls sit full and the kitties will not touch the food. I've noticed similar reviews related to formula changes in the past. Unfortunately, I now need to find a new food that my cats will eat. 
    
    Rating:  2
    Review:  I love eating them and they are good for watching TV and looking at movies! It is not too sweet. I like to transfer them to a zip lock baggie so they stay fresh so I can take my time eating them. 
    
    Rating:  1
    Review:  The candy is just red , No flavor . Just  plan and chewy .  I would never buy them again 
    


**Conclusion** : By looking at above sample we assume 5,4 rating reviews is mainly positive review and 1,2 is mainly negative reviews.

## Data Cleaning and wrangling


```
# Remove duplicates Reviews

review_data = review_data.sort_values(by=['UserId','ProfileName','Time','Text'])
review_data = review_data.drop_duplicates(subset=['UserId','ProfileName','Time','Text'],keep = 'first',inplace = False)
review_data.shape
```




    (393933, 10)




```
review_data.Text.values[0]
```




    "I have to say I was a little apprehensive to buy this product for the price, but I like to keep my K-Cup price under $0.50 and Sam's Club was sold out at the time, and I tried this.  The Fuhgeddaboudit is very strong, but that's how I like it.  Overall, I was impressed."




```
# creating sentiment feature

review_data['Sentiment'] = [1 if x in (4,5) else 0 if x in(1,2) else 2 for x in review_data['Score']]
review_data = review_data[review_data.Sentiment!=2] # remove neutral reviews
review_data = review_data.filter(["Text","Sentiment"])
review_data.Sentiment.value_counts().sort_index().plot(kind='bar',)
temp = review_data.Sentiment.value_counts()
print("Negative Sentiment Percentage in dataset : ",round((temp[0]/temp.sum())*100,2),'%')
print("Postive Sentiment Percentage in dataset : ",round((temp[1]/temp.sum())*100,2),'%')
print('data Shape',review_data.shape)


```

    Negative Sentiment Percentage in dataset :  15.68 %
    Postive Sentiment Percentage in dataset :  84.32 %
    data Shape (364164, 2)



![png](resources/output_12_1.png)



```
#Save preprocessed data
review_data.to_pickle("review_data.pkl")  # where to save it, usually as a .pkl
# review_data = pd.read_pickle(review_data.pkl) #load save .pkl file
```
