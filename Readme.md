# Sentiment Analyzer
![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg)     ![Domain Machine learning](https://img.shields.io/badge/Domain-Machine--Learning-orange.svg) ![Type NLP](https://img.shields.io/badge/Type-NLP-blue.svg)


In this repository we try to build a Sentiment Analyzer for Reviews using Amazon Fine Food Reviews dataset. we try different machine learning approch to solve our problem.
### Dataset
Our Target Machine learning problem is to create sentiment analyzer for reviews. so we need to prepare our dataset for problem. After getting data from source, It is looks like 


| id | ProductId | UserId | ProfileName | Score | HelpfulnessNumerator | HelpfulnessDenominator | Time | Summary | Text |
|----|------------|----------------|---------------------------------|-------|----------------------|------------------------|------------|-----------------------|-------------------------------------------|
|  1 | B001E4KFG0 | A3SGXH7AUHU8GW | delmartian | 5 | 1 | 1 | 1303862400 | Good Quality Dog Food | I have bought several of the Vitality... |
| 2 | B00813GRG4 | A1D87F6ZCVE5NK | dll pa | 1 | 0 | 0 | 1346976000 | Not as Advertised | Product arrived labeled as Jumbo... |
| 3 | B00813GRG4 | ABXLMWJIXXAIN | Natalia Corres | 4 | 1 | 1 | 1219017600 | "Delight" says it all | This is a confection that has been... |


### [Data preprocessing]()
we need to preprocess Data properly so that it make sense and suitable for machine learning models. so first step is Data preprocessing and we also do some feature engineering, after that our data looks like

| Text | Sentiment |
|---------------------------------------------------|-----------|
| I have to say I was a little apprehensive to b... | 1 |
| Received my free K cups as a sample promotion ... | 1 |
| Brooklyn "French Roast" K-Cup Coffee is not on... | 0 |
