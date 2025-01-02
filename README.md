# Hate Speech Detection on Turkish Tweets using Cross Validation

Data analysis on offensive language use in Turkish tweets with the duration of a period of 18 months. 

# Dataset
Dataset is collected and curated by [2] for their article and includes 35284 entries and 4 attributes in Turkish language from April 2018
from September 2019. It has both offensive and non-offensive tweets,and after annotated by volunteers who are Turkish native speakers,
offensive tweets are divided into two sub-branches targeted and non-targeted. Non-targeted tweets are labeled as prof. Targeted
tweets also are sub-categorized based on the target of offensivelanguage: group, individual and other. A piechart to depict the label
in the dataset is below:

![pie2](https://github.com/user-attachments/assets/cf334fba-93a2-400a-b991-0c55783fdcac)

# Classification of Tweets
With this dataset, I intended to apply classification algorithms to correctly categorize tweets according to their labels. To accomplish
that, I initially extracted numerical representation of tweets. For that purpose, I applied tf-idf technique using sklearn library in Python.
After constructing numerical vectors for each tweet, I chose logistic regression algorithm to classify tweets using the above-mentioned
numerical vectors as input. Again, for the classification, I used sklearn library.

# Research Questions & Statistical Analysis

Question I. Is number of words used in tweets with non-offensive and offensive label similar?
Hypothesis: Number of words(length) used in non-offensive and offensive tweets is similar.

First, I conducted the statistical test for offensive and non-offensive words. I used Shapiro-Wilk Normality Test to check distribution of
word counts for non-offensive tweets. Word count for non-offensive distribution comes out to be not normal. Because word count for
non-offensive tweets do not follow normal distribution, I did not check the normality of offensive tweets. Since I have independent,
not-normal and unpaired samples, I conducted Mann-Whitney U

Test with number of words in tweets for non-offensive and offensive tweets in Turkish corpus. The result is 18616058.5 and 0.00
for stat and p values respectively which rejects the hypothesis. In other words, tweets with offensive and non-offensive language
have probably different length distributions. In order to demonstrate the length of non-offensive and offensive
tweets, a density plot is depicted in Figure 3 as follows:

![density](https://github.com/user-attachments/assets/5d473d4e-4697-47e6-b671-2697d4c14f10)

Question II. Is monthly percentage of tweets with non-offensive and offensive language close?
Hypothesis: Monthly percentage of tweets with non-offensive and offensive language is close.

There is a change in the behavior of volume for offensive tweets during the municipal
election occurred in May 2019. Therefore, I wanted to statistically prove that claim with my research question.

I used Shapiro-Wilk Normality Test to check whether the distribution of tweet volumes is normal for non-offensive language. The
result comes out to be not normal. I did not check the normality of offensive tweets. Since I have independent, not-normal and pairedsamples,
I conducted Mann-Whitney U Test with percentage of tweets for non-offensive and offensive language in Turkish corpus.
The result is 179.0 and 0.488 for stat and p values respectively which cannot reject our initial hypothesis. In other words, distribution
of monthly percentage of tweets with non-offensive and offensive language is indeed close to each other.
In order to demonstrate the similarity of such distribution, a plot is depicted in Figure 4 as follows:

![matplot](https://github.com/user-attachments/assets/c0acef40-7411-4cbb-9fab-0c32f8c234c3)

# Classification
# Cross Validation

In order to measure the performance of my model, I applied cross validation technique with five-folds. After splitting all data into
folds of same size, I used four folds for training and use the remaining fold for testing. To implement cross-fold validation, I used
sklearn library in Python.
