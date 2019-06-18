# Emergency Situation Awareness

## Dataset

* https://crisisnlp.qcri.org/lrec2016/lrec2016.html
* https://crisisnlp.qcri.org/
* http://crisislex.org/

## Architecture

### Burst Detection for Unexpected Incidents

In our implementation, we used a training set of around 30 million tweets captured
between June and September 2010. We preprocessed the tweets by removing stop words
and stemming words, which resulted in a set of roughly 2.6 million distinct features,
based on which we built our background alert model. In the online phase, we devised
an alerting scheme that evaluates a sliding 5-minute window of features against
the alert model every minute.

For evaluation, we annotated roughly 2,400 features in a six-month Twitter dataset
that we collected in 2010. We define an actual burst as one feature that suddenly
occurs frequently in a time window and whose occurrence lasts more than 1 minute.
We evaluate our burst-detection module using two commonly used metrics: detection
rate and false-alarm rate. We compute the detection rate as the ratio of the number
of correctly detected bursty features to the total number of actual bursty features,
and the false-alarm rate as the ratio of the number of nonbursty features that are
incorrectly detected as bursty features to the total number of nonbursty features.

### Classification for Impact Assessment

we built statistical classifiers that automatically identify tweets containing
information about the infrastructure status, where the infrastructure includes
assets such as roads, bridges, railways, hospitals, airports, commercial and
residential buildings, water, electricity, gas, and sewerage supplies.

We experimented with two machine learning methods for tweet classification,
naive Bayes and support vector machines (SVM). To extract useful features, we
preprocessed the dataset by removing a list of stop words and tokenizing
the tweets. We then constructed lexical features and Twitter-specific features
for classification. These features include

* word unigrams;
* word bigrams;
* word length;
* the number of hashtags “#” contained in a tweet;
* the number of user mentions, “@username”;
* whether a tweet is retweeted;
* whether a tweet is replied to by other users.
  
After feature extraction, we performed experiments using a 10-fold cross-validation
over our training data.

### Online Clustering for Topic Discovery

To discover important topics from Twitter, we also developed an online incremental
clustering algorithm that automatically groups similar tweets into topic clusters,
so that each cluster corresponds to an event-specific topic. For this task, the desirable
clustering algorithm should be scalable to handle the sheer volume of incoming tweets and
not require a priori knowledge of the number of clusters, given that tweet contents are
constantly evolving over time. So, partitional clustering algorithms such as k-means and
expectation-maximization (EM) aren’t suitable for this problem, because they require the
number of clusters as input.

To capture tweets’ textual information, we represent each tweet using a vector of terms
weighted using term frequency (TF) and inverse document frequency (IDF). Given a Twitter stream in
which the tweets are sorted according to their published time, the basic idea of incremental
clustering is as follows. First, the algorithm takes the first tweet from the stream and uses it
to form a cluster. Next, for each incoming tweet, __T__, the algorithm computes its similarity with
any existing clusters. Let C be the cluster that has the maximum similarity with __T__. If `sim(T, C)`
is greater than a threshold __d__, which is to be determined empirically, tweet __T__ is added to
the cluster __C__; otherwise, a new cluster is formed based on __T__. We define the function `sim(T, C)`
to be the similarity between tweet __T__ and cluster __C__. In the clustering process, whenever a new tweet __T__
is added to a cluster __C__, the centroid of __C__ is updated as the normalized vector sum of all the tweets in __C__.
We use two similarity measures: cosine similarity and Jaccard similarity.
To take into account the temporal dimension, we add another time factor to the similarity measure that favors
a tweet to be added to the clusters whose time centroids are close to the tweet’s publication time.
We measure clustering quality using the Silhouette score, which is a metric-independent measure designed
to describe the ratio between cluster coherence and separation.
