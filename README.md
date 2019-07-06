# Emergency Situation Awareness

## Dataset

* <https://crisisnlp.qcri.org/lrec2016/lrec2016.html>
* <https://crisisnlp.qcri.org/>
* <http://crisislex.org/>

## Architecture

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

## Results

### Keras

**540k tweets:**
Crisis nlp pre-trained

Dev set: 0.98

* Number of crisis tweets: 6000
* Number of non-crisis tweets: 6000

```
Classification report : 
               precision    recall  f1-score   support

      normal       0.83      0.98      0.90      6000
      crisis       0.98      0.80      0.88      6000

    accuracy                           0.89     12000
   macro avg       0.90      0.89      0.89     12000
weighted avg       0.90      0.89      0.89     12000
```

**1759077 tweets:** Crisis nlp pre-trained

Dev set: 0.97

* Number of crisis tweets: 32000
* Number of non-crisis tweets: 32000

```
Classification report : 
               precision    recall  f1-score   support

      normal       0.92      0.97      0.95     32000
      crisis       0.97      0.92      0.94     32000

    accuracy                           0.94     64000
   macro avg       0.95      0.94      0.94     64000
weighted avg       0.95      0.94      0.94     64000
```

### Bayes
* Number of crisis tweets: 270000
* Number of non-crisis tweets: 270000

```
Cross Validation
Accuracy: 0.98
Precision: 0.98
Recall: 0.97
F1 score: 0.98
```

Evaluate Bayes

* Number of crisis tweets: 32000
* Number of non-crisis tweets: 32000

Accuracy Score: 0.96

```
Classification report:

               precision    recall  f1-score   support
      normal       0.94      0.98      0.96     32000
      crisis       0.98      0.94      0.96     32000

    accuracy                           0.96     64000
   macro avg       0.96      0.96      0.96     64000
weighted avg       0.96      0.96      0.96     64000
```