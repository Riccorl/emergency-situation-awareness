import time

import gensim
from sklearn.model_selection import train_test_split

import config
import models
import preprocess
import utils
from sequence import TextSequence


def process():
    print("Loading tweets...")
    features, labels = utils.read_datasets()
    features = preprocess.clear_tweets(features)
    tweets_tr, tweets_dev, labels_tr, labels_dev = train_test_split(
        features, labels, test_size=0.10
    )
    print("Loading pre-trained embeddings...")
    # load the w2v matrix with genism
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        config.CRISIS_PRE_TRAINED, binary=True
    )
    # build the vocab from the w2v model
    w2v_vocab = preprocess.vocab_from_w2v(w2v)
    print("Word2Vec model vocab len:", len(w2v_vocab))
    # build vocab from the dataset
    data_vocab = preprocess.build_vocab([tweets_tr, tweets_dev])
    # filter pretrained w2v with words from the dataset
    w2v = utils.restrict_w2v(w2v, set(data_vocab.keys()))
    w2v_vocab = preprocess.vocab_from_w2v(w2v)
    print("Cleaned vocab len:", len(w2v_vocab))
    # idx2word = {v: k for k, v in vocab.items()}
    return tweets_tr, tweets_dev, labels_tr, labels_dev, w2v, w2v_vocab


def train(tweets_tr, tweets_dev, labels_tr, labels_dev, w2v, w2v_vocab):
    hidden_size = 100
    batch_size = 512
    epochs = 5

    train_gen = TextSequence(
        tweets_tr, labels_tr, vocab=w2v_vocab, batch_size=batch_size, max_len=100
    )

    dev_gen = TextSequence(
        tweets_dev, labels_dev, vocab=w2v_vocab, batch_size=batch_size, max_len=100
    )

    model = models.build_model(
        hidden_size=hidden_size,
        dropout=0.3,
        recurrent_dropout=0.2,
        learning_rate=1.0,
        vocab_size=len(w2v_vocab),
        embedding_size=100,
        word2vec=w2v,
    )

    print("Starting training...")
    start = time.time()
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(tweets_tr) // batch_size,
        epochs=epochs,
        validation_data=dev_gen,
        validation_steps=len(tweets_dev) // batch_size,
        shuffle=True
        # callbacks=[es, cp],
    )
    end = time.time()
    print(utils.timer(start, end))

    # evaluation
    # dev path
    # crisis_test = [config.INDIA_FL_TWEETS]
    # normal_test = [config.NORMAL_TWEETS7]
    # tweets_feat_test, tweets_label_test = utils.read_datasets(crisis_test, normal_test)
    # tweets_feat_test = preprocess.texts_to_sequences(tweets_feat_test, w2v_vocab)
    #
    # evaluation.evaluate(model, tweets_feat_test, tweets_label_test)
    return model


def main():
    tweets_tr, tweets_dev, labels_tr, labels_dev, w2v, w2v_vocab = process()
    # model = train(tweets_tr, tweets_dev, labels_tr, labels_dev, w2v, w2v_vocab)


if __name__ == "__main__":
    main()
