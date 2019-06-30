import gensim

import config
import evaluation
import preprocess
import train
import utils


def process():
    print("Loading tweets...")
    features, labels = utils.load_datasets(
        config.CRISIS_TRAIN_DIR, config.NORMAL_TRAIN_DIR
    )
    print("Clearing tweets...")
    features = preprocess.clear_tweets(features)
    print(features[0])
    print("Loading pre-trained embeddings...")
    # load the w2v matrix with genism
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        config.CRISIS_PRE_TRAINED, binary=True
    )
    # build the vocab from the w2v model
    w2v_vocab = preprocess.vocab_from_w2v(w2v)
    print("Word2Vec model vocab len:", len(w2v_vocab))
    # build vocab from the dataset
    data_vocab = preprocess.build_vocab([features])
    # filter pretrained w2v with words from the dataset
    w2v = utils.restrict_w2v(w2v, set(data_vocab.keys()))
    w2v_vocab = preprocess.vocab_from_w2v(w2v)
    utils.write_dictionary(config.TRAIN_VOCAB, w2v_vocab)
    print("Cleaned vocab len:", len(w2v_vocab))
    # idx2word = {v: k for k, v in vocab.items()}
    return features, labels, data_vocab, w2v, w2v_vocab


def main():
    features, labels, vocab, w2v, w2v_vocab = process()
    model = train.train_keras(features, labels, w2v=w2v, w2v_vocab=w2v_vocab)
    evaluation.evaluate(model)


if __name__ == "__main__":
    main()
