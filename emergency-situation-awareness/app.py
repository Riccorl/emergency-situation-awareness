import time

import config
import models
import preprocess
import utils
from sequence import TextSequence


def train():
    hidden_size = 100
    batch_size = 256
    epochs = 10

    # train paths
    crisis_train = [
        config.CHILE_EQ_TWEETS,
        config.CALIFORNIA_EQ_TWEETS,
        config.NEPAL_EQ_TWEETS,
        config.EBOLA_TWEETS,
        config.ODILE_HR_TWEETS,
    ]
    normal_train = [config.NORMAL_TWEETS1, config.NORMAL_TWEETS2]
    # dev path
    crisis_dev = [config.PAKISTAN_EQ_TWEETS, config.HAGUPIT_HR_TWEETS]
    normal_dev = [config.NORMAL_TWEETS3]

    print("Loading train tweets...")
    tweets_feat, tweets_label = utils.read_datasets(crisis_train, normal_train)
    print("Loading train tweets...")
    tweets_feat_dev, tweets_label_dev = utils.read_datasets(crisis_dev, normal_dev)
    vocab = preprocess.build_vocab([tweets_feat, tweets_feat_dev])

    print("Texts to sequences...")
    tweets_feat = preprocess.texts_to_sequences(tweets_feat, vocab)
    tweets_feat_dev = preprocess.texts_to_sequences(tweets_feat_dev, vocab)

    # train_gen = preprocess.batch_generator(
    #     tweets_feat, tweets_label, vocab=vocab, batch_size=batch_size, max_input_len=100
    # )
    # dev_gen = preprocess.batch_generator(
    #     tweets_feat_dev,
    #     tweets_label_dev,
    #     vocab=vocab,
    #     batch_size=batch_size,
    #     max_input_len=100,
    # )

    train_gen = TextSequence(
        tweets_feat,
        tweets_label,
        vocab=vocab,
        batch_size=batch_size,
        max_len=100,
        num_classes=2,
    )
    dev_gen = TextSequence(
        tweets_feat_dev,
        tweets_label_dev,
        vocab=vocab,
        batch_size=batch_size,
        max_len=100,
        num_classes=2,
    )

    model = models.build_model(
        hidden_size=hidden_size,
        dropout=0.45,
        recurrent_dropout=0.2,
        learning_rate=1.0,
        vocab_size=len(vocab),
        # input_length=100
    )

    print("Starting training...")
    start = time.time()
    history = model.fit_generator(
        train_gen,
        steps_per_epoch=len(tweets_feat) // batch_size,
        epochs=epochs,
        validation_data=dev_gen,
        validation_steps=len(tweets_feat_dev) // batch_size,
        shuffle=True
        # callbacks=[es, cp],
    )
    end = time.time()
    print(utils.timer(start, end))


def main():
    pass
    # paths = [config.CRISIS_NLP_VOLUNTEERS]  # config.CRISIS_NLP_WORKERS
    # utils.unzip_all(paths)
    # df = utils.read_datasets(paths)
    # for i in df:
    #     print(i, "-->\n", "\t text:", df[i]["text"], "\t class: ", df[i]["class"])


if __name__ == "__main__":
    train()
    # utils.split_dataset(config.NORMAL_TWEETS, 60)
