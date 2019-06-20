import config
import utils


def main():
    paths = [config.CRISIS_NLP_VOLUNTEERS] # config.CRISIS_NLP_WORKERS
    utils.unzip_all(paths)
    df = utils.read_datasets(paths)
    for i in df:
        print(i, '-->\n', '\t text:', df[i]['text'], '\t class: ', df[i]['class'])


if __name__ == "__main__":
    main()
