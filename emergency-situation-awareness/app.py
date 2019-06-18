import utils
import config
import pandas as pd


def main():
    df = utils.read_dataset(config.CRISIS_NLP_VOLUNTEERS)
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
        print(df)


if __name__ == "__main__":
    main()
