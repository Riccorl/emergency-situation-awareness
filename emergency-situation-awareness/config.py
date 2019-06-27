import pathlib

# directories
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
CRISISNLP_DIR = DATA_DIR / "crisisnlp"
CRISISLEX_DIR = DATA_DIR / "crisislex"
NORMAL_DIR = DATA_DIR / "normal"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"

#
CRISIS_PRE_TRAINED = pathlib.Path(__file__).resolve().parent.parent / "embeddings" / "crisisNLP_emb.bin"

# crisis nlp
PAKISTAN_EQ_TWEETS = CRISISNLP_DIR / "2013_pakistan_eq.csv"
CALIFORNIA_EQ_TWEETS = CRISISNLP_DIR / "2014_california_eq.csv"
CHILE_EQ_TWEETS = CRISISNLP_DIR / "2014_chile_eq.csv"
NEPAL_EQ_TWEETS = CRISISNLP_DIR / "2015_nepal_eq_cf_labels.csv"
EBOLA_TWEETS = CRISISNLP_DIR / "2014_ebola_virus.csv"
ODILE_HR_TWEETS = CRISISNLP_DIR / "2014_hurricane_odile.csv"
HAGUPIT_HR_TWEETS = CRISISNLP_DIR / "2014_typhoon_hagupit_cf_labels.csv"
PAM_HR_TWEETS = CRISISNLP_DIR / "2015_cyclone_pam_cf_labels.csv"
MERS_TWEETS = CRISISNLP_DIR / "2014_mers_cf_labels.csv"
PAKISTAN_FL_TWEETS = CRISISNLP_DIR / "2014_pakistan_floods_cf_labels.csv"
INDIA_FL_TWEETS = CRISISNLP_DIR / "2014_india_floods.csv"

# normal tweets
NORMAL_TWEETS = NORMAL_DIR / "tweets_normal.csv"
NORMAL_TWEETS1 = NORMAL_DIR / "tweets_normal_1.csv"
NORMAL_TWEETS2 = NORMAL_DIR / "tweets_normal_2.csv"
NORMAL_TWEETS3 = NORMAL_DIR / "tweets_normal_3.csv"
NORMAL_TWEETS4 = NORMAL_DIR / "tweets_normal_4.csv"
NORMAL_TWEETS5 = NORMAL_DIR / "tweets_normal_5.csv"
NORMAL_TWEETS6 = NORMAL_DIR / "tweets_normal_6.csv"
NORMAL_TWEETS7 = NORMAL_DIR / "tweets_normal_7.csv"
NORMAL_TWEETS8 = NORMAL_DIR / "tweets_normal_8.csv"
NORMAL_TWEETS9 = NORMAL_DIR / "tweets_normal_9.csv"
NORMAL_TWEETS10 = NORMAL_DIR / "tweets_normal_10.csv"
