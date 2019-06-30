import pathlib

# directories
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
RES_DIR = pathlib.Path(__file__).resolve().parent.parent / "resources"
OUTPUT_DIR = RES_DIR / "output"
VOCABS_DIR = RES_DIR / "vocabs"

TRAIN_DIR = DATA_DIR / "train"
CRISIS_TRAIN_DIR = TRAIN_DIR / "crisisnlp"
CRISISLEX_DIR = TRAIN_DIR / "crisislex"
NORMAL_TRAIN_DIR = TRAIN_DIR / "normal"
EVAL_DIR = DATA_DIR / "evaluation"
CRISIS_EVAL_DIR = EVAL_DIR / "crisisnlp"
NORMAL_EVAL_DIR = EVAL_DIR / "normal"

# embeddings
CRISIS_PRE_TRAINED = RES_DIR / "embeddings" / "crisisNLP_emb.bin"

# vocabs
TRAIN_VOCAB = VOCABS_DIR / "train_vocab.txt"

# crisis nlp
PAKISTAN_EQ_TWEETS = CRISIS_TRAIN_DIR / "2013_pakistan_eq.csv"
CALIFORNIA_EQ_TWEETS = CRISIS_TRAIN_DIR / "2014_california_eq.csv"
CHILE_EQ_TWEETS = CRISIS_TRAIN_DIR / "2014_chile_eq.csv"
NEPAL_EQ_TWEETS = CRISIS_TRAIN_DIR / "2015_nepal_eq_cf_labels.csv"
EBOLA_TWEETS = CRISIS_TRAIN_DIR / "2014_ebola_virus.csv"
ODILE_HR_TWEETS = CRISIS_TRAIN_DIR / "2014_hurricane_odile.csv"
HAGUPIT_HR_TWEETS = CRISIS_TRAIN_DIR / "2014_typhoon_hagupit_cf_labels.csv"
PAM_HR_TWEETS = CRISIS_TRAIN_DIR / "2015_cyclone_pam_cf_labels.csv"
MERS_TWEETS = CRISIS_TRAIN_DIR / "2014_mers_cf_labels.csv"
PAKISTAN_FL_TWEETS = CRISIS_TRAIN_DIR / "2014_pakistan_floods_cf_labels.csv"
INDIA_FL_TWEETS = CRISIS_TRAIN_DIR / "2014_india_floods.csv"
