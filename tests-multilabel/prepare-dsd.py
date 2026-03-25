import pandas as pd 
from datasets import DatasetDict, Dataset

DATA_PATH = "./data/multilabel_classification_emotions.csv"
RANDOM_SEED = 2306406

# ====  ====  ====  ====  ====  ====  ====  ====  ====  ====  ====  ====  ====  
df = pd.read_csv(DATA_PATH)

# Create splits 
N = len(df)
N_train = int(N * 0.7)
N_train_eval = int(N * 0.1)
N_test = int(N * 0.1)
N_final_eval = N - N_train - N_train_eval - N_test 

assert N_final_eval > 0


indices = df.index.to_series()
indices_train = (
    indices
    .sample(n = N_train, random_state=RANDOM_SEED)
)
indices_train_eval = (
    indices
    .drop(index=indices_train.index)
    .sample(n = N_train_eval, random_state=RANDOM_SEED)
)
indices_test = (
    indices
    .drop(index=[*indices_train.index, *indices_train_eval.index])
    .sample(n = N_train_eval, random_state=RANDOM_SEED)
)
indices_final_test = (
    indices
    .drop(index = [*indices_train, *indices_train_eval,*indices_test])
)

df_split = (
    pd.concat({
        "train"         : df.loc[indices_train      , :],
        "train_eval"    : df.loc[indices_train_eval , :],
        "test"          : df.loc[indices_test       , :],
        "final_test"     : df.loc[indices_final_test  , :],
    })
    .reset_index()
    .drop(columns=["level_1"])
    .rename(columns = {"level_0": "split"})
)

grouped_ds_split = df_split.groupby("split")
dsd = DatasetDict({
    split :( 
        Dataset
        .from_pandas(grouped_ds_split.get_group(split))
    )
    for split in ["train", "train_eval", "test", "final_test"]
})

dsd.save_to_disk("./data/multi-label-split")