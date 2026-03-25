import json 
import os 

from datasets import load_from_disk
import pandas as pd 

from src.prediciton_pipe import PredictionPipe

PATH_MODEL = "./models"
checkpoint = "test-multilabel/checkpoint-2"

dsd = load_from_disk("./data/multi-label-split/")
print(dsd)

with open("./models/config/test.json") as file:
    config = json.load(file)

pred_pipe = PredictionPipe(
    checkpoint_dir = f"{PATH_MODEL}/{checkpoint}",
    model_name = config["model_name"],
    text_column=config["text_column"],
    labels=config["labels"],
    tokenizer_parameters = config["tokenizer_parameters"],
    device_batch_size = 8
)

output : dict[str:pd.DataFrame]= pred_pipe.predict(dsd)
for split in output:
    output[split].to_csv(f"./outputs/{checkpoint.replace('/', '_')}_{split}.csv")

# delete heavy files
os.system(f"rm -rf {PATH_MODEL}/{checkpoint}")