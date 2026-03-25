from datasets import load_from_disk

from src.functions import clean_memory
from src.pipe_multilabel import PipeMultilabel

MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 512
N_CLASSES = 4
CLASSES = [
    'loneliness', 
    'emptiness', 
    'hopelessness', 
    'worthlessness', 
    'sadness', 
    'suicide intent', 
    'brain dysfunction (forget)', 
    'anger'
]
DEVICE = "cpu"
DEVICE_BATCH_SIZE = 4

dsd = load_from_disk("./data/multi-label-split/")
print(dsd)
try: 
    pipe = PipeMultilabel(
        text_column = "text-clean-no-emoji",
        label_columns = CLASSES,
        output_dir = "./models/test",
        model_name = MODEL_NAME, 
        device_batch_size = DEVICE_BATCH_SIZE, 
        device = DEVICE, 
        tokenizer_max_length = MAX_LENGTH,
    )

    dsd = pipe.tokenize(dsd)

    pipe.train(dsd, test_mode = True)
    pipe.save_important_info("./models/config/test.json")

except Exception as e:
    print(f"ERROR IN ROUTINE\n{e}")
finally:
    del pipe, dsd
    clean_memory()
    