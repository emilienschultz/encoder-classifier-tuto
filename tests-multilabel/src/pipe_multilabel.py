import json
from mergedeep import merge as dict_merge

from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer
)
from torch import Tensor

from .functions import compute_metrics, clean_memory

class PipeMultilabel:
    def __init__(
        self, 
        text_column : str, 
        label_columns : list[str], 
        output_dir: str, 
        model_name : str, 
        device_batch_size : int,
        device : str | None = None, 
        tokenizer_max_length : int | None = None, 
        additional_training_arguments : dict|None = None
        ) -> None :
        """
        PipeMultilabel
        """
        self.__device = device if device is not None else "cpu"
        self.__text_column = text_column
        self.__labels = list(label_columns)
        self.__num_labels = len(self.__labels)
        self.__id2label = {id:label for id, label in enumerate(self.__labels)}
        self.__label2id = {label:id for id, label in enumerate(self.__labels)}

        self.__tokenizer_parameters = {
            "truncation":True, 
            "padding":"max_length",
            "max_length": tokenizer_max_length if tokenizer_max_length is not None else 512,
            "return_tensors":"pt"
        }

        self.__model_name = model_name
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_name)
        self.__model = AutoModelForSequenceClassification.from_pretrained(
            self.__model_name, 
            num_labels=self.__num_labels,
            id2label=self.__id2label,
            label2id=self.__label2id,               
            problem_type = "multi_label_classification"                   
        ).to(device=self.__device)

        training_arguments = {
            # Hyperparameters
            # num_train_epochs = 5,
            "num_train_epochs" : 7,
            "learning_rate" : 5e-5,
            "weight_decay" : 0.0,
            "warmup_ratio" : 0.0,
            "optim" : "adamw_torch_fused",
            # Second order hyperparameters
            "per_device_train_batch_size" : 4,
            "per_device_eval_batch_size" : 4,
            "gradient_accumulation_steps" : 8,
            # Pipe
            "output_dir" : "./models/training",
            "overwrite_output_dir" : True,
            
            "logging_strategy" : "epoch",
            # eval_strategy = "epoch",
            "eval_strategy" : "steps",
            "eval_steps" : 32,
            "save_strategy" : "epoch",
            # load_best_model_at_end = True,
            # save_total_limit = 5 + 1,

            "disable_tqdm" : False,
        }
        dict_merge(
            # Default training arguments
            training_arguments, 
            # Overwrite training arguments
            {
                "output_dir" : output_dir,
                "per_device_train_batch_size" : device_batch_size,
                "gradient_accumulation_steps" : 32 // device_batch_size,
                "per_device_eval_batch_size" : device_batch_size * 2,
                "use_cpu":  self.__device == "cpu",
                "use_mps_device":  self.__device == "mps",
                "no_cuda" : self.__device != "cuda"
            },
            # include edditional training arguments
            additional_training_arguments if additional_training_arguments is not None else {}
        )
        self.__training_arguments = TrainingArguments(**training_arguments)

    def tokenize(self, dsd: DatasetDict): 
        def preprocess_dataset(row: dict):
            tokenized_entry = self.__tokenizer(row[self.__text_column], **self.__tokenizer_parameters)
            id_label_as_tensor = [int(row[label]) for label in self.__labels]
            id_label_as_tensor = Tensor(id_label_as_tensor)
            return {
                **row.copy(),
                "labels": id_label_as_tensor,
                "attention_mask" : tokenized_entry["attention_mask"].reshape(-1),
                "input_ids" : tokenized_entry["input_ids"].reshape(-1)
            }


        dsd = dsd.map(preprocess_dataset, batch_size=32)
        
        for split in dsd:
            dsd[split] = dsd[split].with_format("torch", device=self.__device)

        return dsd
    
    def train(self, dsd : DatasetDict, test_mode : bool = False):
        if test_mode : 
            print("TEST - train : 50 - train_eval : 50")
            dsd["train"] = dsd["train"].select(range(50))
            dsd["train_eval"] = dsd["train_eval"].select(range(50))
            
        try:
            trainer = Trainer(
                model = self.__model, 
                args = self.__training_arguments,
                train_dataset=dsd["train"],
                eval_dataset=dsd["train_eval"],
                compute_metrics = compute_metrics
            )

            trainer.train()
            # WARNING MODEL switches to mps need to handle that
        except Exception as e:
            print("# ERROR" + "#" * 93)
            print(e)
            print("#" * 100)

        finally:
            del self.__tokenizer, self.__model, dsd, trainer # All objects that are moved to the GPU
            clean_memory()

    def save_important_info(self, output_file : str):
        important_info = {
            "model_name" : self.__model_name,
            "tokenizer_parameters": self.__tokenizer_parameters,
            "output_dir" : self.__training_arguments.output_dir,
            "device-requested" : self.__device,
            "device-used": str(self.__training_arguments.device), 
            "text_column" : self.__text_column,
            "label_column" : self.__label_column,
            "labels" : self.__labels,
            "num_labels" : self.__num_labels,
            "id2label" : self.__id2label,
            "label2id" : self.__label2id,
        }
        with open(output_file, "w") as file: 
            json.dump(important_info, file, indent = 4, ensure_ascii = True)
