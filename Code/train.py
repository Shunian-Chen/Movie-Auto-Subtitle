# !pip install transformers
# !pip install datasets
import os
import torch
import numpy as np
import soundfile as sf
import pandas as pd
from transformers import Trainer
from dataclasses import dataclass
from transformers import TrainingArguments
from typing import Dict, List, Union
from datasets import load_metric, Dataset, load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def train_test_split(ds, test_size = 0.3):
    n = len(ds)
    idx = np.random.permutation(n)
    train = ds.select(idx[round(n*test_size):])
    test = ds.select(idx[:round(n*test_size)])
    return train, test

def prepare_dataset(batch):
    # audio = batch["audio"]
    audio_input, sample_rate = sf.read(batch["path"])
    # batched output is "un-batched"
    batch["input_values"] = processor(audio_input, sampling_rate=sample_rate).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

device = 'cpu'
if __name__ == "__main__":
    project_path = os.path.abspath(os.path.join(os.path.abspath("."), ".."))
    repo_name = os.path.join(project_path, "wav2vec2-large-xlsr-53-english")
    AUDIO_BASE = os.path.join(project_path, "Data", "Train")
    CLEAN_AUDIO = os.path.join(project_path, "Data", "Train", "Clean")
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

    #load dataset
    print("Loading data...")
    label = pd.read_csv(os.path.join(AUDIO_BASE, "labels.csv"))
    label.path = label['path'].apply(lambda x: os.path.join(CLEAN_AUDIO, x))
    dataset = Dataset.from_pandas(label)
    # dataset = load_dataset("mozilla-foundation/common_voice_2_0", "en", use_auth_token=True, split = "train", cache_dir="K:\\AIPI540\\Individual Project")
    
    # load pretrained model
    print("Loading model..")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
    #set trianing arguments
    model.freeze_feature_encoder()
    print("Load Model Done!")

    #split dataset
    train, test = train_test_split(dataset)
    train_pro = train.map(prepare_dataset, remove_columns=train.column_names)
    test_pro = test.map(prepare_dataset, remove_columns=test.column_names)
    print("Load Data Done!")

    #save path
    training_args = TrainingArguments(
        output_dir=repo_name,
        group_by_length=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",
        num_train_epochs=50,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=50,
        save_total_limit=5,
        push_to_hub=False,
        )

    #load data class
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    #set trainer
    wer_metric = load_metric("wer")
    trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_pro,
    eval_dataset=test_pro,
    tokenizer=processor.feature_extractor,
    )
    print("Training...")
    #train the model
    trainer.train()
    print("Train Done!")