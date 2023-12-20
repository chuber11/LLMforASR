
from data import MyDataset, DataCollatorSpeechSeq2SeqWithPadding, compute_metrics
from model import ASRModel

from transformers import AutoTokenizer
from transformers import WhisperProcessor
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import TrainerCallback

import math

class MySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.statistics = [0.0,0.0]

    #def evaluate(self, *args, **kwargs):
    #    kwargs["max_length"] = max_length
    #    kwargs["no_repeat_ngram_size"] = 6
    #    kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    #    return super().evaluate(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        res = super().compute_loss(model, inputs, return_outputs=True)

        if model.training:
            self.statistics[0] += res[0].detach()
            self.statistics[1] += 1

        if return_outputs:
            return res
        else:
            return res[0]

    def log(self, logs):
        if not "eval_ppl" in logs:
            if self.statistics[1] > 0:
                logs["ppl"] = math.exp(self.statistics[0].item()/self.statistics[1])
                self.statistics = [0.0,0.0]
        super().log(logs)

class MyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs): # Evaluate at the beginning of the training
        if state.global_step == 1:
            pass #control.should_evaluate = True

dataset = {} #DatasetDict()
dataset["train"] = MyDataset()
dataset["test"] = MyDataset(dev=True)

print(len(dataset["train"]))
print(len(dataset["test"]))

decoder_name = "mistralai/Mistral-7B-Instruct-v0.2"
audio_encoder_name = "openai/whisper-large-v3"

tokenizer = AutoTokenizer.from_pretrained(decoder_name)
tokenizer.pad_token = tokenizer.unk_token
processor = WhisperProcessor.from_pretrained(audio_encoder_name)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, tokenizer=tokenizer)

model = ASRModel(tokenizer=tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir="./saves",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=2e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=0, #500,
    max_steps=40000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=False,#True,
    save_steps=800,
    eval_steps=800,
    logging_steps=10,
    save_total_limit=1,
    #report_to=["tensorboard"],
    #load_best_model_at_end=True,
    #metric_for_best_model="wer",
    #greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_num_workers=1,
)

trainer = MySeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #tokenizer=processor.feature_extractor,
    #tokenizer=processor.tokenizer,
    callbacks=[MyCallback],
)

trainer.train(resume_from_checkpoint=True)

