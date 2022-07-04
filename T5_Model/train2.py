import transformers
import pickle
import torch
model = transformers.T5ForConditionalGeneration.from_pretrained("checkpoint-221600")

EPOCHS = 40
TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 256
SEQ_LEN=17

with open("train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)
with open("test_dataset.pkl", "rb") as f:
    test_dataset = pickle.load(f)

training_args = transformers.TrainingArguments(
    # ------------------------------------------------------- [epochs and batch size]
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE*2,
    gradient_accumulation_steps=1,
    # ------------------------------------------------------- [hyperparams]
    warmup_steps=100, 
    weight_decay=0.01,
    # ------------------------------------------------------- [save and logging]
    output_dir=".", 
    overwrite_output_dir = True,
    do_eval = False,
    logging_strategy="epoch", # activate if interested
    save_strategy="no",
    save_total_limit = None,
    # -------------------------------------------------------
)
trainer = transformers.Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset,
    data_collator = lambda data: {
        'input_ids': torch.stack([f[0] for f in data]), 
        # 'attention_mask': torch.stack([f[1] for f in data]), 
        'labels': torch.stack([f[1] for f in data]),
    },
    # resume_from_checkpoint=True
)
trainer.train(resume_from_checkpoint=True)
trainer.save_model("t5_20+20_epochs")