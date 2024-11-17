import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments 
from datasets import load_dataset

model_name = 'bert-base-uncased'#name of the model
model = BertForMaskedLM.from_pretrained(model_name)#model
tok = BertTokenizer.from_pretrained(model_name)#tokenizer

data = load_dataset("Plishing_validation_emails.csv")

