# logic referenced from https://spotintelligence.com/2023/04/21/fine-tuning-gpt-3/#A_step-by-step_guide_to_fine-tuning_GPT-3

import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments

task_name = "phishing_classification"
num_labels = 3

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2ForSequenceClassification.from_pretrained('EleutherAI/gpt-neo-1.3') 

model.resize_token_embeddings(len(tokenizer))
model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)