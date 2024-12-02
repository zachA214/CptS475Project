# logic referenced from https://spotintelligence.com/2023/04/21/fine-tuning-gpt-3/#A_step-by-step_guide_to_fine-tuning_GPT-3

from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import evaluate
import pandas as pd
import numpy as np
#import tensorflow as tf
#import tensorflow_hub as hub
#import tensorflow_text as text
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Phishing_validation_emails.csv')
model_path = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
id2label = {0: "Safe Email", 1: "Phishing Email"}
label2id = {"Safe Email": 0, "Phishing Email": 1}
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = 2, idl2label=id2label, idlabel2id=label2id)


"""
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments

task_name = "phishing_classification"
num_labels = 3

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2ForSequenceClassification.from_pretrained('EleutherAI/gpt-neo-1.3') 

model.resize_token_embeddings(len(tokenizer))
model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
"""
"""
data = pd.read_csv('Phishing_validation_emails.csv')

data['phishing'] = data['Email Type'].apply(lambda x: 1 if x=='Phishing Email' else 0)


x_train, x_test, y_train, y_test = train_test_split(data['Email Text'], data['phishing'], stratify=data['phishing'])
print(x_train.head(4))
"""
