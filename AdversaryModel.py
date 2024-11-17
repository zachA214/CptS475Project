import torch
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments 
from datasets import Dataset

import logging
logging.basicConfig(level=logging.DEBUG)



modelName = 'bert-base-uncased'#name of the model
model = BertForMaskedLM.from_pretrained(modelName)#model
tok = BertTokenizer.from_pretrained(modelName)#tokenizer

#loading the dataset
data = pd.read_csv('Phishing_validation_emails.csv')

#Putting into a dataframe, then filtering to only have the phishing emails
df = pd.DataFrame(data)
filtered = df[df['Email Type'] == 'Phishing Email']
#receiving only the first column (so only the emails) and making that our main dataset
#filtered = filtered[['Email Text']]

#making them dicts 
dicts = filtered.to_dict(orient = 'records')

#Making it a dataset again so it can still be trained
phishingData = Dataset.from_pandas(pd.DataFrame(dicts))