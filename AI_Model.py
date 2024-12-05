# logic referenced from https://spotintelligence.com/2023/04/21/fine-tuning-gpt-3/#A_step-by-step_guide_to_fine-tuning_GPT-3

from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, DefaultDataCollator
import evaluate
import pandas as pd
import numpy as np
import torch
#import tensorflow as tf
#import tensorflow_hub as hub
#import tensorflow_text as text
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('Phishing_validation_emails.csv')
label_mapping = {"Safe Email": 0, "Phishing Email": 1} 
dataset["Email Type"] = dataset["Email Type"].map(label_mapping)

#edits
train_df, test_df = train_test_split(dataset)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
#edits end


#model_path = "google-bert/bert-base-uncased"

model_path = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path)
id2label = {0: "Safe Email", 1: "Phishing Email"}
label2id = {"Safe Email": 0, "Phishing Email": 1}
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = 2, id2label=id2label, label2id=label2id)

for name, param in model.base_model.named_parameters():
    param.requires_grad = False

for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True
    
def preprocess_function(examples): 
    return tokenizer(examples["Email Text"], padding='max_length', truncation=True, max_length=95)


tokenized_train_dataset = dataset_dict["train"].map(preprocess_function, batched=True) 
tokenized_test_dataset = dataset_dict["test"].map(preprocess_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.add_column("labels", train_dataset["Email Type"])
tokenized_test_dataset = tokenized_test_dataset.add_column("labels", test_dataset["Email Type"])

#tokenized_data = dataset_dict.map(preprocess_function,batched=True)

#added
#tokenized_data = tokenized_data.cast_column("Email Type", dataset_dict["train"]["Email Type"])
#added

#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
data_collator = DefaultDataCollator()

accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    #probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims = True)
    probabilities = torch.softmax(torch.tensor(predictions), dim=1).numpy()
    positive_class_probs = probabilities[:,1]
    auc=np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'],3)
    predicted_classes = np.argmax(probabilities, axis=1)
    acc=np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'],3)
    return{"Accuracy": acc, "AUC": auc}


lr = 2e-4
batch_size = 8
num_epochs = 10
training_args = TrainingArguments(
    output_dir = "bert-phishing-classifier_teacher",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args = training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer = tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

#Testing on input emails

# What to do:
# - File processing in python to read the data
# - Create a seperate csv for testing
# - Run in batches of 10 = 1 trial
# - Pull 50 pieces of data from Phishing_validation_emails.csv to new csv
# - Pull all data from genereated data into new csv


#email_texts = ["Thank you for your order. Your order will ship out soon. Click here to claim.", "Welcome to our program.", "Give me your credit card."]

email_texts = open('trial_data.txt', 'r')

for email_text in email_texts:
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.softmax(logits, dim=1)

        
    predicted_class = torch.argmax(predictions, dim=1).item()
    probabilities = predictions[0].cpu().numpy() 
    predicted_label = id2label[predicted_class]

    print(f"Email: {email_text}")
    print(f"Predicted Label: {predicted_label}")
    #print(f"Probability: {probabilities}")
    if "Safe" in predicted_label:
        print(f"Probability: {probabilities[0]}")
    else:
        print(f"Probability: {probabilities[1]}")

print("Finished all tasks   ")








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
