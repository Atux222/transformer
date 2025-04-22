import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os
from datasets import load_dataset, load_from_disk
import wandb

wandb.init(project="De-En Translation")



from transformers import AutoModel, AutoTokenizer

# 指定本地模型路径
model_path = "/home/autolab/xyy/token_model"

# 加载模型和分词器
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


dataset = load_dataset("parquet", data_files={
    "train": [
        "/home/autolab/xyy/dataset/train-00000-of-00003.parquet",
        "/home/autolab/xyy/dataset/train-00001-of-00003.parquet",
        "/home/autolab/xyy/dataset/train-00002-of-00003.parquet",
    ],
    "validation": "/home/autolab/xyy/dataset/validation-00000-of-00001.parquet",
    "test": "/home/autolab/xyy/dataset/test-00000-of-00001.parquet",
})


def check_dataset_exists(dataset_path):
     return os.path.exists(dataset_path)

def load_or_process_dataset(max_length):
     train_path = "train_dataset_processed"
     valid_path = "valid_dataset_processed"
     test_path = "test_dataset_processed"

     if check_dataset_exists(train_path) and check_dataset_exists(valid_path) and check_dataset_exists(test_path):
         print("加载已处理的数据集...")
         train_dataset = load_from_disk(train_path)
         valid_dataset = load_from_disk(valid_path)
         test_dataset = load_from_disk(test_path)
     else:
         print("数据集不存在，进行处理...")

         def tokenize_function(examples):
             de_texts = [item['de'] for item in examples['translation']]
             en_texts = [item['en'] for item in examples['translation']]
             encoding = tokenizer(en_texts, truncation=True, padding="max_length", max_length=max_length,  add_special_tokens=False)
             labels = tokenizer(de_texts, truncation=True, padding="max_length", max_length=max_length,  add_special_tokens=True)
             encoding['labels'] = labels['input_ids']
             return encoding


         train_dataset = dataset["train"].map(tokenize_function, batched=True)
         valid_dataset = dataset["validation"].map(tokenize_function, batched=True)
         test_dataset = dataset["test"].map(tokenize_function, batched=True)

         train_dataset.save_to_disk(train_path)
         valid_dataset.save_to_disk(valid_path)
         test_dataset.save_to_disk(test_path)

     return train_dataset, valid_dataset, test_dataset

# 调用数据处理函数
if __name__ == "__main__":
    max_length = 128  # 可以调整最大长度
    train_dataset, valid_dataset, test_dataset = load_or_process_dataset(max_length)