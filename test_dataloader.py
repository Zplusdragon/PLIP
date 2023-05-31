import torch
import torch.utils.data as data
import os
import json
from transformers import BertTokenizer
import random
import copy
from torch.utils.data import DataLoader
from PIL import Image
from prefetch_generator import BackgroundGenerator
import numpy as np

class Dataset_test_image(data.Dataset):
    def __init__(self, image_path, dataset_path, transform=None):
        assert transform is not None, 'transform must not be None'
        self.impath = image_path
        self.datapath = dataset_path
        with open(dataset_path, 'r', encoding='utf8') as fp:
            self.dataset = json.load(fp)
        self.transform = transform
        print("Information about image gallery:{}".format(len(self)))

    def __getitem__(self, index):
        label = self.dataset[index]["id"]
        file_path = self.dataset[index]["file_path"]
        image = Image.open(os.path.join(self.impath, file_path)).convert('RGB')
        image_gt = self.transform(image)
        label = torch.tensor(label)
        return label,image_gt

    def __len__(self):
        return len(self.dataset)

class Dataset_test_text(data.Dataset):
    def __init__(self, image_path, dataset_path):
        self.impath = image_path
        self.datapath = dataset_path
        with open(dataset_path, 'r', encoding='utf8') as fp:
            self.dataset = json.load(fp)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.initial_data = []
        self.caption_depart_initial()
        print("Information about text query:{}".format(len(self)))

    def caption_to_tokens(self, caption):
        result = self.tokenizer(caption, padding="max_length", max_length=64, truncation=True, return_tensors='pt')
        token, mask = result["input_ids"], result["attention_mask"]
        token, mask = token.squeeze(), mask.squeeze()
        return token, mask

    def caption_depart_initial(self):
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            label = item["id"]
            captions_list = item["captions"]
            for j in range(len(captions_list)):
                caption = captions_list[j]
                self.initial_data.append([label,caption])

    def __getitem__(self, index):
        caption = self.initial_data[index][1]
        label = self.initial_data[index][0]
        caption_tokens, masks = self.caption_to_tokens(caption)
        caption_tokens = torch.tensor(caption_tokens)
        label = torch.tensor(label)
        return label, caption_tokens, masks

    def __len__(self):
        return len(self.initial_data)

def get_loader_test(args, transform, batch_size, num_workers):
    image_path = args.image_path
    test_path = args.test_path
    dataset_image = Dataset_test_image(image_path, test_path,transform=transform)
    dataset_text = Dataset_test_text(image_path,test_path)
    image_dataloader = DataLoader(dataset_image, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    text_dataloader = DataLoader(dataset_text, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return image_dataloader, text_dataloader