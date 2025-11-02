import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPTokenizer
import random
import numpy as np

class CTDataset(Dataset):
    def __init__(self, data_dir, split, text_model_name, image_size=(256, 256)):
        self.split = split
        self.image_dir = os.path.join(data_dir, split, "images")
        self.mask_dir = os.path.join(data_dir, split, "masks")
        metadata_path = os.path.join(data_dir, split, 'metadata.csv')
        self.metadata = pd.read_csv(metadata_path)
        self.image_size = image_size
        self.tokenizer = CLIPTokenizer.from_pretrained(text_model_name)
        if split == 'train':
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=25, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample_info = self.metadata.iloc[idx]
        image_filename = sample_info['image_filename']
        mask_filename = sample_info['mask_filename']
        original_prompt = sample_info['prompt']
        is_positive_sample = True
        prompt = original_prompt

        if self.split == 'train' and random.random() < 0.5:
            is_positive_sample = False
            if 'crack' in image_filename:
                taping_prompts = self.metadata[self.metadata['image_filename'].str.contains('taping')]['prompt'].unique()
                if len(taping_prompts) > 0:
                    prompt = random.choice(taping_prompts)
                else: is_positive_sample = True
            elif 'taping' in image_filename:
                crack_prompts = self.metadata[self.metadata['image_filename'].str.contains('crack')]['prompt'].unique()
                if len(crack_prompts) > 0:
                    prompt = random.choice(crack_prompts)
                else: is_positive_sample = True
            else:
                is_positive_sample = True

        image_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if is_positive_sample:
            mask_path = os.path.join(self.mask_dir, mask_filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype=np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        mask = mask.unsqueeze(0).float() / 255.0
        text_tokens = self.tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        text_tokens = {key: val.squeeze(0) for key, val in text_tokens.items()}

        return { "image": image, "mask": mask, "text_tokens": text_tokens, "prompt": prompt }
