import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class FractureDataset(Dataset):
    def __init__(self, case_ids, labels, root_dir, image_transform=None, tokenizer=None, max_token_len=40):
        self.case_ids = case_ids
        self.labels = labels
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        case_id, label = self.case_ids[idx], self.labels[idx]
        case_folder = os.path.join(self.root_dir, case_id)
        subfolders = os.listdir(case_folder)
        assert len(subfolders) == 1, f"Expected 1 subfolder in {case_folder}, found {subfolders}"
        inner_folder = os.path.join(case_folder, subfolders[0])

        # Load images
        image_paths = [os.path.join(inner_folder, f) for f in os.listdir(inner_folder) if f.endswith('.jpg')]
        images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            if self.image_transform:
                image = self.image_transform(image)
                image = torch.clamp(image, 0.0, 1.0)
            images.append(image)

        # Load text
        text_file = next((f for f in os.listdir(inner_folder) if f.endswith(".txt")), None)
        assert text_file is not None, f"No text file found in {inner_folder}"
        with open(os.path.join(inner_folder, text_file), "r", encoding="utf-8") as f:
            full_text = f.read().strip()

        if self.tokenizer:
            tokenized = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_token_len,
                return_tensors="pt"
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
        else:
            input_ids = attention_mask = None

        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)

        return {
            "pixel_values": images,
            "text": full_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "case_id": case_id
        }