
import os
import json
from torch.utils.data import Dataset
from PIL import Image

class SimpleCaptionDataset(Dataset):
    """
    Loads COCO captions (2014 or 2017).
    Each entry: {"image_id": "path/to/image.jpg", "caption": "..."}
    """
    def __init__(self, captions_json, transform=None):
        if not os.path.exists(captions_json):
            raise FileNotFoundError(f"Captions file not found: {captions_json}")
        with open(captions_json, "r") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry["image_id"]).convert("RGB")
        caption = entry["caption"]
        if self.transform:
            image = self.transform(image)
        return image, caption


class SimpleVQADataset(Dataset):
    """
    Loads VQA dataset (2014 annotations + questions preprocessed into vqa_train.json).
    Each entry: {"image_id": "path/to/image.jpg", "question": "...", "answer": "..."}
    """
    def __init__(self, vqa_json, transform=None):
        if not os.path.exists(vqa_json):
            raise FileNotFoundError(f"VQA JSON file not found: {vqa_json}")
        with open(vqa_json, "r") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry["image_id"]).convert("RGB")
        question = entry["question"]
        answer = entry.get("answer", None)
        if self.transform:
            image = self.transform(image)
        return image, question, answer



 
