# ================================
# dataset.py
# ================================
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CatKeypointDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, image_size=384):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.image_size = image_size
        self.items = self._load_items()

    def _load_items(self):
        image_files = sorted([
            f for f in os.listdir(self.image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        items = []
        for img_file in image_files:
            json_file = os.path.splitext(img_file)[0] + ".json"
            json_path = os.path.join(self.label_dir, json_file)
            if os.path.exists(json_path):
                items.append((img_file, json_file))
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_file, json_file = self.items[idx]
        img_path = os.path.join(self.image_dir, img_file)
        label_path = os.path.join(self.label_dir, json_file)

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        with open(label_path, "r") as f:
            label_json = json.load(f)
            keypoints = label_json["labels"]

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        keypoints[:, 0] /= width
        keypoints[:, 1] /= height
        keypoints = keypoints.flatten()

        image = self.transform(image)

        return image, keypoints