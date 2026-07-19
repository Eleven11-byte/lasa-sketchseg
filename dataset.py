import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


class FSCOCOTrain(Dataset):
    """FS-COCO training split.

    Expected layout:
        root/sketches/*/*.png
        root/text/*/*.txt
    """

    def __init__(self, root, transform=None, augment=False, sketch_size=512):
        self.root = root
        self.transform = transform
        self.augment = augment
        self.sketch_dir = os.path.join(root, "sketches")
        self.text_dir = os.path.join(root, "text")
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomCrop(450),
            transforms.Resize((sketch_size, sketch_size)),
        ])

        self.sketch_files = []
        self.text_files = []
        sketch_subdirs = sorted(os.listdir(self.sketch_dir))
        text_subdirs = sorted(os.listdir(self.text_dir))
        for sketch_subdir, text_subdir in zip(sketch_subdirs, text_subdirs):
            sketch_subdir_path = os.path.join(self.sketch_dir, sketch_subdir)
            text_subdir_path = os.path.join(self.text_dir, text_subdir)
            for sketch_file, text_file in zip(sorted(os.listdir(sketch_subdir_path)), sorted(os.listdir(text_subdir_path))):
                self.sketch_files.append(os.path.join(sketch_subdir, sketch_file))
                self.text_files.append(os.path.join(text_subdir, text_file))

    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, index):
        with open(os.path.join(self.text_dir, self.text_files[index]), "r", encoding="utf-8") as f:
            caption = f.read()

        sketch = Image.open(os.path.join(self.sketch_dir, self.sketch_files[index])).convert("RGB")
        aug_sketch = ImageOps.invert(self.augmentation(ImageOps.invert(sketch)))

        if self.transform:
            sketch = self.transform(sketch)
            aug_sketch = self.transform(aug_sketch)

        if self.augment:
            sketch = torch.stack([sketch, aug_sketch])

        return sketch, caption


class FSCOCOTest(Dataset):
    """FS-COCO vector-sketch test split.

    Expected layout:
        root/images/*
        root/captions/*.txt
        root/vector_sketches/*.npy
        root/classes/*.json
        root/all_classes.json
    """

    def __init__(self, root):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.text_dir = os.path.join(root, "captions")
        self.stroke_dir = os.path.join(root, "vector_sketches")
        self.label_dir = os.path.join(root, "classes")
        self.img_files = sorted(os.listdir(self.img_dir))
        self.txt_files = sorted(os.listdir(self.text_dir))
        self.stroke_files = sorted(os.listdir(self.stroke_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.stroke_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        with open(os.path.join(self.label_dir, self.label_files[index]), "r", encoding="utf-8") as f:
            labels = json.load(f)
        with open(os.path.join(self.text_dir, self.txt_files[index]), "r", encoding="utf-8") as f:
            caption = f.read()
        pen_state = np.load(os.path.join(self.stroke_dir, self.stroke_files[index]), allow_pickle=True)
        return pen_state, labels, caption, img_path


fscoco_train = FSCOCOTrain
fscoco_test = FSCOCOTest
