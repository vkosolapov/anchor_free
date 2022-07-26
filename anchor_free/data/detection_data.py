import os
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from data.abstract_data import AbstractDataModule
from data.augmentation import augmentations
from consts import *


class DetectionDataModule(AbstractDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = (
            "/kaggle/working/data/afo-aerial-dataset-of-floating-objects/PART_1/PART_1"
        )
        self.image_size = DATA_IMAGE_SIZE_DETECTION

    def get_dataset(self, phase):
        image_dataset = YOLODataset(
            self.data_dir, phase, image_size=self.image_size, transform=self.transform,
        )
        return image_dataset


class YOLODataset(Dataset):
    def __init__(
        self, data_dir, phase, image_size, transform,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase
        with open(os.path.join(self.data_dir, f"{phase}.txt")) as file:
            self.annotation_lines = file.readlines()
        self.length = len(self.annotation_lines)
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        image, labels = self.load_image(index)
        image = np.array(image, np.float32)
        if self.phase == "train":
            result = augmentations(image=image, bboxes=labels)
            image = result["image"]
            labels = result["bboxes"]
            labels = np.array(labels, dtype=np.int32)

        image = self.transform(image)

        if len(labels.shape) < 2:
            labels = np.expand_dims(labels, axis=1)
        labels_count = labels.shape[0]
        labels = F.pad(
            torch.Tensor(labels),
            (0, 5 - labels.shape[1], 0, DATA_MAX_BOXES_COUNT - labels.shape[0]),
            "constant",
            0.0,
        )
        return (
            image,
            labels,
            labels_count,
        )

    def load_image(self, index):
        annotation_line = self.annotation_lines[index]

        image = Image.open(
            os.path.join(self.data_dir, "images", annotation_line.strip("\n"))
        )
        if len(np.shape(image)) != 3 or np.shape(image)[2] != 3:
            image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        # image = np.asarray(image)

        with open(
            os.path.join(
                self.data_dir,
                "6categories",
                annotation_line.replace("jpg", "txt").strip("\n"),
            )
        ) as file:
            labels = file.readlines()
        labels_ = []
        for label in labels:
            label_ = label.strip("\n").split(" ")
            label_.append(label_.pop(0))
            label_[0] = int(float(label_[0]) * self.image_size)
            label_[1] = int(float(label_[1]) * self.image_size)
            label_[2] = int(float(label_[2]) * self.image_size)
            label_[2] += label_[0]
            label_[3] = int(float(label_[3]) * self.image_size)
            label_[3] += label_[1]
            labels_.append(label_)
        labels = np.array(labels_, dtype=np.int32)

        if len(labels) > 0:
            labels[:, :4][labels[:, :4] < 0] = 0
            labels[:, :4][labels[:, :4] > self.image_size] = self.image_size
            box_w = labels[:, 2] - labels[:, 0]
            box_h = labels[:, 3] - labels[:, 1]
            labels = labels[np.logical_and(box_w > 1, box_h > 1)]

        return image, labels
