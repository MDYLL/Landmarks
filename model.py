import glob
import dlib
import numpy as np
import os
from tqdm import tqdm

from utils import get_data

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn as nn
from torch.utils.data import Dataset

NEW_SIZE = (96, 96)
train_transform = A.Compose(
    [A.Resize(*NEW_SIZE),
     A.ShiftScaleRotate(shift_limit=.05,
                        scale_limit=.1,
                        rotate_limit=10,
                        p=0.5),
     # A.HorizontalFlip(p=0.5),
     A.Normalize(),
     ToTensorV2()
     ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
)
test_transform = A.Compose(
    [A.Resize(*NEW_SIZE),
     A.Normalize(),
     ToTensorV2()
     ],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
)


class LandmarksDataset(Dataset):

    def __init__(self, img_pathes, transform):
        super(LandmarksDataset).__init__()
        self.image_names = []
        self.data = {}
        self.transform = transform
        for extension in ["*jpg", "*png"]:
            for path in img_pathes:
                for f in tqdm(glob.glob(os.path.join(path, extension))):
                    data = get_data(f)
                    if data is None:
                        continue
                    key_points, left, right, top, bottom, height, width = data
                    self.image_names.append(f)
                    self.data[f] = {"key_points": key_points, "left": left, "right": right, "top": top,
                                    "bottom": bottom}

    def __getitem__(self, index):
        img_path = self.image_names[index]
        img = dlib.load_rgb_image(img_path)
        key_points = self.data[img_path]["key_points"].copy()
        left = self.data[img_path]["left"]
        right = self.data[img_path]["right"]
        top = self.data[img_path]["top"]
        bottom = self.data[img_path]["bottom"]
        img = img[top:bottom, left:right]
        key_points[:, 0] = key_points[:, 0] - left
        key_points[:, 1] = key_points[:, 1] - top

        if self.transform is not None:
            transformed = self.transform(image=img, keypoints=key_points)
            image = transformed["image"]
            key_points = transformed["keypoints"]
            key_points = np.array(key_points).reshape(-1)
        return image.float(), key_points, img_path, left, top, right, bottom

    def __len__(self):
        return len(self.image_names)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1 = nn.Conv2d(32, 32, 3, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=0)
        self.conv4 = nn.Conv2d(64, 128, 2, padding=0)
        self.bn0 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.max_pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 68 * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = x.reshape(-1, 1152)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
