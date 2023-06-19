import torch
from torch.utils.data import Dataset
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_folder = '/mnt/nikita_disk/Neuroscience/BCI_C_IV/images_train/'
image_size = 64


class CustomDataset(Dataset):
    def __init__(self,
                 root_dir,
                 image_size,
                 df,
                 transform=None,
                 mode="val"):
        self.root_dir = root_dir
        self.df = df
        self.file_names = df['file_name'].values
        self.labels = df['label'].values
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file_path and label for index
        label = self.labels[idx]
        file_path = os.path.join(self.root_dir, self.file_names[idx])

        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # Convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        augmented = self.transform(image=image)
        image = augmented['image']

        # Normalize because ToTensorV2() doesn't normalize the image
        image = image/255

        return image, label
