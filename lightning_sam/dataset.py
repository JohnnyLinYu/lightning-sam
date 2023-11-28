import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
# from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class COCODataset(Dataset):

    def __init__(self, transform=None, Name = "training",TV_ratio = 0.8):
        # TV_ratio是tainingset Validationset之間的比例
#         self.root_dir = root_dir
        self.transform = transform
#         self.coco = COCO(annotation_file)
        
        self.Imgpaths = "/kaggle/input/cpn-trainingdata/TrainingSet/RawData"
        # 遍歷每個路徑
        files = os.listdir(self.Imgpaths)
        if(Name == "training"):
            files_name = files[:int(len(files) * TV_ratio)]
        else:
            files_name = files[int(len(files) * TV_ratio):]
        prefix_length = -4
        prefixes = [file_name[:prefix_length] for file_name in files_name]
        self.image_ids = prefixes

        # Filter out image_ids without any annotations
        #self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
#         image_info = self.coco(image_id)[0]
        image_path = os.path.join(self.Imgpaths, image_id + '.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        bboxes = []
        masks = []
        # paths = ["/kaggle/input/cpn-trainingdata/TrainingSet/Mask/F1",
        #          "/kaggle/input/cpn-trainingdata/TrainingSet/Mask/S1",
        #          "/kaggle/input/cpn-trainingdata/TrainingSet/Mask/S2",
        #          "/kaggle/input/cpn-trainingdata/TrainingSet/Mask/S3"]
        paths = ["/kaggle/input/cpn-trainingdata/TrainingSet/Mask/F1"]
        for path in paths:
            file_path = os.path.join(path, image_id + '.png')
            im = cv2.imread(file_path)
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
            if len(contours) > 0:
                x,y,w,h = cv2.boundingRect(contours[0])
                bboxes.append([x, y, x + w, y + h])
                masks.append(gray)

        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).float()


def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, masks, bboxes


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCODataset(transform=transform, Name = "training", TV_ratio = 0.8)
    val = COCODataset(transform=transform, Name = "Validation", TV_ratio = 0.8)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=torch.cuda.device_count(),
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=torch.cuda.device_count(),
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader
