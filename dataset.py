import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from os.path import join
from PIL import Image

class AugmentedDataset(Dataset):
    def __init__(self, img_list, img_size):
        self.img_list = img_list
        self.img_size = img_size
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = self.transform(image=img)
        return {'img_list': idx, 'img': img['image']}


if __name__ == '__main__':
    img_paths = '/home/ganpan-retrieval/dataset/00.query/qh1.jpg'
    img = cv2.imread(f'{img_paths}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        #A.LongestMaxSize(max_size=224, interpolation=0, p=1.0),
        #A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0,0,0)),
        A.Resize(224,224)
        ])
    augmented_image = transform(image=img)['image']
    tmp = Image.fromarray(augmented_image)
    tmp.save("./sample.jpg")
