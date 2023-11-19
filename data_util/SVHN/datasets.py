import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import SVHN

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class svhn_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, split="train", transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        svhn_dataobj = SVHN(self.root, self.split, self.transform, self.target_transform, self.download)

        if self.split == "train":
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = svhn_dataobj.data
            target = np.array(svhn_dataobj.labels)
        else:
            data = svhn_dataobj.data
            target = np.array(svhn_dataobj.labels)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            # 使用 permute 将维度重新排列
            img = np.transpose(img, (1, 2, 0))
            img = self.transform(img)


        if self.target_transform is not None:
            target = np.transpose(img, (1, 2, 0))
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
