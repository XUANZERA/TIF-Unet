import logging
import numpy as np
import torch
import tifffile
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
# Image.resize() -> cv2.resize()
import cv2


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext in ['.tif', '.tiff']:
        img = tifffile.imread(filename)
        img = np.array(img) 
        # print(f"[DEBUG1] Loaded tif: shape={img.shape}")
        # img = Image.fromarray(img)
        # print(f"[DEBUG2] Loaded tif: shape={img.shape}")
        return img
    # 我的mask的后缀是.png
    elif ext in ['.png']:
        img = Image.open(filename)
        img = np.array(img)
        return img
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    # FileNotFoundError: No mask file found for 0913_PC116_Scan1_[10108,46662]_component_data_tile_0_0 in /home/ubuntu/disk1/lzx/dataset/neuron/masks_croped512
    mask_file_found = list(mask_dir.glob(idx + mask_suffix + '.*'))
    # glob_pattern = idx + mask_dir + ".*"
    # mask_file_found = mask_dir / glob_pattern

    if not mask_file_found:
        raise FileNotFoundError(f'No mask file found for {idx} in {mask_dir}')
    mask_file = mask_file_found[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        # debug
        # return np.unique(mask)
        m = np.unique(mask)
        return m # 输出=[0, 1]
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids) 
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def preprocess(mask_values, np_img, scale, is_mask):
        # mask_values <- self.mask_values [0, 1]
        # w, h = pil_img.shape
        if len(np_img.shape) == 2:
            h, w = np_img.shape
        elif len(np_img.shape) == 3:
            c, h, w = np_img.shape
        else:
            raise ValueError(f'Image should have 2 or 3 dimensions, found {np_img.ndim}')

        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # np_img = cv2.resize(np_img, (newW, newH), interpolation=Image.NEAREST if is_mask else Image.BICUBIC)
        num_channels = c if len(np_img.shape) == 3 else 1
        img = []
        for i in range(num_channels):
            single_channel_img = np_img[i, :, :] if num_channels > 1 else np_img
            single_channel_img = cv2.resize(single_channel_img, (newW, newH), interpolation=Image.NEAREST if is_mask else Image.BICUBIC)
            img.append(single_channel_img)
        img = np.stack(img, axis=0) if num_channels > 1 else img[0]
        img = np.asarray(np_img)
        # print(f"DEBUG: img.shape: {img.shape}")

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    # print(f"DEBUG: mask.shape: {mask.shape}") # 该是0还是0 该是1还是1
                    # print(f"DEBUG: img.shape: {img.shape}")
                    # print(f"{(img == v).shape}")
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                pass

            if (img > 1).any():
                img = img / 255.0
            # print(f"DEBUG: img.shape: {img.shape}")
            return img
    # def preprocess(mask_values, pil_img, scale, is_mask):
    #     w, h = pil_img.size
    #     newW, newH = int(scale * w), int(scale * h)
    #     assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    #     pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    #     img = np.asarray(pil_img)

    #     if is_mask:
    #         mask = np.zeros((newH, newW), dtype=np.int64)
    #         for i, v in enumerate(mask_values):
    #             if img.ndim == 2:
    #                 mask[img == v] = i
    #             else:
    #                 mask[(img == v).all(-1)] = i

    #         return mask

    #     else:
    #         if img.ndim == 2:
    #             img = img[np.newaxis, ...]
    #         else:
    #             img = img.transpose((2, 0, 1))

    #         if (img > 1).any():
    #             img = img / 255.0

    #         return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # np.size和PIL.size意义不同，PIL.size是一个元组，而np.size是一个int
        # print(f"img.shape = {img.shape}")
        # print(f"mask.shape = {mask.shape}")
        # >>> img.shape = (8, 512, 512)
        # >>> mask.shape = (512, 512)
        assert (img.shape[1], img.shape[2]) == mask.shape, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='')
