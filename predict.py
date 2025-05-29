import argparse
import logging
import os
import tifffile
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import time
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

# 记录验证Data的DICE分数路径
dice_score_path = "/home/ubuntu/disk1/lzx/Pytorch-UNet/TestRawData_DiceScore.txt"

# 计算DICE分数
from utils.dice_score import multiclass_dice_coeff, dice_coeff

# image路径
image_path = "/home/ubuntu/disk1/lzx/dataset/neuron/images_croped512/"

# mask路径
mask_path = "/home/ubuntu/disk1/lzx/dataset/neuron/masks_croped512/"

# dice分数列表
dice_scores = []

i = 0

# 根据image找到mask的函数
def find_mask(img_filename):
    """
    mask的命名和image相同 后缀是.png
    :param img_filename: str, image的文件名
    :return: np.array 打开的mask图像
    """
    base_name = os.path.basename(img_filename)
    mask_name = os.path.splitext(base_name)[0] + '.png'
    mask_file = os.path.join(mask_path, mask_name)
    if not os.path.isfile(mask_file):
        raise FileNotFoundError(f'Mask file {mask_file} does not exist for image {img_filename}')
    return np.array(Image.open(mask_file))

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        print(f"[DEBUG] full_img size: {full_img.size} full_img shape: {full_img.shape}")
        output = F.interpolate(output, (full_img.shape[2], full_img.shape[1]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')

    parser.add_argument('--model', '-m', 
                        default='./checkpoints/checkpoint_epoch5.pth', 
                        metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', 
                        default="/home/ubuntu/disk1/lzx/dataset/neuron/images_croped512/",
                        metavar='INPUT', 
                        nargs='+', 
                        help='dirnames of input images', 
                        required=True)
    parser.add_argument('--output', '-o', 
                        metavar='OUTPUT', 
                        nargs='+', 
                        help='Filenames of output images')
    parser.add_argument('--viz', '-v', 
                        default=False,
                        action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', 
                        default=True,
                        action='store_true', 
                        help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', 
                        type=float, 
                        default=1.0,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', 
                        type=float, 
                        default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', 
                        action='store_true', 
                        default=False, 
                        help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', 
                        type=int, 
                        default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    ### DEBUG ###
    # print(f'Input files: {in_files}')
    out_files = get_output_filenames(args)
    ### DEBUG ###
    # print(f'Output files: {out_files}')

    net = UNet(n_channels=8, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    print(f'[DEBUG] state_dict keys: {state_dict.keys()}')
    mask_values = state_dict.pop('mask_values', [0, 1])
    print(f'[DEBUG] mask_values: {mask_values}')
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    # 修改args.input为目录，遍历文件进行predict
    # args.input是list，提取args.input的第一个元素作为目录
    if isinstance(args.input, list):
        args.input = args.input[0]

    ### DEBUG ###
    # print(f'[DEBUG] args.input: {args.input}')

    if not os.path.isdir(args.input):
        raise ValueError(f'Input path {args.input} is not a directory.')
    
    # 计算时间
    t0 = time.time()

    for filename in os.listdir(args.input):
        logging.info(f'Predicting image {i} ...')

        # 找到对应的mask.png文件
        mask_file = find_mask(filename)
        logging.info(f'Found mask file: {mask_file}')
        # 加载tif图像
        filename = os.path.join(image_path, filename)
        img = tifffile.imread(filename)
        img = np.asarray(img)

        mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=args.scale,
                        out_threshold=args.mask_threshold,
                        device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

        # 计算DICE分数
        input_mask = torch.tensor(mask_file).long()
        target_mask = torch.tensor(mask).long()
        dice_score = dice_coeff(
            input=input_mask,
            target=target_mask,
            reduce_batch_first=False
        )
        with open(dice_score_path, "a") as f:
            f.write(f"Image {i}: Dice Score = {dice_score.item()}\n")
            dice_scores.append(dice_score.item())
        logging.info(f'Dice Score for image {i}: {dice_score.item()}')
        print(f"Image {i}: Dice Score = {dice_score.item()}\n")
        # 更新i
        i += 1
    
    t1 = time.time()
    t_diff = t1 - t0
    with open(dice_score_path, "a") as f:
        avg_dice_score = sum(dice_scores) / len(dice_scores)
        f.write(f"Average Dice Score: {avg_dice_score}\n")
        f.write(f"Total time taken: {t_diff} seconds\n")
        f.write(f"Average time per image: {t_diff / len(dice_scores)} seconds\n")