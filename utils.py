import os, random
import torch
from PIL import Image
from torch.utils.data import DataLoader
from dataloader import MRIDataset
import numpy as np


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def get_data(args):
    train_dataset = MRIDataset(os.path.join(args.dataset_path, args.train_folder), (args.img_size, args.img_size))
    val_dataset = MRIDataset(os.path.join(args.dataset_path, args.val_folder), (args.img_size, args.img_size))
    print(f'Train Dataset Size: {len(train_dataset)}')
    print(f'Val Dataset Size: {len(val_dataset)}')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataloader, val_dataloader


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)