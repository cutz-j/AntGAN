# motivated source code
# https://github.com/AliaksandrSiarohin/motion-cosegmentation
#

import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pandas as pd
# from augmentation import AllAugmentationTransform
import glob
import random
from PIL import Image
from copy import deepcopy


def read_video(name, image_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with images
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array([img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + image_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = video
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, videos can be represented as:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with images
    """

    def __init__(self, root_dir, image_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, crop_prob=0.5):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.image_shape = tuple(image_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
               train_images = {os.path.basename(image).split('#')[0] for image in os.listdir(os.path.join(root_dir, 'train'))}
               train_images = list(train_images)
            else:
               train_images = os.listdir(os.path.join(root_dir, 'train'))
            test_images = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_images, test_images = train_test_split(self.images, random_state=random_seed, test_size=0.2)

        if is_train:
            self.images = train_images
        else:
            self.images = test_images

        self.is_train = is_train

        if self.is_train:
            crop = transforms.RandomResizedCrop(image_shape[0], scale=[0.8, 1.0], ratio=[0.9, 1.1])
            rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < crop_prob else x)

            self.transform = transforms.Compose([
                rand_crop,
                transforms.Resize([image_shape[0], image_shape[1]]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #                     std=[0.5, 0.5, 0.5]),
                ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize([image_shape[0], image_shape[1]]),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #         std=[0.5, 0.5, 0.5]),
                ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
           name = self.images[idx]
           path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
           name = self.images[idx]
           path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, image_shape=self.image_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames) # train --> random 2 --> sort // test --> all frame range
            video_array = video_array[frame_idx] # (2, w, h, c) // (frame, w, h, c)

        # if self.transform is not None:
        video_array = [self.transform(Image.fromarray(img)) for img in video_array]    
        out = {}
        if self.is_train:
            out['source'] = video_array[0] # (w, h, c)
            out['target'] = video_array[1] # (w, h, c)

            # out['target'] = target.transpose((2, 0, 1))
            # out['source'] = source.transpose((2, 0, 1))
        else:
            out['video'] = video_array # (f, c, w, h)
            

        out['name'] = video_name

        return out


class MotionDataset(Dataset):
    """
    Dataset of videos, videos can be represented as:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with images
    """

    def __init__(self, root_dir, K=8, image_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, crop_prob=0.5):
        self.root_dir = root_dir
        self.images = os.listdir(root_dir)
        self.image_shape = tuple(image_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.K = K
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
               train_images = {os.path.basename(image).split('#')[0] for image in os.listdir(os.path.join(root_dir, 'train'))}
               train_images = list(train_images)
            else:
               train_images = os.listdir(os.path.join(root_dir, 'train'))
            test_images = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_images, test_images = train_test_split(self.images, random_state=random_seed, test_size=0.2)

        if is_train:
            self.images = train_images
        else:
            self.images = test_images

        self.is_train = is_train

        if self.is_train:
            crop = transforms.RandomResizedCrop(image_shape[0], scale=[0.8, 1.0], ratio=[0.9, 1.1])
            rand_crop = transforms.Lambda(lambda x: crop(x) if random.random() < crop_prob else x)

            self.transform = transforms.Compose([
                rand_crop,
                transforms.Resize([image_shape[0], image_shape[1]]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #                     std=[0.5, 0.5, 0.5]),
                ])

        else:
            self.transform = transforms.Compose([
                # transforms.Resize([image_shape[0], image_shape[1]]),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                #         std=[0.5, 0.5, 0.5]),
                ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
           name = self.images[idx]
           path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
           name = self.images[idx]
           path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, image_shape=self.image_shape)
            num_frames = len(video_array)
            frame_idx = (np.random.choice(num_frames, replace=True, size=1)) if self.is_train else range(num_frames) # train: source --> random frame // target --> K-frame random chosen
            target_frame_idx = np.random.choice(num_frames-self.K, replace=False, size=1)
            source_video_array = video_array[frame_idx] # (1, w, h, c) // (frame, w, h, c)
            target_video_array = video_array[target_frame_idx[0]:target_frame_idx[0]+self.K, :, :, :] # (K, w, h, c)
            
        # if self.transform is not None:
        source_video_array = [self.transform(Image.fromarray(img)) for img in source_video_array]
        target_video_array = [self.transform(Image.fromarray(img)) for img in target_video_array]

        out = {}
        if self.is_train:
            out['source'] = source_video_array[0] # (c, w, h)
            out['target'] = target_video_array # (K, c, w, h)

        else:
            out['video'] = source_video_array # (f, c, w, h)
            

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """
    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats


    def __len__(self):
        return self.num_repeats * self.dataset.__len__()


    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]
