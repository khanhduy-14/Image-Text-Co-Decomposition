# ------------------------------------------------------------------------------
# CoDe
# Copyright (C) 2024 by Ji-Jia Wu. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from TCL (https://github.com/kakaobrain/tcl)
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import os
import os.path as osp
import random
import warnings
from functools import partial
import glob

import numpy as np
import torch.distributed as dist
import webdataset as wds
from braceexpand import braceexpand
from timm.data import create_transform
from torchvision import transforms as T
import us

from sclip.clip import tokenize

from torch.utils.data._utils.collate import default_collate as torch_default_collate
from .noun_parser import WordAugTokenizeWrapper
from PIL import Image

import torch
from sclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from sclip import tokenize


def collate(data):
    data = [(sample['image'], sample['text'][0], sample['text'][1], sample['text'][2])
            for sample in data]
    output = torch_default_collate(data)
    image, nouns, caption, pseudo_text_mask = output
    return {
        "image": image,
        "nouns": nouns,
        "caption": caption,
        "pseudo_text_mask": pseudo_text_mask,
    }


class NounNotEnoughError(Exception):
    pass


class TextPreprocess:
    def __init__(self, num_words, word_type="noun_phrase") -> None:
        self._tokenizer = _Tokenizer()
        self.num_words = num_words
        self.parser = WordAugTokenizeWrapper(word_type=word_type)

    def get_noun_mask(self, full_tokens, noun):
        masks = []
        noun_token = self._tokenizer.encode(noun)

        mask = torch.zeros(77)
        for start_index in range(77 - len(noun_token) + 1):
            is_same = True
            for offset in range(len(noun_token)):
                if full_tokens[start_index+offset] != noun_token[offset]:
                    is_same = False
                    break
            if not is_same:
                continue

            for offset in range(len(noun_token)):
                mask[start_index+offset] = 1

        return mask

    def get_noun_masks(self, full_caption, noun_lists, all_nouns):
        full_tokens = tokenize(
            full_caption,
            context_length=77,
            truncate=True
        )[0].numpy()

        pseudo_label = torch.zeros(77) - 1
        for noun in all_nouns:
            unassigned = self.get_noun_mask(full_tokens, noun)
            pseudo_label = torch.where(unassigned == 1, 0, pseudo_label)

        for i, noun in enumerate(noun_lists):
            assign = self.get_noun_mask(full_tokens, noun)
            pseudo_label = torch.where(assign == 1, i+1, pseudo_label)

        return pseudo_label

    def __call__(self, caption):
        nouns = self.parser(caption)
        if len(nouns) < self.num_words:
            raise NounNotEnoughError()

        random.shuffle(nouns)
        selected_nouns = nouns[:self.num_words]
        pseudo_text_mask = self.get_noun_masks(caption, selected_nouns, nouns)
        return selected_nouns, caption, pseudo_text_mask.long()


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(config):
    dataset_train = build_dataset(config=config)
    us.dprint("successfully build train dataset")

    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=dist.get_rank(), seed=config.seed
    )
    data_loader_train = wds.WebLoader(
        dataset_train.batched(config.batch_size, collate, partial=False),
        batch_size=None,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn,
    )

    train_len = len(dataset_train)
    train_nbatches = max(
        1, train_len // (config.batch_size * dist.get_world_size()))
    data_loader_train = data_loader_train.with_epoch(
        train_nbatches).with_length(train_nbatches)

    return dataset_train, data_loader_train


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    if isinstance(exn, NounNotEnoughError):
        return True
    warnings.warn(repr(exn))
    return True


def is_extracted_webdataset(path):
    """Check if path is an extracted webdataset directory"""
    if not osp.isdir(path):
        return False

    # Check for numbered subdirectories (00000, 00001, etc.)
    subdirs = [d for d in os.listdir(path)
               if osp.isdir(osp.join(path, d)) and d.isdigit()]

    if not subdirs:
        return False

    # Check if first subdir has image and text files
    first_dir = osp.join(path, subdirs[0])
    files = os.listdir(first_dir)
    has_image = any(f.endswith(('.jpg', '.png', '.jpeg')) for f in files)
    has_text = any(f.endswith(('.txt', '.text')) for f in files)

    return has_image and has_text


class ExtractedWebDataset(torch.utils.data.IterableDataset):
    """Dataset for extracted webdataset directories"""

    def __init__(self, dir_path, img_transform, text_transform):
        self.dir_path = dir_path
        self.img_transform = img_transform
        self.text_transform = text_transform

        # Collect all sample directories
        self.sample_dirs = sorted([
            osp.join(dir_path, d)
            for d in os.listdir(dir_path)
            if osp.isdir(osp.join(dir_path, d)) and d.isdigit()
        ])

        self._length = len(self.sample_dirs)

    def __len__(self):
        return self._length

    def __iter__(self):
        while True:
            for sample_dir in self.sample_dirs:
                try:
                    files = os.listdir(sample_dir)

                    # Find image and text files
                    image_file = None
                    text_file = None

                    for f in files:
                        if f.endswith(('.jpg', '.png', '.jpeg')) and image_file is None:
                            image_file = f
                        elif f.endswith(('.txt', '.text')) and text_file is None:
                            text_file = f

                    if not image_file or not text_file:
                        continue

                    # Load image
                    img_path = osp.join(sample_dir, image_file)
                    image = Image.open(img_path).convert('RGB')

                    # Load text
                    text_path = osp.join(sample_dir, text_file)
                    with open(text_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()

                    # Apply transforms
                    image = self.img_transform(image)
                    nouns, caption, pseudo_text_mask = self.text_transform(caption)

                    yield {
                        'image': image,
                        'text': (nouns, caption, pseudo_text_mask)
                    }

                except (NounNotEnoughError, Exception) as e:
                    if not isinstance(e, NounNotEnoughError):
                        warnings.warn(repr(e))
                    continue


def build_dataset(config):
    """
    Args:
        config: CONFIG.data (CONFIG = global config)
    """
    img_transform = build_img_transform(config.img_aug)
    text_transform = TextPreprocess(
        num_words=config.num_words, word_type=config.word_type)
    split = "train"
    dataset_type = None
    tar_file_list = []
    extracted_dirs = []
    total_length = 0

    for ds in config.dataset[split]:
        ds_meta = config.dataset.meta[ds]
        if dataset_type is None:
            dataset_type = ds_meta.type
        else:
            assert dataset_type == ds_meta.type, "All datasets must be of the same type"

        prefix = ds_meta.prefix
        path = ds_meta.path
        length = ds_meta.length

        print(f"\n=== Dataset: {ds} ===")
        print(f"Path: {path}")
        print(f"Prefix: {prefix}")

        expanded_paths = list(braceexpand(osp.join(path, prefix)))
        print(f"Expanded paths: {expanded_paths}")

        for expanded_path in expanded_paths:
            print(f"\nChecking: {expanded_path}")
            print(f"  Exists: {osp.exists(expanded_path)}")

            if not osp.exists(expanded_path):
                print(f"  -> Path does not exist, skipping")
                continue

            print(f"  Is file: {osp.isfile(expanded_path)}")
            print(f"  Is dir: {osp.isdir(expanded_path)}")

            # Check if it's an extracted webdataset directory
            if is_extracted_webdataset(expanded_path):
                extracted_dirs.append(expanded_path)
                print(f"  -> Found extracted webdataset: {expanded_path}")
            # Check if it's a tar file
            elif expanded_path.endswith('.tar') and osp.isfile(expanded_path):
                tar_file_list.append(expanded_path)
                print(f"  -> Found tar file")
            # Check for tar files in directory
            elif osp.isdir(expanded_path):
                found_tars = glob.glob(osp.join(expanded_path, '*.tar'))
                if found_tars:
                    tar_file_list.extend(found_tars)
                    print(f"  -> Found {len(found_tars)} tar files in {expanded_path}")
                else:
                    print(f"  -> Is directory but no tar files found")
                    # Debug: list contents
                    try:
                        contents = os.listdir(expanded_path)
                        print(f"     Directory contents (first 10): {contents[:10]}")
                    except Exception as e:
                        print(f"     Could not list directory: {e}")

        total_length += length

    print(f"Found {len(tar_file_list)} tar files, {len(extracted_dirs)} extracted directories")

    # Build dataset based on what we found
    if tar_file_list and not extracted_dirs:
        # Use tar files
        dataset = (
            wds.WebDataset(tar_file_list, repeat=True, handler=warn_and_continue)
            .shuffle(40000)
            .decode("pil", handler=warn_and_continue)
            .rename(
                image="jpg;png;jpeg",
                text="text;txt",
                caption="text;txt",
                keep=False,
                handler=warn_and_continue,
            )
            .map_dict(image=img_transform, text=text_transform, handler=warn_and_continue)
            .with_length(total_length)
        )
    elif extracted_dirs and not tar_file_list:
        # Use extracted directories (combine multiple if needed)
        if len(extracted_dirs) == 1:
            dataset = ExtractedWebDataset(extracted_dirs[0], img_transform, text_transform)
        else:
            # Combine multiple extracted directories
            combined_dir = extracted_dirs[0]
            dataset = ExtractedWebDataset(combined_dir, img_transform, text_transform)
    else:
        raise ValueError(f"No tar files or extracted directories found. tar_files={len(tar_file_list)}, extracted_dirs={len(extracted_dirs)}")

    # Add length if dataset supports it
    if hasattr(dataset, 'with_length'):
        dataset = dataset.with_length(total_length)

    return dataset


def build_img_transform(config):
    if not config.deit_aug:
        transform = T.Compose(
            [
                T.RandomResizedCrop(config.img_size, scale=config.img_scale),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=us.DEFAULT_MEAN, std=us.DEFAULT_STD),
            ]
        )
    else:
        # deit_aug
        transform = create_transform(
            input_size=config.img_size,
            is_training=True,
            color_jitter=config.color_jitter if config.color_jitter > 0 else None,
            auto_augment=config.auto_augment if config.auto_augment != "none" else None,
            re_prob=config.re_prob,
            re_mode=config.re_mode,
            re_count=config.re_count,
        )

    return transform
