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


def build_loader(config, max_datasets_to_check=None):
    dataset_train = build_dataset(config=config, max_datasets_to_check=max_datasets_to_check)
    us.dprint("successfully build train dataset")

    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=dist.get_rank(), seed=config.seed
    )
    loader = wds.WebLoader(
        dataset_train,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    train_len = len(dataset_train)
    train_nbatches = max(
        1, train_len // (config.batch_size * dist.get_world_size()))
    loader = loader.with_epoch(train_nbatches).with_length(train_nbatches)

    return dataset_train, loader


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    if isinstance(exn, NounNotEnoughError):
        return True
    warnings.warn(repr(exn))
    return True


def is_flat_webdataset(path):
    """Check if path is a flat webdataset numbered subdirectory with paired image-text files

    This function checks if the given path IS a numbered directory (like 00000) containing
    image and text files DIRECTLY (not in nested subdirectories).

    Structure: /base/00000/000000000.jpg, /base/00000/000000000.txt, etc.
    When called with path=/base/00000, this should detect the flat structure.
    """
    print(f"    [is_flat_webdataset] Checking: {path}")

    if not osp.isdir(path):
        print(f"    [is_flat_webdataset] Not a directory, skipping")
        return False

    # Get the directory name to check if it's a numbered directory (like 00000, 00001, etc.)
    dir_name = osp.basename(path.rstrip('/'))
    is_numbered_dir = dir_name.isdigit() and len(dir_name) == 5  # e.g., 00000, 00001

    print(f"    [is_flat_webdataset] Directory name: '{dir_name}'")
    print(f"    [is_flat_webdataset] Is numbered directory: {is_numbered_dir}")

    if not is_numbered_dir:
        print(f"    [is_flat_webdataset] Not a numbered directory")
        return False

    # Get all files in this directory
    all_items = os.listdir(path)

    # Check if there are any subdirectories (nested structure)
    has_subdirs = any(osp.isdir(osp.join(path, item)) for item in all_items)

    print(f"    [is_flat_webdataset] Total items: {len(all_items)}")
    print(f"    [is_flat_webdataset] Has subdirectories: {has_subdirs}")

    if has_subdirs:
        print(f"    [is_flat_webdataset] Found subdirectories - not a flat structure")
        return False

    # Count image and text files directly in this directory
    files = [f for f in all_items if osp.isfile(osp.join(path, f))]
    image_files = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
    text_files = [f for f in files if f.endswith(('.txt', '.text'))]

    print(f"    [is_flat_webdataset] Total files: {len(files)}")
    print(f"    [is_flat_webdataset] Image files: {len(image_files)}, Text files: {len(text_files)}")

    # For flat structure: we should have multiple image AND text files
    # (as opposed to extracted structure which might have just one of each)
    has_multiple_images = len(image_files) > 1
    has_multiple_texts = len(text_files) > 1

    result = has_multiple_images and has_multiple_texts
    print(f"    [is_flat_webdataset] Result: {result}")
    return result


def is_extracted_webdataset(path):
    """Check if path is an extracted webdataset directory"""
    print(f"    [is_extracted_webdataset] Checking: {path}")

    if not osp.isdir(path):
        print(f"    [is_extracted_webdataset] Not a directory, skipping")
        return False

    # Check for numbered subdirectories (00000, 00001, etc.)
    all_items = os.listdir(path)
    subdirs = [d for d in all_items
               if osp.isdir(osp.join(path, d)) and d.isdigit()]

    print(f"    [is_extracted_webdataset] Total items: {len(all_items)}")
    print(f"    [is_extracted_webdataset] Numeric subdirs found: {len(subdirs)}")

    if not subdirs:
        print(f"    [is_extracted_webdataset] No numeric subdirectories found")
        return False

    # Check if first subdir has image and text files
    first_dir = osp.join(path, subdirs[0])
    files = os.listdir(first_dir)
    has_image = any(f.endswith(('.jpg', '.png', '.jpeg')) for f in files)
    has_text = any(f.endswith(('.txt', '.text')) for f in files)

    print(f"    [is_extracted_webdataset] First subdir ({subdirs[0]}): {len(files)} files")
    print(f"    [is_extracted_webdataset] Files: {files}")
    print(f"    [is_extracted_webdataset] Has image: {has_image}, Has text: {has_text}")

    result = has_image and has_text
    print(f"    [is_extracted_webdataset] Result: {result}")
    return result


class FlatWebDataset(torch.utils.data.IterableDataset):
    """Dataset for flat webdataset directories with paired image-text files

    Can handle two scenarios:
    1. dir_path is a numbered subdirectory like /base/00000 (direct flat structure)
    2. dir_path is a base directory containing numbered subdirectories (legacy)

    In either case, files are grouped by base name (without extension).
    """

    def __init__(self, dir_path, img_transform, text_transform):
        self.dir_path = dir_path
        self.img_transform = img_transform
        self.text_transform = text_transform

        # Check if dir_path itself is a numbered directory (flat structure)
        dir_name = osp.basename(dir_path.rstrip('/'))
        is_numbered_dir = dir_name.isdigit() and len(dir_name) >= 5

        if is_numbered_dir:
            # Path itself is a numbered directory - use it directly
            print(f"[FlatWebDataset] Detected numbered directory: {dir_path}")
            self.subdirs = [dir_path]
        else:
            # Path is a base directory - collect numbered subdirectories
            print(f"[FlatWebDataset] Detecting numbered subdirectories in: {dir_path}")
            self.subdirs = [
                osp.join(dir_path, d)
                for d in sorted(os.listdir(dir_path))
                if osp.isdir(osp.join(dir_path, d)) and d.isdigit()
            ]

        print(f"[FlatWebDataset] Processing {len(self.subdirs)} directories")

        # Build list of all samples (path, image_file, text_file)
        self.samples = []
        for subdir in self.subdirs:
            self.samples.extend(self._get_paired_samples(subdir))

        self._length = len(self.samples)
        print(f"[FlatWebDataset] Total samples: {self._length}")

    def _get_paired_samples(self, dir_path):
        """Find all paired image-text files in a directory.

        Returns list of tuples: (image_path, text_path)
        """
        files = os.listdir(dir_path)

        # Group files by base name (without extension)
        base_names = {}
        for f in files:
            # Split by last dot to separate extension
            if '.' in f:
                base = f.rsplit('.', 1)[0]
                ext = f.rsplit('.', 1)[1].lower()
            else:
                continue

            if base not in base_names:
                base_names[base] = {}
            base_names[base][ext] = f

        # Find pairs
        paired_samples = []
        for base, extensions in base_names.items():
            # Find image file
            image_file = None
            for img_ext in ['jpg', 'jpeg', 'png']:
                if img_ext in extensions:
                    image_file = extensions[img_ext]
                    break

            # Find text file
            text_file = None
            for txt_ext in ['txt', 'text']:
                if txt_ext in extensions:
                    text_file = extensions[txt_ext]
                    break

            # Add if both found
            if image_file and text_file:
                img_path = osp.join(dir_path, image_file)
                txt_path = osp.join(dir_path, text_file)
                paired_samples.append((img_path, txt_path))

        return paired_samples

    def __len__(self):
        return self._length

    def __iter__(self):
        while True:
            for img_path, txt_path in self.samples:
                try:
                    # Load image
                    image = Image.open(img_path).convert('RGB')

                    # Load text
                    with open(txt_path, 'r', encoding='utf-8') as f:
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


def build_dataset(config, max_datasets_to_check=None):
    """
    Args:
        config: CONFIG.data (CONFIG = global config)
        max_datasets_to_check (int, optional): Max number of datasets to check. Defaults to None.
    """
    img_transform = build_img_transform(config.img_aug)
    text_transform = TextPreprocess(
        num_words=config.num_words, word_type=config.word_type)
    split = "train"
    dataset_type = None
    tar_file_list = []
    extracted_dirs = []
    total_length = 0

    datasets_to_process = config.dataset[split]
    if max_datasets_to_check is not None:
        print(f"[INFO] Checking only the first {max_datasets_to_check} dataset(s).")
        datasets_to_process = datasets_to_process[:max_datasets_to_check]

    for ds in datasets_to_process:
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
        print(f"Dataset type: {ds_meta.type}")
        print(f"Expected length: {length}")

        # DEBUG: Check if path exists and list contents
        print(f"\n[DEBUG] Checking raw path: {path}")
        print(f"[DEBUG] Raw path exists: {osp.exists(path)}")
        if osp.exists(path):
            try:
                contents = os.listdir(path)
                print(f"[DEBUG] Path is directory: {osp.isdir(path)}")
                print(f"[DEBUG] Number of items in path: {len(contents)}")
                print(f"[DEBUG] First 15 items: {sorted(contents)[:15]}")
                # Check if any are numeric directories (extracted webdataset)
                numeric_dirs = [c for c in contents if osp.isdir(osp.join(path, c)) and c.isdigit()]
                print(f"[DEBUG] Numeric directories found: {len(numeric_dirs)}")
                if numeric_dirs:
                    print(f"[DEBUG] Numeric dir examples: {sorted(numeric_dirs)[:5]}")
            except Exception as e:
                print(f"[DEBUG] Error listing path: {e}")

        expanded_paths = list(braceexpand(osp.join(path, prefix)))
        if max_datasets_to_check is not None:
            expanded_paths = expanded_paths[:max_datasets_to_check]
        print(f"\nExpanded paths ({len(expanded_paths)} total): {expanded_paths}")

        for i, expanded_path in enumerate(expanded_paths):
            print(f"\n[{i+1}/{len(expanded_paths)}] Checking: {expanded_path}")
            print(f"  Exists: {osp.exists(expanded_path)}")

            if not osp.exists(expanded_path):
                print(f"  -> Path does not exist, skipping")
                continue

            print(f"  Is file: {osp.isfile(expanded_path)}")
            print(f"  Is dir: {osp.isdir(expanded_path)}")

            # Check if it's a flat webdataset directory first (more specific)
            if is_flat_webdataset(expanded_path):
                extracted_dirs.append((expanded_path, 'flat'))
                print(f"  ✓ Found flat webdataset: {expanded_path}")
                # Count subdirectories
                subdirs = [d for d in os.listdir(expanded_path) if osp.isdir(osp.join(expanded_path, d)) and d.isdigit()]
                print(f"    - Subdirectories: {len(subdirs)}")
                print(f"    - Examples: {sorted(subdirs)[:5]}")
            # Then check if it's an extracted webdataset directory
            elif is_extracted_webdataset(expanded_path):
                extracted_dirs.append((expanded_path, 'extracted'))
                print(f"  ✓ Found extracted webdataset: {expanded_path}")
                # Count subdirectories
                subdirs = [d for d in os.listdir(expanded_path) if osp.isdir(osp.join(expanded_path, d)) and d.isdigit()]
                print(f"    - Subdirectories: {len(subdirs)}")
                print(f"    - Examples: {sorted(subdirs)[:5]}")
            # Check if it's a tar file
            elif expanded_path.endswith('.tar') and osp.isfile(expanded_path):
                tar_file_list.append(expanded_path)
                print(f"  ✓ Found tar file")
            # Check for tar files in directory
            elif osp.isdir(expanded_path):
                found_tars = glob.glob(osp.join(expanded_path, '*.tar'))
                if found_tars:
                    tar_file_list.extend(found_tars)
                    print(f"  ✓ Found {len(found_tars)} tar files in {expanded_path}")
                else:
                    print(f"  ! Is directory but no tar files found")
                    # Debug: list contents
                    try:
                        contents = os.listdir(expanded_path)
                        print(f"    - Total items: {len(contents)}")
                        print(f"    - First 10 items: {sorted(contents)[:10]}")
                        # Check for numeric directories
                        numeric_dirs = [c for c in contents if osp.isdir(osp.join(expanded_path, c)) and c.isdigit()]
                        print(f"    - Numeric directories: {len(numeric_dirs)}")
                        if numeric_dirs:
                            print(f"    - Examples: {sorted(numeric_dirs)[:5]}")
                            # Check first numeric dir
                            first_numeric = osp.join(expanded_path, sorted(numeric_dirs)[0])
                            first_contents = os.listdir(first_numeric)
                            print(f"    - Contents of {sorted(numeric_dirs)[0]}: {first_contents}")
                    except Exception as e:
                        print(f"    - Could not list directory: {e}")

        total_length += length

    print(f"\n" + "="*60)
    print(f"SUMMARY:")
    print(f"  Tar files found: {len(tar_file_list)}")
    print(f"  Extracted directories found: {len(extracted_dirs)}")
    print(f"  Total expected length: {total_length}")
    print(f"="*60)

    # Build dataset based on what we found
    if tar_file_list and not extracted_dirs:
        # Use tar files
        print(f"\n[INFO] Using {len(tar_file_list)} tar files")
        print(f"[INFO] Tar files: {tar_file_list}")
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
        print(f"\n[INFO] Using {len(extracted_dirs)} extracted directories")
        print(f"[INFO] Extracted dirs: {extracted_dirs}")
        if len(extracted_dirs) == 1:
            dir_path, dir_type = extracted_dirs[0]
            if dir_type == 'flat':
                dataset = FlatWebDataset(dir_path, img_transform, text_transform)
            else:
                dataset = ExtractedWebDataset(dir_path, img_transform, text_transform)
        else:
            # Combine multiple extracted directories
            dir_path, dir_type = extracted_dirs[0]
            print(f"[WARNING] Multiple extracted directories found, using only: {dir_path}")
            if dir_type == 'flat':
                dataset = FlatWebDataset(dir_path, img_transform, text_transform)
            else:
                dataset = ExtractedWebDataset(dir_path, img_transform, text_transform)
    else:
        print(f"\n[ERROR] No tar files or extracted directories found!")
        print(f"[ERROR] tar_files={len(tar_file_list)}, extracted_dirs={len(extracted_dirs)}")
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
