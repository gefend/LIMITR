import re
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random
import torch
import torch.utils.data as data

from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer, BertTokenizer
from LIMITR.constants import *
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self,cfg, split="train", transform=None):
        super().__init__()

        self.cfg = cfg
        self.transform = transform
        self.max_words = self.cfg.data.text.word_num
        self.imsize = self.cfg.data.image.imsize

        if split == 'test':
            self.df = pd.read_csv(MIMIC_CXR_TEST_CSV)
        elif split == 'train':
            self.df = pd.read_csv(MIMIC_CXR_TRAIN_CSV)
        elif split == 'valid':
            self.df = pd.read_csv(MIMIC_CXR_VALID_CSV)

        self.filenames, self.path2sent,self.pathF2pathL = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            self.cfg.model.text.bert_type)

    def load_text_data(self, split):
        # get study to captions mapping
        if split == 'test':
            filepath = os.path.join(
                BASE_DIR, "../../data/captions_test.pickle")
            filepath_FL = os.path.join(
                BASE_DIR, "../../data/FL_test.pickle")
        elif split == 'train':
            filepath = os.path.join(
                BASE_DIR, "../../data/captions.pickle")
            filepath_FL = os.path.join(
                BASE_DIR, "../../data/FL_train.pickle")
        elif split == 'valid':
            filepath = os.path.join(
                BASE_DIR, "../../data/captions_valid.pickle")
            filepath_FL = os.path.join(
                BASE_DIR, "../../data/FL_valid.pickle")

        if not os.path.isfile(filepath_FL):
            print(
                f"Caption file {filepath_FL} does not exit. Creating captions...")
            path2sent, pathF2pathL = self.create_path_2_sent_mapping()
            with open(filepath_FL, "wb") as f:
                pickle.dump(pathF2pathL, f, protocol=2)
                print("Save to: ", filepath_FL)
        else:
            with open(filepath_FL, "rb") as f:
                pathF2pathL = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            path = getattr(row, MIMIC_CXR_PATH_COL_F)
            filenames.append(path)

        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent, pathF2pathL = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        return filenames, path2sent, pathF2pathL

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        pathF2pathL = {}
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            captions = row[MIMIC_CXR_REPORT_COL]
            # use space instead of newline
            captions = captions.replace("\n", " ")
            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]
            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            path2sent[row[MIMIC_CXR_PATH_COL_F]] = study_sent
            pathF2pathL[row[MIMIC_CXR_PATH_COL_F]] = row[MIMIC_CXR_PATH_COL_L]

        return path2sent, pathF2pathL

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def get_imgs(self, img_path, scale, transform=None):
        my_path = DATA_BASE_DIR
        joined_path = os.path.join(my_path, img_path.strip("/"))
        x = cv2.imread(str(joined_path), 0)
        # tranform images
        x = self.resize_img(x, scale)
        img = Image.fromarray(x).convert("RGB")
        if transform is not None:
            img = transform(img)
        return img

    def resize_img(self,img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img

    def __getitem__(self, index):
        key = self.filenames[index]
        pathL = self.pathF2pathL[key]
        caps, cap_len = self.get_caption(key)
        imgs_F = self.get_imgs(key, self.imsize, self.transform)
        if pathL != pathL:
            imgs_L = torch.zeros((3, 224, 224))
        else:
            imgs_L = self.get_imgs(pathL, self.imsize, self.transform)
        return imgs_F, imgs_L, caps, cap_len, key


def multimodal_collate_fn(batch):
    """sort sequence"""
    imgs_F,imgs_L, cap_len, ids, tokens, attention = [], [], [], [], [],[]
    path = []
    for b in batch:
        img_F,img_L, cap, cap_l, p = b
        imgs_F.append(img_F)
        imgs_L.append(img_L)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs_F = torch.stack(imgs_F)
    imgs_L = torch.stack(imgs_L)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)

    path = np.array(path)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs_F": imgs_F[sorted_cap_indices],
        "imgs_L": imgs_L[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices]
    }
    return return_dict

