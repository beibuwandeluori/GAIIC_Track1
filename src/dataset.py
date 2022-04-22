import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import random

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data(args):
    data_type = args.type
    fold = args.get("fold", -1)
    batch_size = args.batch_size
    max_length = args.max_length

    X = np.load("./data/train/train.npy")
    X_test = np.load("./data/test_A/testA.npy")
    split = KFold(5, random_state=0, shuffle=True)
    train_idx, valid_idx = list(split.split(X))[fold]
    if fold == -1:
        train_idx = np.concatenate([train_idx, valid_idx])

    if data_type == "TextData":
        neg_ratio = args.get("neg_ratio", 0)
        shuffle_ratio = args.get("shuffle_ratio", 0)
        extend_ratio = args.get("extend_ratio", 0)
        tokenizer = BertTokenizer.from_pretrained(args.model_name)

        DataClass = TextData

        df = pd.read_csv("./data/train/train_text.csv")
        df['keys'] = df['keys'].apply(eval)
        df['values'] = df['values'].apply(eval)

        df_test = pd.read_csv("./data/test_A/testA_text.csv")
        df_test['keys'] = df_test['keys'].apply(eval)
        df_test['values'] = df_test['values'].apply(eval)

        train_data = {
            "df": df.loc[train_idx].reset_index(drop=True),
            "X": X,
            "neg_ratio": neg_ratio,
            "shuffle_ratio": shuffle_ratio,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
        valid_data = {
            "df": df.loc[valid_idx].reset_index(drop=True),
            "X": X,
            "neg_ratio": 0,
            "shuffle_ratio": 0,
            "extend_neg": extend_ratio,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
        test_data = {
            "df": df_test,
            "X": X_test,
            "neg_ratio": 0, "shuffle_ratio": 0, "extend_neg": 0,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
    elif data_type == "TagData":
        neg_ratio = args.get("neg_ratio", 0)
        extend_ratio = args.get("extend_ratio", 0)
        tokenizer = BertTokenizer.from_pretrained(args.model_name)

        DataClass = TagData
        df = pd.read_csv("./data/train/train_tag.csv")
        df_test = pd.read_csv("./data/test_A/testA_tag.csv")

        train_data = {
            "df": df.loc[df.img_idx.isin(train_idx)].reset_index(drop=True),
            "X": X,
            "neg_ratio": neg_ratio,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
        valid_data = {
            "df": df.loc[df.img_idx.isin(valid_idx)].reset_index(drop=True),
            "X": X,
            "neg_ratio": 0,
            "extend_neg": extend_ratio,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
        test_data = {
            "df": df_test,
            "X": X_test,
            "neg_ratio": 0,
            "extend_neg": 0,
            "tokenizer": tokenizer,
            "max_length": max_length
        }

    ds_train = DataClass(**train_data)
    ds_valid = DataClass(**valid_data)
    ds_test = DataClass(**test_data, with_label=False)

    def dl_train(shuffle=True, drop_last=True, num_workers=4):
        return DataLoader(ds_train, batch_size, shuffle=shuffle, drop_last=drop_last,
                          num_workers=num_workers, worker_init_fn=worker_init_fn)  # TODO 加上worker_init_fn

    def dl_valid(shuffle=False, num_workers=4):
        return DataLoader(ds_valid, batch_size, shuffle=shuffle, num_workers=num_workers)

    def dl_test(shuffle=False, num_workers=4):
        return DataLoader(ds_test, batch_size, shuffle=shuffle, num_workers=num_workers)

    return (ds_train, ds_valid, ds_test), (dl_train, dl_valid, dl_test)


def get_data_B(args):
    data_type = args.type
    fold = args.get("fold", -1)
    batch_size = args.batch_size
    max_length = args.max_length

    X = np.load("./data/train/train.npy")
    X_test = np.load("./data/test_B/testB.npy")
    split = KFold(5, random_state=0, shuffle=True)
    train_idx, valid_idx = list(split.split(X))[fold]
    if fold == -1:
        train_idx = np.concatenate([train_idx, valid_idx])

    if data_type == "TextData":
        neg_ratio = args.get("neg_ratio", 0)
        shuffle_ratio = args.get("shuffle_ratio", 0)
        extend_ratio = args.get("extend_ratio", 0)
        tokenizer = BertTokenizer.from_pretrained(args.model_name)

        DataClass = TextData

        df = pd.read_csv("./data/train/train_text.csv")
        df['keys'] = df['keys'].apply(eval)
        df['values'] = df['values'].apply(eval)

        df_test = pd.read_csv("./data/test_B/testB_text.csv")
        df_test['keys'] = df_test['keys'].apply(eval)
        df_test['values'] = df_test['values'].apply(eval)

        train_data = {
            "df": df.loc[train_idx].reset_index(drop=True),
            "X": X,
            "neg_ratio": neg_ratio,
            "shuffle_ratio": shuffle_ratio,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
        valid_data = {
            "df": df.loc[valid_idx].reset_index(drop=True),
            "X": X,
            "neg_ratio": 0,
            "shuffle_ratio": 0,
            "extend_neg": extend_ratio,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
        test_data = {
            "df": df_test,
            "X": X_test,
            "neg_ratio": 0, "shuffle_ratio": 0, "extend_neg": 0,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
    elif data_type == "TagData":
        neg_ratio = args.get("neg_ratio", 0)
        extend_ratio = args.get("extend_ratio", 0)
        tokenizer = BertTokenizer.from_pretrained(args.model_name)

        DataClass = TagData
        df = pd.read_csv("./data/train/train_tag.csv")
        df_test = pd.read_csv("./data/test_B/testB_tag.csv")

        train_data = {
            "df": df.loc[df.img_idx.isin(train_idx)].reset_index(drop=True),
            "X": X,
            "neg_ratio": neg_ratio,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
        valid_data = {
            "df": df.loc[df.img_idx.isin(valid_idx)].reset_index(drop=True),
            "X": X,
            "neg_ratio": 0,
            "extend_neg": extend_ratio,
            "tokenizer": tokenizer,
            "max_length": max_length
        }
        test_data = {
            "df": df_test,
            "X": X_test,
            "neg_ratio": 0,
            "extend_neg": 0,
            "tokenizer": tokenizer,
            "max_length": max_length
        }

    ds_train = DataClass(**train_data)
    ds_valid = DataClass(**valid_data)
    ds_test = DataClass(**test_data, with_label=False)

    def dl_train(shuffle=True, drop_last=True, num_workers=4):
        return DataLoader(ds_train, batch_size, shuffle=shuffle, drop_last=drop_last,
                          num_workers=num_workers, worker_init_fn=worker_init_fn)  # TODO 加上worker_init_fn

    def dl_valid(shuffle=False, num_workers=4):
        return DataLoader(ds_valid, batch_size, shuffle=shuffle, num_workers=num_workers)

    def dl_test(shuffle=False, num_workers=4):
        return DataLoader(ds_test, batch_size, shuffle=shuffle, num_workers=num_workers)

    return (ds_train, ds_valid, ds_test), (dl_train, dl_valid, dl_test)


def sample_another():
    with open("./data/train/attr_to_attrvals.json") as f:
        atts = json.load(f)
        for k in atts:
            att = {}
            for i, vi in enumerate(sorted(atts[k])):
                for vii in vi.split("="):
                    att[vii] = i  # 避免出现等价的
            atts[k] = att

    def sample(k, v):
        return np.random.choice([_ for _ in atts[k] if atts[k][_] != atts[k][v]])

    return sample


class TextData(Dataset):
    def __init__(self, df, X, neg_ratio, shuffle_ratio, tokenizer, max_length, extend_neg=0, with_label=True):
        self.df = df
        self.X = X
        self.neg_ratio = neg_ratio
        self.shuffle_ratio = shuffle_ratio
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extend_neg = extend_neg
        self.with_label = with_label
        self.sample = sample_another()
        if self.extend_neg != 0:
            self.extend()

    def extend(self):
        np.random.seed(0)
        pos_idxes = np.where(self.df.label == 1)[0]
        idxes = np.random.choice(pos_idxes, int(self.extend_neg * self.df.shape[0]))
        ext = []
        for idx in idxes:
            text, img_idx, label, keys, values = self.df.loc[idx]
            label = 0
            for key_idx in range(len(keys)):
                text = self.replace_text(text, keys[key_idx], values[key_idx])
            ext.append([text, img_idx, label, keys, values])
        ext = pd.DataFrame(ext)
        ext.columns = self.df.columns
        self.df = pd.concat([self.df, ext]).reset_index(drop=True)

    def replace_text(self, text, k, v):
        text = text.replace(v, self.sample(k, v))
        return text

    def shuffle_text(self, text, keys, values):
        idxes = [0]
        for v in values:
            idxes.append(text.index(v))
            idxes.append(idxes[-1] + len(v))
        idxes.append(len(text))
        idxes = sorted(idxes)
        texts = [text[idxes[i - 1]:idxes[i]] for i in range(1, len(idxes))]
        np.random.shuffle(texts)
        text = "".join(texts)
        return text

    def __getitem__(self, idx):
        text, img_idx, label, keys, values = self.df.loc[idx]

        x = self.X[img_idx].astype(np.float32)
        x = x.reshape(1, -1)
        x_token_type_ids = np.ones(x.shape[:-1], dtype=int)
        x_attention_mask = np.ones(x.shape[:-1], dtype=np.float32)

        if np.random.random() < self.shuffle_ratio and len(keys) > 0:
            text = self.shuffle_text(text, keys, values)
        if label == 1 and len(keys) > 0 and np.random.random() < self.neg_ratio:
            label = 0
            if np.random.random() < 0.5:  # TODO 以一定的概率只替换部分属性, hard samples
                replace_num = np.random.randint(len(keys)) + 1
                idxes = np.random.choice([_ for _ in range(len(keys))], size=replace_num)
                keys = np.array(keys)[idxes]
                values = np.array(values)[idxes]
            for key_idx in range(len(keys)):
                text = self.replace_text(text, keys[key_idx], values[key_idx])

        tok = self.tokenizer.encode_plus(text,
                                         max_length=self.max_length,
                                         truncation=True,
                                         padding="max_length")
        tok = {k: np.array(v) for k, v in tok.items()}
        tok.update({
            "visual_embeds": x,
            "visual_token_type_ids": x_token_type_ids,
            "visual_attention_mask": x_attention_mask,
        })

        if self.with_label:
            return tok, label
        else:
            return tok

    def __len__(self):
        return self.df.shape[0]


class TagData(Dataset):
    def __init__(self, df, X, neg_ratio, tokenizer, max_length, extend_neg=0, with_label=True):
        self.df = df
        self.X = X
        self.neg_ratio = neg_ratio
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extend_neg = extend_neg
        self.with_label = with_label
        self.sample = sample_another()
        if self.extend_neg != 0:
            self.extend()

    def extend(self):
        np.random.seed(0)
        pos_idxes = np.where(self.df.label == 1)[0]
        idxes = np.random.choice(pos_idxes, int(self.extend_neg * self.df.shape[0]))
        ext = []
        for idx in idxes:
            text, img_idx, label, keys = self.df.loc[idx]
            values = text
            label = 0
            text = self.replace_text(text, keys, values)
            ext.append([text, img_idx, label, keys])
        ext = pd.DataFrame(ext)
        ext.columns = self.df.columns
        self.df = pd.concat([self.df, ext]).reset_index(drop=True)

    def replace_text(self, text, k, v):
        text = self.sample(k, v)
        return text

    def __getitem__(self, idx):
        text, img_idx, label, keys = self.df.loc[idx]

        x = self.X[img_idx].astype(np.float32)
        x = x.reshape(1, -1)
        x_token_type_ids = np.ones(x.shape[:-1], dtype=int)
        x_attention_mask = np.ones(x.shape[:-1], dtype=np.float32)

        if label == 1 and len(keys) > 0 and np.random.random() < self.neg_ratio:
            values = text
            label = 0
            text = self.replace_text(text, keys, values)

        tok = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length")
        tok = {k: np.array(v) for k, v in tok.items()}
        tok.update({
            "visual_embeds": x,
            "visual_token_type_ids": x_token_type_ids,
            "visual_attention_mask": x_attention_mask,
        })

        if self.with_label:
            return tok, label
        else:
            return tok

    def __len__(self):
        return self.df.shape[0]


if __name__ == "__main__":
    sample_another()

    from omegaconf import OmegaConf

    args = OmegaConf.create(dict(
        type="TextData",
        max_length=64,
        batch_size=64,
        fold=0,
        neg_ratio=0.5,
        model_name="hfl/chinese-roberta-wwm-ext"
    ))
    (ds_train, ds_valid, ds_test), (dl_train, dl_valid, dl_test) = get_data(args)
    for i, data in enumerate(dl_train()):
        tok, label = data
        print(i, tok.keys(), label)
        print(tok['input_ids'].shape)
        for key in tok.keys():
            print(key, tok[key].shape, type(tok[key]), tok[key].device)
        # tok = {k: Variable(v.cuda(0)) for k, v in tok.items()}
        # for key in tok.keys():
        #     print(key, tok[key].shape, type(tok[key]), tok[key].device)

        if i == 10:
            break
