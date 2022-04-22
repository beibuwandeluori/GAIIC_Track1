import os
import json
import pandas as pd
import numpy as np

with open("./data/train/attr_to_attrvals.json") as f:
    atts = json.load(f)
    for k in atts:
        att = {}
        for i, vi in enumerate(sorted(atts[k])):
            for vii in vi.split("="):
                att[vii] = i
        atts[k] = att


def get_v(t, k):
    if (k == "裤门襟" and "裤" not in t) or (k == "闭合方式" and ("鞋" not in t and "靴" not in t)):
        return ""
    mv = ""
    for v in atts[k]:
        if v in t and len(v) > len(mv):
            mv = v
    return mv


def v2id(k, v):
    if v == "":
        return -1
    return atts[k][v]


def train():
    text_df = []
    tag_df = []
    feat = []
    with open("./data/train/train_fine.txt") as f:
        for l in f:
            l = json.loads(l.strip())
            img_idx = len(feat)
            feat.append(l.pop("feature"))
            text_df.append([l["title"], img_idx, l["match"]["图文"], [], []])
            for k in l["key_attr"]:
                tag_df.append([l["key_attr"][k], img_idx, l["match"][k], k])
                text_df[-1][-2].append(k)
                text_df[-1][-1].append(l["key_attr"][k])

    with open("./data/train/train_coarse.txt") as f:
        for l in f:
            l = json.loads(l.strip())
            img_idx = len(feat)
            feat.append(l.pop("feature"))
            text_df.append([l["title"], img_idx, l["match"]["图文"], [], []])
            if l["match"]["图文"]:
                for k in atts:
                    v = get_v(l["title"], k)
                    if v:
                        tag_df.append([v, img_idx, 1, k])
                        text_df[-1][-2].append(k)
                        text_df[-1][-1].append(v)

    text_df = pd.DataFrame(text_df)
    text_df.columns = ["text", "img_idx", "label", "keys", "values"]
    text_df.to_csv("./data/train/train_text.csv", index=False)
    tag_df = pd.DataFrame(tag_df)
    tag_df.columns = ["text", "img_idx", "label", "keys"]
    tag_df.to_csv("./data/train/train_tag.csv", index=False)
    np.save("./data/train/train.npy", np.stack(feat))


def test_A():
    text_df = []
    tag_df = []
    feat = []
    name2id = []
    with open("./data/test_A/preliminary_testA.txt") as f:
        for l in f:
            l = json.loads(l.strip())
            img_idx = len(feat)
            feat.append(l.pop("feature"))
            name2id.append([l['img_name'], img_idx])
            text_df.append([l["title"], img_idx, 0, [], []])
            for k in l["query"]:
                if k == "图文": continue
                v = get_v(l["title"], k)
                tag_df.append([v, img_idx, 0, k])
                text_df[-1][-2].append(k)
                text_df[-1][-1].append(v)

    text_df = pd.DataFrame(text_df)
    text_df.columns = ["text", "img_idx", "label", "keys", "values"]
    text_df.to_csv("./data/test_A/testA_text.csv", index=False)
    tag_df = pd.DataFrame(tag_df)
    tag_df.columns = ["text", "img_idx", "label", "keys"]
    tag_df.to_csv("./data/test_A/testA_tag.csv", index=False)
    name2id = pd.DataFrame(name2id)
    name2id.columns = ["img_name", "img_idx"]
    name2id.to_csv("./data/test_A/name_to_id.csv", index=False)
    np.save("./data/test_A/testA.npy", np.stack(feat))


def test_B():
    text_df = []
    tag_df = []
    feat = []
    name2id = []
    with open("./data/test_B/preliminary_testB.txt") as f:
        for l in f:
            l = json.loads(l.strip())
            img_idx = len(feat)
            feat.append(l.pop("feature"))
            name2id.append([l['img_name'], img_idx])
            text_df.append([l["title"], img_idx, 0, [], []])
            for k in l["query"]:
                if k == "图文": continue
                v = get_v(l["title"], k)
                tag_df.append([v, img_idx, 0, k])
                text_df[-1][-2].append(k)
                text_df[-1][-1].append(v)

    text_df = pd.DataFrame(text_df)
    text_df.columns = ["text", "img_idx", "label", "keys", "values"]
    text_df.to_csv("./data/test_B/testB_text.csv", index=False)
    tag_df = pd.DataFrame(tag_df)
    tag_df.columns = ["text", "img_idx", "label", "keys"]
    tag_df.to_csv("./data/test_B/testB_tag.csv", index=False)
    name2id = pd.DataFrame(name2id)
    name2id.columns = ["img_name", "img_idx"]
    name2id.to_csv("./data/test_B/name_to_id.csv", index=False)
    np.save("./data/test_B/testB.npy", np.stack(feat))


if __name__ == '__main__':
    # train()
    # test_A()
    test_B()


