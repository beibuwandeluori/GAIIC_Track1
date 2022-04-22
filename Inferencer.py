import json
import pandas as pd
import os
from Solver import *

if __name__ == '__main__':
    gpus = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    save_root = '/data1/chenby/py_project/GAIIC_Track1/logs/results/baseline_clf'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    trainer = pl.Trainer(gpus=len(gpus.split(",")))

    model = Model.load_from_checkpoint("logs/text/baseline_clf/checkpoints/last.ckpt")
    outputs = trainer.predict(model)
    outputs = torch.cat(outputs)
    print(f'text:{outputs.shape}')

    df_test_text = model.ds_test.df
    pred_text = outputs.argmax(1).cpu().numpy()
    df_test_text["pred"] = pred_text

    model = Model.load_from_checkpoint("logs/tag/baseline_clf/checkpoints/last.ckpt")
    outputs = trainer.predict(model)
    outputs = torch.cat(outputs)
    print(f'tag:{outputs.shape}')

    df_test_tag = model.ds_test.df
    pred_tag = outputs.argmax(1).cpu().numpy()
    df_test_tag["pred"] = pred_tag

    name2id = pd.read_csv("./data/name_to_id.csv")

    with open(f"{save_root}/submission.txt", "w") as f:
        for idx, (img_name, img_idx) in name2id.iterrows():
            pred_text = df_test_text[df_test_text.img_idx == img_idx]
            pred_tag = df_test_tag[df_test_tag.img_idx == img_idx]
            row = {"img_name": img_name,
                   "match": {"图文": int(pred_text.pred.iloc[0]),
                             **{k: int(v) for k, v in zip(pred_tag["keys"], pred_tag.pred)}
                             }
                   }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
