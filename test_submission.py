import json
import pandas as pd
from torch.autograd import Variable
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import os

from src.dataset import get_data, get_data_B
from src.model import get_model


def predict(models, token, weights=None, activation=torch.nn.Softmax(dim=1)):
    if weights is None:
        weights = [1.0 / len(models) for _ in range(len(models))]
    with torch.no_grad():
        outputs = weights[0] * activation(models[0](token))
        for i in range(1, len(models)):
            outputs += weights[i] * activation(models[i](token))

    return outputs


def load_model(model_paths, config_file='./logs/baseline_tag.yaml'):
    args = OmegaConf.load(config_file)
    models = []
    for i, model_path in enumerate(model_paths):
        model = get_model(args.model)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.cuda(0)
        model.eval()
        models.append(model)
        print(f'Load {i} model in {model_path}')

    # (_, _, ds_test), (_, _, test_loader) = get_data(args.data)
    (_, _, ds_test), (_, _, test_loader) = get_data_B(args.data)
    test_loader = test_loader()

    return models, test_loader, ds_test.df


def test_model(models, test_loader, weights=None, activation=torch.nn.Softmax(dim=1)):
    outputs = []
    for i, token in enumerate(tqdm(test_loader)):
        token = {k: Variable(v.cuda(0)) for k, v in token.items()}
        y_pred = predict(models, token, weights=weights, activation=activation)

        outputs.append(y_pred)

    return outputs


if __name__ == '__main__':
    gpus = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    save_root = '/data1/chenby/py_project/GAIIC_Track1/output/results_B/chinese-roberta-wwm-ext_5folds'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    model_paths_text = ['./output/text/weights/baseline/fold0/15_acc0.9638.pth',
                        './output/text/weights/baseline/fold1/15_acc0.9665.pth',
                        './output/text/weights/baseline/fold2/7_acc0.9564.pth',
                        './output/text/weights/baseline/fold3/18_acc0.9593.pth',
                        './output/text/weights/baseline/fold4/17_acc0.9660.pth']
    models, test_loader_text, df_test_text = load_model(model_paths=model_paths_text,
                                                        config_file='./logs/baseline_text.yaml')
    outputs = test_model(models, test_loader_text)
    outputs = torch.cat(outputs)
    print(f'text:{outputs.shape}')

    pred_text = outputs.argmax(1).cpu().numpy()
    df_test_text["pred"] = pred_text

    model_paths_tag = ['./output/tag/weights/baseline/fold0/15_acc0.9300.pth',
                       './output/tag/weights/baseline/fold1/17_acc0.9278.pth',
                       './output/tag/weights/baseline/fold2/17_acc0.9308.pth',
                       './output/tag/weights/baseline/fold3/15_acc0.9283.pth',
                       './output/tag/weights/baseline/fold4/15_acc0.9316.pth']
    models, test_tag_loader, df_test_tag = load_model(model_paths=model_paths_tag,
                                                      config_file='./logs/baseline_tag.yaml')
    outputs = test_model(models, test_tag_loader)
    outputs = torch.cat(outputs)
    print(f'tag:{outputs.shape}')

    pred_tag = outputs.argmax(1).cpu().numpy()
    df_test_tag["pred"] = pred_tag

    name2id = pd.read_csv("./data/test_B/name_to_id.csv")

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
