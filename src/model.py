import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, out_size, hidden_dim=512, dropout_prob=0.5, num_classes=2):
        super().__init__()
        self.dense1 = nn.Linear(out_size, hidden_dim * 2)
        self.norm_0 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_prob)

        self.out_proj = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.relu(self.norm_0(x))
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class VisualBert(nn.Module):
    def __init__(self, args):
        super(VisualBert, self).__init__()
        model_name = args.model_name
        num_classes = args.num_classes
        drop_rate = args.drop_rate
        use_clf_head = args.use_clf_head

        ref_model = transformers.BertModel.from_pretrained(model_name)
        self.model = transformers.VisualBertModel(transformers.VisualBertConfig(
            hidden_size=ref_model.config.hidden_size,
            num_attention_heads=ref_model.config.num_attention_heads,
            intermediate_size=ref_model.config.intermediate_size,
            num_hidden_layers=ref_model.config.num_hidden_layers,
            vocab_size=ref_model.config.vocab_size,
            visual_embedding_dim=2048))
        self.model.load_state_dict(ref_model.state_dict(), strict=False)

        if not use_clf_head:
            self.fc = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(self.model.config.hidden_size, num_classes)
            )
        else:
            self.fc = ClassificationHead(out_size=self.model.config.hidden_size,
                                         hidden_dim=1024,
                                         dropout_prob=drop_rate,
                                         num_classes=num_classes)

    def forward(self, x):
        out = self.model(**x)[1]
        out = self.fc(out)
        return out


def get_model(args):
    return eval(args.type)(args)


def get_loss(args):
    args = args.copy()
    if "type" not in args:
        args["type"] = "CrossEntropyLoss"
    return eval("nn." + args.pop("type"))(**args)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    args = {
        'type': 'VisualBert',
        'model_name': 'hfl/chinese-roberta-wwm-ext-large',
        'num_classes': 2,
        'drop_rate': 0.5
    }
    args = OmegaConf.create(args)
    print(args.type)
    model = get_model(args)
