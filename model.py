import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RoFormerModel
from transformers.modeling_outputs import SequenceClassifierOutput

from chinesebert import ChineseBertConfig, ChineseBertModel


class CLSPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MeanPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        mean_tensor = hidden_states.mean(1)
        pooled_output = self.dense(mean_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RDropModelForSequenceClassification(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        model_type="roberta",
        pooler_type="cls",
        alpha=1.0,
        **kwargs
    ):
        super().__init__()
        if model_type == "chinesebert":
            config = ChineseBertConfig.from_pretrained(model_name_or_path)
            model_name_or_path = model_name_or_path.replace("junnyu", "ShannonAI")
            for k, v in kwargs.items():
                setattr(config, k, v)
            self.model = ChineseBertModel.from_pretrained(
                model_name_or_path, config=config
            )
        elif model_type in ["bert", "roberta", "wobert"]:
            self.model = BertModel.from_pretrained(model_name_or_path, **kwargs)
        elif model_type == "roformer":
            self.model = RoFormerModel.from_pretrained(model_name_or_path, **kwargs)
        else:
            raise ValueError(
                "model_type must be in chinesebert/bert/roberta/wobert/roformer"
            )

        self.config = self.model.config
        self.alpha = alpha
        self.num_labels = self.config.num_labels
        self.pooler = (
            CLSPooler(self.config) if pooler_type == "cls" else MeanPooler(self.config)
        )
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self._init_weights()

    def _init_weights(self):
        self.pooler.dense.weight.data.normal_(
            mean=0.0, std=self.config.initializer_range
        )
        self.pooler.dense.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pinyin_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rdrop=False,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None and rdrop:
            if input_ids is not None:
                input_ids = input_ids.repeat(2, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(2, 1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.repeat(2, 1)
            if pinyin_ids is not None:
                pinyin_ids = pinyin_ids.repeat(2, 1)

        default_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pinyin_ids=pinyin_ids,
        )
        if pinyin_ids is None:
            default_kwargs.pop("pinyin_ids", None)

        outputs = self.model(**default_kwargs)
        pooled_output = self.pooler(outputs[0])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                if rdrop:
                    labels = labels.repeat(2)
                loss = loss_fct(logits.reshape(-1, self.num_labels), labels.reshape(-1))

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                if rdrop:
                    labels = labels.repeat(2, 1)
                loss = loss_fct(logits, labels)
            else:
                raise ValueError(
                    "problem_type must be in single_label_classification/multi_label_classification"
                )

            if rdrop:
                pq = torch.log_softmax(logits.reshape(-1, self.num_labels), dim=-1)
                p, q = pq.chunk(2, dim=0)
                pq_tec = torch.softmax(logits.reshape(-1, self.num_labels), dim=-1)
                p_tec, q_tec = pq_tec.chunk(2, dim=0)
                kl_loss = F.kl_div(p, q_tec, reduction="none").sum()
                reverse_kl_loss = F.kl_div(q, p_tec, reduction="none").sum()
                loss += self.alpha * (kl_loss + reverse_kl_loss) / 4.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
