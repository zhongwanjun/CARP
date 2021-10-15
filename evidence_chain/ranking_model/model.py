from transformers import  RobertaForSequenceClassification
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import CrossEntropyLoss
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class RobertaForSequenceClassificationEC(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.num_labels = 1
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        a_input_ids=None,
        a_attention_mask=None,
        a_token_type_ids=None,
        pos_b_input_ids=None,
        pos_b_attention_mask=None,
        pos_b_token_type_ids=None,
        neg_b_input_ids=None,
        neg_b_attention_mask=None,
        neg_b_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        a_outputs = self.roberta(
            a_input_ids,
            attention_mask=a_attention_mask,
            token_type_ids=a_token_type_ids,
            position_ids=position_ids,
        )
        a_sequence_output = a_outputs[0][:, 0, :]
        pos_b_outputs = self.roberta(
            pos_b_input_ids,
            attention_mask=pos_b_attention_mask,
            token_type_ids=pos_b_token_type_ids,
        )
        pos_b_sequence_output = pos_b_outputs[0][:, 0, :]
        neg_b_outputs = self.roberta(
            neg_b_input_ids,
            attention_mask=neg_b_attention_mask,
            token_type_ids=neg_b_token_type_ids,
        )
        neg_b_sequence_output = neg_b_outputs[0][:, 0, :]
        # print(a_sequence_output.size(),pos_b_sequence_output.size())
        product_in_batch = torch.mm(a_sequence_output, pos_b_sequence_output.t())
        product_neg = (a_sequence_output * neg_b_sequence_output).sum(-1).unsqueeze(1)
        product = torch.cat([product_in_batch, product_neg], dim=-1)

        # return {'inbatch_qc_scores': inbatch_qc_scores, 'neg_qc_score': neg_qc_score}
        # logits = self.classifier(sequence_output)
        target = torch.arange(product.size(0)).to(product.device)
        loss_fct =  CrossEntropyLoss()
        loss = loss_fct(product, target)
        # if not return_dict:
        output = [a_sequence_output,pos_b_sequence_output,neg_b_sequence_output]#(logits,) + a_sequence_outputs[2:]
        return ((loss,output)) if loss is not None else output

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

class RobertaForSequenceClassificationECMASK(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.num_labels = 1
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(
        self,
        a_input_ids=None,
        a_attention_mask=None,
        a_token_type_ids=None,
        pos_b_input_ids=None,
        pos_b_attention_mask=None,
        pos_b_token_type_ids=None,
        neg_b_input_ids=None,
        neg_b_attention_mask=None,
        neg_b_token_type_ids=None,
        mask_idx=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        a_outputs = self.roberta(
            a_input_ids,
            attention_mask=a_attention_mask,
            token_type_ids=a_token_type_ids,
            position_ids=position_ids,
        )
        a_sequence_output = a_outputs[0][:, 0, :]
        # a_sequence_output = batched_index_select(a_sequence_output, 1, mask_idx).squeeze(1)
        # print(a_sequence_output.size())
        # print(mask_idx,a_sequence_output)
        pos_b_outputs = self.roberta(
            pos_b_input_ids,
            attention_mask=pos_b_attention_mask,
            token_type_ids=pos_b_token_type_ids,
        )
        pos_b_sequence_output = pos_b_outputs[0][:, 0, :]
        neg_b_outputs = self.roberta(
            neg_b_input_ids,
            attention_mask=neg_b_attention_mask,
            token_type_ids=neg_b_token_type_ids,
        )
        neg_b_sequence_output = neg_b_outputs[0][:, 0, :]
        # print(a_sequence_output.size(),pos_b_sequence_output.size())
        product_in_batch = torch.mm(a_sequence_output, pos_b_sequence_output.t())
        product_neg = (a_sequence_output * neg_b_sequence_output).sum(-1).unsqueeze(1)
        product = torch.cat([product_in_batch, product_neg], dim=-1)

        # return {'inbatch_qc_scores': inbatch_qc_scores, 'neg_qc_score': neg_qc_score}
        # logits = self.classifier(sequence_output)
        target = torch.arange(product.size(0)).to(product.device)
        loss_fct =  CrossEntropyLoss()
        loss = loss_fct(product, target)
        # if not return_dict:
        output = [a_sequence_output,pos_b_sequence_output,neg_b_sequence_output]#(logits,) + a_sequence_outputs[2:]
        return ((loss,output)) if loss is not None else output
