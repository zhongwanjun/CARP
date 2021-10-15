# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.

from transformers import AutoModel,  BertModel, LongformerForQuestionAnswering
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F
from qa.utils_qa import LongformerQuestionAnsweringModelOutput
import logging
logger = logging.getLogger(__name__)



def _get_question_end_index(input_ids, sep_token_id):
    """
    Computes the index of the first occurrence of `sep_token_id`.
    """

    sep_token_indices = (input_ids == sep_token_id).nonzero()
    batch_size = input_ids.shape[0]

    assert sep_token_indices.shape[1] == 2, "`input_ids` should have two dimensions"
    assert (
        sep_token_indices.shape[0] == 3 * batch_size
    ), f"There should be exactly three separator tokens: {sep_token_id} in every sample for questions answering. You might also consider to set `global_attention_mask` manually in the forward function to avoid this error."
    return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]

def _compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True):
    """
    Computes global attention mask by putting attention on all tokens before `sep_token_id` if `before_sep_token is
    True` else after `sep_token_id`.
    """
    question_end_index = _get_question_end_index(input_ids, sep_token_id)
    question_end_index = question_end_index.unsqueeze(dim=1)  # size: batch_size x 1
    # bool attention mask with True in locations of global attention
    attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
    if before_sep_token is True:
        attention_mask = (attention_mask.expand_as(input_ids) < question_end_index).to(torch.uint8)
    else:
        # last token is separation token and should not be counted and in the middle are two separation tokens
        attention_mask = (attention_mask.expand_as(input_ids) > (question_end_index + 1)).to(torch.uint8) * (
            attention_mask.expand_as(input_ids) < input_ids.shape[-1]
        ).to(torch.uint8)
    return attention_mask

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)

class LongformerQAModelEC(LongformerForQuestionAnswering):
    def __init__(self,config):
        super(LongformerQAModelEC, self).__init__(config)
        #self.tb_scorer = nn.Linear(config.hidden_size, 2)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ec_mask=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            if input_ids is None:
                logger.warning(
                    "It is not possible to automatically generate the `global_attention_mask` because input_ids is None. Please make sure that it is correctly set."
                )
            else:
                # set global attention on question tokens automatically
                global_attention_mask = _compute_global_attention_mask(input_ids, self.config.sep_token_id)

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).clone() - 1e-9 * ec_mask
        end_logits = end_logits.squeeze(-1).clone() - 1e-9 * ec_mask
        start_logits = start_logits.contiguous()
        end_logits = end_logits.contiguous()#.squeeze(-1).contiguous()
        #assert(start_logits.size()==ec_mask.size())


        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        return_dict = False
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LongformerQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

class LongformerQAModel(LongformerForQuestionAnswering):
    def __init__(self,config):
        super(LongformerQAModel, self).__init__(config)
        self.tb_scorer = nn.Linear(config.hidden_size, 2)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        tb_pos=None,
        tb_mask=None,
        tb_num=None,
        tb_labels=None
    ):
        if global_attention_mask is None:
            if input_ids is None:
                logger.warning(
                    "It is not possible to automatically generate the `global_attention_mask` because input_ids is None. Please make sure that it is correctly set."
                )
            else:
                # set global attention on question tokens automatically
                global_attention_mask = _compute_global_attention_mask(input_ids, self.config.sep_token_id)

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]#batch size, seq_length, emb_dim

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        #select module: update logits with the score of table block selection
        tb_reps = batched_index_select(sequence_output, 1, tb_pos)
        tb_logits = self.tb_scorer(tb_reps)
        tb_scores = torch.nn.functional.softmax(tb_logits[:,:,1].squeeze(-1), dim=-1)
        tb_num_mask = torch.arange(tb_scores.size(1)).cuda().expand(tb_scores.size(0), tb_scores.size(1)) <= tb_num.unsqueeze(1)
        tb_scores = tb_scores.masked_fill(tb_num_mask.ne(1),float(0)) #[o.float().masked_fill(tb_num_mask.ne(1), float("-inf")).type_as(o) for o in tb_scores]
        select_start_logits = torch.bmm(tb_scores.unsqueeze(-1),start_logits.unsqueeze(1))
        select_end_logits = torch.bmm(tb_scores.unsqueeze(-1), end_logits.unsqueeze(1))
        select_start_logits = torch.sum(select_start_logits * tb_mask,dim=1)
        select_end_logits = torch.sum(select_end_logits * tb_mask,dim=1)
        start_logits = select_start_logits
        end_logits = select_end_logits

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            rank_loss = loss_fct(tb_logits.view(-1,tb_logits.size(-1)), tb_labels.view(-1))  #torch.sum(loss_fct(tb_scores, tb_labels))
            total_loss = (start_loss + end_loss + rank_loss) / 3

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LongformerQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

'''
class LongformerForQuestionAnswering(LongformerPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        Returns:
        Examples::
            >>> from transformers import LongformerTokenizer, LongformerForQuestionAnswering
            >>> import torch
            >>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
            >>> model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
            >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
            >>> encoding = tokenizer(question, text, return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> # default is local attention everywhere
            >>> # the forward method will automatically set global attention on question tokens
            >>> attention_mask = encoding["attention_mask"]
            >>> outputs = model(input_ids, attention_mask=attention_mask)
            >>> start_logits = outputs.start_logits
            >>> end_logits = outputs.end_logits
            >>> all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            >>> answer_tokens = all_tokens[torch.argmax(start_logits) :torch.argmax(end_logits)+1]
            >>> answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens)) # remove space prepending space token
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            if input_ids is None:
                logger.warning(
                    "It is not possible to automatically generate the `global_attention_mask` because input_ids is None. Please make sure that it is correctly set."
                )
            else:
                # set global attention on question tokens automatically
                global_attention_mask = _compute_global_attention_mask(input_ids, self.config.sep_token_id)

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return LongformerQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )
'''


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class QAModel(nn.Module):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__()
        self.model_name = args.model_name
        self.sp_weight = args.sp_weight
        self.sp_pred = args.sp_pred
        self.encoder = AutoModel.from_pretrained(args.model_name)

        if "electra" in args.model_name:
            self.pooler = BertPooler(config)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.rank = nn.Linear(config.hidden_size, 1) # noan

        if self.sp_pred:
            self.sp = nn.Linear(config.hidden_size, 1)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction="none")

    def forward(self, batch):

        outputs = self.encoder(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids', None))

        if "electra" in self.model_name:
            sequence_output = outputs[0]
            pooled_output = self.pooler(sequence_output)
        else:
            sequence_output, pooled_output = outputs[0], outputs[1]

        logits = self.qa_outputs(sequence_output)
        outs = [o.squeeze(-1) for o in logits.split(1, dim=-1)]
        outs = [o.float().masked_fill(batch["paragraph_mask"].ne(1), float("-inf")).type_as(o) for o in outs]

        start_logits, end_logits = outs[0], outs[1]
        rank_score = self.rank(pooled_output)

        if self.sp_pred:
            gather_index = batch["sent_offsets"].unsqueeze(2).expand(-1, -1, sequence_output.size()[-1])
            sent_marker_rep = torch.gather(sequence_output, 1, gather_index)
            sp_score = self.sp(sent_marker_rep).squeeze(2)
        else:
            sp_score = None

        if self.training:

            rank_target = batch["label"]
            if self.sp_pred:
                sp_loss = F.binary_cross_entropy_with_logits(sp_score, batch["sent_labels"].float(), reduction="none")
                sp_loss = (sp_loss * batch["sent_offsets"]) * batch["label"]
                sp_loss = sp_loss.sum()

            start_positions, end_positions = batch["starts"], batch["ends"]

            rank_loss = F.binary_cross_entropy_with_logits(rank_score, rank_target.float(), reduction="sum")

            start_losses = [self.loss_fct(start_logits, starts) for starts in torch.unbind(start_positions, dim=1)]
            end_losses = [self.loss_fct(end_logits, ends) for ends in torch.unbind(end_positions, dim=1)]
            loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)
            log_prob = - loss_tensor
            log_prob = log_prob.float().masked_fill(log_prob == 0, float('-inf')).type_as(log_prob)
            marginal_probs = torch.sum(torch.exp(log_prob), dim=1)
            m_prob = [marginal_probs[idx] for idx in marginal_probs.nonzero()]
            if len(m_prob) == 0:
                span_loss = self.loss_fct(start_logits, start_logits.new_zeros(
                    start_logits.size(0)).long()-1).sum()
            else:
                span_loss = - torch.log(torch.cat(m_prob)).sum()

            if self.sp_pred:
                loss = rank_loss + span_loss + sp_loss * self.sp_weight
            else:
                loss = rank_loss + span_loss
            return loss.unsqueeze(0)

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'rank_score': rank_score,
            "sp_score": sp_score
            }
