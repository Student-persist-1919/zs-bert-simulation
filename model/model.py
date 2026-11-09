"""ZS-BERT model definition (exact decompilation from dinobby/ZS-BERT + modern transformers patch).

Paper: "ZS-BERT: Towards Zero-Shot Relation Extraction with Attribute Representation Learning" (NAACL 2021)
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertTokenizer


class ZSBert(BertPreTrainedModel):
    """Zero-Shot BERT (ZS-BERT) model."""

    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super(ZSBert, self).__init__(config)
        self.num_labels = config.num_labels
        self.relation_emb_dim = config.relation_emb_dim
        self.margin = config.margin
        self.alpha = config.alpha
        self.dist_func = config.dist_func
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fclayer = nn.Linear(config.hidden_size * 3, self.relation_emb_dim)
        self.classifier = nn.Linear(self.relation_emb_dim, self. config.num_labels)
        self.last_rank_loss = 0.0
        self.last_ce_loss = 0.0
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        e1_mask=None,
        e2_mask=None,
        head_mask=None,
        inputs_embeds=None,
        input_relation_emb=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]  # Sequence of hidden-states of the last layer
        pooled_output = outputs[1]    # Last layer hidden-state of the [CLS] token

        def extract_entity(sequence_output, e_mask):
            extended_e_mask = e_mask.unsqueeze(1)
            extended_e_mask = torch.bmm(
                extended_e_mask.float(), sequence_output.float()
            ).squeeze(1)
            return extended_e_mask.float()

        e1_h = extract_entity(sequence_output, e1_mask)
        e2_h = extract_entity(sequence_output, e2_mask)
        context = self.dropout(pooled_output.float())
        pooled_output = torch.cat([context, e1_h, e2_h], dim=-1)
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.fclayer(pooled_output)
        relation_embeddings = torch.tanh(pooled_output)
        relation_embeddings = self.dropout(relation_embeddings)
        logits = self.classifier(relation_embeddings)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            gamma = torch.tensor(float(self.margin), dtype=torch.float32, device=device)
            ce_loss_fn = nn.CrossEntropyLoss()
            ce_val = ce_loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            loss = ce_val * self.alpha
            zeros = torch.tensor(0.0, dtype=torch.float32, device=device)
            rank_loss_accum = 0.0

            for a, b in enumerate(relation_embeddings):
                max_val = torch.tensor(0.0, dtype=torch.float32, device=device)
                pos = torch.tensor(0.0, dtype=torch.float32, device=device)
                for i, j in enumerate(input_relation_emb):
                    if a == i:
                        if self.dist_func == "inner":
                            pos = torch.dot(b, j).to(device)
                        elif self.dist_func in ("euclidian", "euclid"):
                            pos = torch.dist(b, j, 2).to(device)
                        elif self.dist_func == "cosine":
                            pos = torch.cosine_similarity(b, j, dim=0).to(device)
                    else:
                        if self.dist_func == "inner":
                            tmp = torch.dot(b, j).to(device)
                        elif self.dist_func in ("euclidian", "euclid"):
                            tmp = torch.dist(b, j, 2).to(device)
                        elif self.dist_func == "cosine":
                            tmp = torch.cosine_similarity(b, j, dim=0).to(device)
                        if tmp > max_val:
                            if labels[a] != labels[i]:
                                max_val = tmp
                neg = max_val.to(device)
                if self.dist_func == "inner" or self.dist_func == "cosine":
                    step_rank = torch.max(zeros, neg - pos + gamma)
                    loss += step_rank * (1 - self.alpha)
                    rank_loss_accum += float(step_rank.detach().cpu().item())
                elif self.dist_func in ("euclidian", "euclid"):
                    step_rank = torch.max(zeros, pos - neg + gamma)
                    loss += step_rank * (1 - self.alpha)
                    rank_loss_accum += float(step_rank.detach().cpu().item())

            self.last_ce_loss = float(ce_val.detach().cpu().item())
            self.last_rank_loss = rank_loss_accum
            outputs = (loss,) + outputs

        return outputs, relation_embeddings
