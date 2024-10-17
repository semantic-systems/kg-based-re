from collections import defaultdict

import math
import numpy as np
import torch.nn
import pytorch_lightning as pl
from opt_einsum import contract
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import torch.nn.functional as F

from src.supervised.models.nbf_alt import NBFAltNet
from src.supervised.models.relation_candidate_retriever import RelationCandidateRetriever


class ATLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, share_threshold=False):
        labels = torch.clone(labels)
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels


        if share_threshold:
            thresholds = logits[:, 0]
            maximum_threshold = torch.sum(torch.softmax(thresholds, dim=0) * thresholds,dim=0).unsqueeze(0).unsqueeze(0).repeat(logits.size(0), 1)
            minimum_threshold = torch.sum(torch.softmax(-thresholds, dim=0) * thresholds,dim=0).unsqueeze(0).unsqueeze(0).repeat(logits.size(0), 1)
            positive_logits = torch.cat([maximum_threshold, logits[:, 1:]], dim=-1)
            negative_logits = torch.cat([minimum_threshold, logits[:, 1:]], dim=-1)
        else:
            positive_logits = logits
            negative_logits = logits
        # Rank positive classes to TH
        logit1 = positive_logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
        # Rank TH to negative classes
        logit2 = negative_logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):

        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]  # smallest logits among the num_labels
            # predictions are those logits > thresh and logits >= smallest
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        # if no such relation label exist: set its label to 'Nolabel'
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output.bool()

    def get_score(self, logits, num_labels=-1):

        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:, 1] - logits[:, 0], 0


class HingeABL(torch.nn.Module):
    def forward(self, logits, labels, m=5):
        """
           HingeABL
           """


        p_num = labels[:, 1:].sum(dim=1)
        p_logits_diff = logits[:, 0].unsqueeze(dim=1) - logits
        p_logits_imp = F.relu(p_logits_diff + m)
        p_logits_imp = p_logits_imp * labels
        p_logits_imp = p_logits_imp[:, 1:]
        p_logits_imp = p_logits_imp / (p_logits_imp.sum(dim=1).unsqueeze(dim=1) + 1e-30)

        n_logits_diff = logits - logits[:, 0].unsqueeze(dim=1)
        n_logits_imp = F.relu(n_logits_diff + m)
        n_logits_imp = n_logits_imp * (1 - labels)
        n_logits_imp = n_logits_imp[:, 1:]
        n_logits_imp = n_logits_imp / (n_logits_imp.sum(dim=1).unsqueeze(dim=1) + 1e-30)

        # Calculate exp_th in a numerically stable manner
        logits_margin = logits[:, 0].unsqueeze(dim=1)  # Extract the margin logit
        # Subtract the maximum logit for numerical stability
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        # Numerically stable computation
        numerator = torch.exp(logits - max_logits)
        exp_th = torch.exp(logits_margin - max_logits)
        # Compute p_prob and n_prob using the log-sum-exp trick
        p_prob = numerator / (numerator + exp_th)
        n_prob = exp_th / (numerator + exp_th)


        # exp_th = torch.exp(logits[:, 0].unsqueeze(dim=1))  # margin=5
        # p_prob = torch.exp(logits) / (torch.exp(logits) + exp_th)
        # n_prob = exp_th / (exp_th + torch.exp(logits))

        p_item = -torch.log(p_prob + 1e-30) * labels
        p_item = p_item[:, 1:] * p_logits_imp
        n_item = -torch.log(n_prob + 1e-30) * (1 - labels)
        n_item = n_item[:, 1:] * n_logits_imp

        p_loss = p_item.sum(1)
        n_loss = n_item.sum(1)
        loss = p_loss + n_loss
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):

        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]  # smallest logits among the num_labels
            # predictions are those logits > thresh and logits >= smallest
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        # if no such relation label exist: set its label to 'Nolabel'
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output.bool()

    def get_score(self, logits, num_labels=-1):

        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:, 1] - logits[:, 0], 0

class TATLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        labels = torch.clone(labels)
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        num_positives = torch.sum(labels)
        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum()/num_positives
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).mean()


        return  (loss1 + loss2) / 2


    def get_label(self, logits, num_labels=-1):

        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]  # smallest logits among the num_labels
            # predictions are those logits > thresh and logits >= smallest
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        # if no such relation label exist: set its label to 'Nolabel'
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output.bool()

    def get_score(self, logits, num_labels=-1):

        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:, 1] - logits[:, 0], 0


class SepBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels):

        pos = labels == 1
        neg = labels == 0
        labels[labels == -1] = 0
        loss = self.bce(logits, labels)

        positive_loss = loss[pos].mean()
        negative_loss = loss[neg].mean()
        sorted_negatives = torch.sort(loss[neg], descending=True).values
        top_k_negative_loss = sorted_negatives[:pos.sum()].mean()
        other_negative_loss = sorted_negatives[pos.sum():].mean()

        final_loss = (positive_loss + negative_loss) / 2

        return final_loss

    def get_label(self, logits, num_labels=-1):
        predictions = torch.sigmoid(logits)
        mask = (predictions > 0.5)
        if num_labels > 0:
            top_v, _ = torch.topk(predictions, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (predictions >= top_v.unsqueeze(1)) & mask

        return mask

class CELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.ce(logits, labels)

    def get_label(self, logits, num_labels=-1):
        labels = torch.zeros((logits.size(0), logits.size(1) + 1), dtype=torch.long).to(logits.device)
        max_indices = torch.argmax(logits, dim=1)
        labels[torch.arange(logits.size(0)), max_indices + 1] = 1
        return labels

    def get_score(self, logits, num_labels=-1):
        return torch.topk(logits, num_labels, dim=1)


from torch.cuda.amp import custom_bwd, custom_fwd


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def dclamp(input, min, max):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, min, max)


class REModel(pl.LightningModule):
    def __init__(self, num_output_relation: int, num_classes: int, num_relations=None, model_name: str = "roberta-large", lr_encoder=3e-5, at_loss: bool = False,
                 warmup_steps = 1000, total_steps = 10000, num_rules = 1, num_hops = 6,
                 re_weight=0.5, use_hinge_abl=False, lr_classifier=1e-4, cross_relation_weight=1.0, evaluator=None,
                 alt_mode=False, deactivate_graph=False, use_only_types=False, random_dropout=0.2,
                 short_cut=False, gnn_checkpoint=None, labels_to_predict=4, graph_dim=128, separated=False,
                 full_document=False, dynamic_freeze=False, graph_only=False, post_prediction=False
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.has_no_label = False
        if at_loss:
            self.loss_fn = ATLoss()
            self.has_no_label = True
        elif use_hinge_abl:
            self.loss_fn = HingeABL()
            self.has_no_label = True
        else:
            self.loss_fn = CELoss()
        self.gating_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.alt_mode = alt_mode
        self.alt_loss_fn = ATLoss() # torch.nn.CrossEntropyLoss()

        self.subject = torch.nn.Sequential(
            torch.nn.Linear( 2 * self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.Tanh()
        )
        self.object = torch.nn.Sequential(
            torch.nn.Linear( 2 * self.model.config.hidden_size, self.model.config.hidden_size),
            torch.nn.Tanh()
        )
        self.labels_to_predict = labels_to_predict
        self.random_dropout = random_dropout
        self.deactivate_graph = deactivate_graph
        self.block_size = 64
        self.emb_size = self.model.config.hidden_size
        # self.bilinear = torch.nn.Bilinear(self.model.config.hidden_size, self.model.config.hidden_size, num_relations)
        self.lr_encoder = lr_encoder
        self.lr_classifier = lr_classifier
        self.outputs = []
        self.evaluator=evaluator
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.at_loss = at_loss
        self.re_weight = re_weight
        self.cross_relation_weight = cross_relation_weight
        self.full_document = full_document
        self.graph_only = graph_only
        assert (graph_only and separated) or not graph_only

        self.separated = separated
        if num_relations is None:
            num_relations = num_output_relation

        self.post_prediction = post_prediction

        self.post_prediction_offset = 2 * num_relations
        if self.post_prediction:
            num_relations += num_output_relation - self.has_no_label

        self.graph_predictor = NBFAltNet(
            **{
                "input_dim": graph_dim,
                "hidden_dims": [graph_dim] * num_hops,
                "short_cut": True,
                "layer_norm": True,
                "num_output_relation": num_output_relation,
                "num_relation": num_relations,
            }, num_rep=num_rules)

        if gnn_checkpoint is not None:
            state_dict = torch.load(gnn_checkpoint)['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            self.graph_predictor.load_state_dict(state_dict)

            for parameter in self.graph_predictor.parameters():
                parameter.requires_grad = False


        self.relation_candidate_retriever = RelationCandidateRetriever(num_classes=num_classes, num_relations=num_relations)
        if self.deactivate_graph:
            self.bilinear = torch.nn.Linear(self.model.config.hidden_size * 64, num_output_relation)
            self.predictor = torch.nn.Linear(graph_dim, num_output_relation)
        else:
            if self.separated:
                self.bilinear = torch.nn.Linear(self.model.config.hidden_size * 64, num_output_relation)
            else:
                self.bilinear = torch.nn.Linear(self.model.config.hidden_size * 64, graph_dim)
            if self.graph_only:
                self.predictor = torch.nn.Linear(graph_dim, num_output_relation)
            else:
                self.predictor = torch.nn.Linear(graph_dim, num_output_relation - 1)
        self.no_path_embedding = torch.nn.Parameter(torch.randn(graph_dim))
        self.deactivate_scheduler = False
        self.num_relations = num_relations
        self.deactivate_graph = deactivate_graph
        self.use_only_types = use_only_types
        self.short_cut = short_cut
        self.dynamic_freeze = dynamic_freeze

    def get_hrt(self, sequence_output, attention, entity_pos, hts):

        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)):  # for each batch
            entity_embs, entity_atts = [], []

            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]):  # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, start in enumerate(e):  # for every mention
                        if start < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start])
                            e_att.append(attention[i, :, start])

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.model.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                elif len(e) == 1:
                    start = e[0]
                    if start < c:
                        e_emb = sequence_output[i, start]
                        e_att = attention[i, :, start]
                    else:
                        e_emb = torch.zeros(self.model.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    e_emb = torch.zeros(self.model.config.hidden_size).to(sequence_output)
                    e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1)  # average over all heads
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)

            # obtain local context embeddings.
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0)  # (num_ent_pairs_all_batches, max_doc_len)


        return hss, rss, tss, ht_atts, rels_per_batch

    def forward_rel(self, hs, ts, rs, graph_info, no_relation=None, graph_batch_size: int =512):
        '''
        Forward computation for RE.
        Inputs:
            :hs: (num_ent_pairs_all_batches, emb_size)
            :ts: (num_ent_pairs_all_batches, emb_size)
            :rs: (num_ent_pairs_all_batches, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
        '''

        batch, all_hts_indices = self.construct_graph_input(graph_info)
        local_graph, alternative_hts = graph_info

        hs = self.subject(torch.cat([hs, rs], dim=-1))
        ts = self.object(torch.cat([ts, rs], dim=-1))


        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        queries = self.bilinear(bl)
        representation_to_include = queries
        additional_edge_index = torch.tensor(alternative_hts, device=self.device, dtype=torch.int64).T

        if self.deactivate_graph:
            logits = queries
        else:
            if self.separated:
                edge_index = local_graph.edge_index
                edge_type = local_graph.edge_type
                valid_predictions = queries[:, 1:] > queries[:, 0].unsqueeze(1)
                if self.post_prediction and torch.sum(valid_predictions) < 5000:
                    arg_indices = torch.nonzero(valid_predictions, as_tuple=False)
                    new_edge_index = []
                    new_edge_type = []
                    for i, j in arg_indices:
                        edge = additional_edge_index[:, i]
                        new_edge_index.append(edge)
                        new_edge_type.append(j + self.post_prediction_offset)
                        new_edge_index.append(torch.flip(edge, [0]))
                        new_edge_type.append(2 * j + self.post_prediction_offset)
                    if new_edge_index:
                        new_edge_index = torch.stack(new_edge_index, dim=1)
                        new_edge_type = torch.tensor(new_edge_type, device=self.device, dtype=torch.int64)
                        edge_index = torch.cat([local_graph.edge_index, new_edge_index], dim=-1)
                        edge_type = torch.cat([local_graph.edge_type, new_edge_type], dim=-1)
                local_graph_without = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=local_graph.num_nodes)
                relation_scores_with, path_boundaries = self.graph_predictor(local_graph_without, batch)
                valid_mask = (all_hts_indices != -1)
                selected_features = relation_scores_with[valid_mask]
                update_indices = all_hts_indices[valid_mask]
                graph_tensors = torch.zeros((queries.size(0), relation_scores_with.size(2)), device=self.device)
                graph_tensors = graph_tensors.scatter(0, update_indices.unsqueeze(1).expand_as(selected_features),
                                                      selected_features)
                path_boundaries = path_boundaries[valid_mask]
                restruct_path_boundaries = torch.zeros((queries.size(0), 1), device=self.device)
                restruct_path_boundaries = restruct_path_boundaries.scatter(0, update_indices.unsqueeze(1), path_boundaries)
                graph_tensors[restruct_path_boundaries.squeeze(-1)==0, :] = self.no_path_embedding

                if self.graph_only:
                    logits = self.predictor(graph_tensors)
                else:
                    logits = queries
                    graph_predictions = self.graph_predictor.scorer(graph_tensors)
                    #graph_predictions[restruct_path_boundaries.squeeze(-1)==0, :] = 0
                    logits += graph_predictions
                # logits = self.predictor(graph_tensors)


            else:
                edge_index_with = torch.cat([local_graph.edge_index, additional_edge_index], dim=-1)
                edge_type_with = torch.cat([local_graph.edge_type,
                                       2 * self.num_relations + torch.arange(0, additional_edge_index.size(1),
                                                                             device=self.device)], dim=-1)
                # edge_index_without = local_graph.edge_index
                # edge_type_without = local_graph.edge_type

                local_graph_with = Data(edge_index=edge_index_with, edge_attr=representation_to_include, edge_type=edge_type_with, num_nodes=local_graph.num_nodes)
                # local_graph_without = Data(edge_index=edge_index_without, edge_type=edge_type_without, num_nodes=local_graph.num_nodes)

                all_relation_features = []
                for i in range(0, len(batch), graph_batch_size):
                    batch_ = batch[i:i + graph_batch_size]
                    relation_scores_with, _ = self.graph_predictor(local_graph_with, batch_)
                    all_relation_features.append(relation_scores_with)
                all_relation_features = torch.cat(all_relation_features, dim=0)

                valid_mask = (all_hts_indices != -1)
                selected_features = all_relation_features[valid_mask]
                update_indices = all_hts_indices[valid_mask]
                graph_tensors = torch.zeros_like(queries, device=self.device)
                graph_tensors = graph_tensors.scatter(0, update_indices.unsqueeze(1).expand_as(selected_features),
                                                      selected_features)

                if self.short_cut:
                    graph_tensors = graph_tensors + queries
                logits = self.graph_predictor.scorer(graph_tensors)
        return logits


    def encode(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        hidden_state = torch.mean(torch.stack(output.hidden_states[-3:], dim=-1), dim=-1)
        attention = torch.mean(torch.stack(output.attentions[-3:], dim=1), dim=1)
        return hidden_state, attention

    def create_graph_input(self, edge_index, edge_attr, hts, logits, relevant_edges):
        data_objects = []
        offset = 0
        for elements, edge_index_, edge_attr_ in zip(hts,edge_index, edge_attr):
            x = torch.zeros_like(logits[offset:offset + len(elements)])
            x[relevant_edges[offset:offset + len(elements)]] = logits[offset:offset + len(elements)][relevant_edges[offset:offset + len(elements)]]
            valid = torch.all(relevant_edges[offset:offset + len(elements)][edge_index_.T], dim=-1)
            sub_edge_index_ = edge_index_[:, valid]
            sub_edge_attr_ = edge_attr_[valid]

            data = Data(x, sub_edge_index_, sub_edge_attr_)
            data_objects.append(data)

            offset += len(elements)

        batch = Batch.from_data_list(data_objects)
        return batch

    def construct_graph_input(self, graph_info):
        local_graph, alternative_hts = graph_info
        batch = []
        subjects = defaultdict(list)
        for idx, (h, t) in enumerate(alternative_hts):
            subjects[h].append((t, idx))
        longest = max([len(ts) for ts in subjects.values()])
        all_hts_indices = []
        for h, ts in subjects.items():
            # Extract tensors from ts
            ts_tensors = torch.tensor([t for t, _ in ts], device=self.device)
            indices = torch.tensor([idx for _, idx in ts], device=self.device, dtype=torch.int64)

            # Filter tensors based on condition

            # Get indices of filtered scores

            h = torch.tensor([h], device=self.device)
            h = h.unsqueeze(1).expand(-1, len(ts_tensors))
            ht = torch.stack([h, ts_tensors.unsqueeze(0).expand(h.size(0), -1)], dim=-1)
            if ht.size(1) < longest:
                ht = torch.cat([ht, ht[:, 0].unsqueeze(1).expand(-1, longest - ht.size(1), -1)], dim=1)
                indices = torch.cat([indices,
                                     -torch.ones(longest - indices.size(0),
                                                 device=self.device, dtype=torch.int64)], dim=0)
            all_hts_indices.append(indices.unsqueeze(0).expand(ht.size(0), -1))
            batch.append(ht)

        all_hts_indices = torch.cat(all_hts_indices, dim=0)
        batch = torch.cat(batch, dim=0)

        return batch, all_hts_indices

    def get_graph_features(self, graph_info, entity_types, hts, output_size: int, graph_batch_size: int =512):
        type_based_relation_scores = []
        for hts_, entity_types_ in zip(hts, entity_types):
            s, o = zip(*hts_)
            s = torch.tensor(s).to(self.device)
            o = torch.tensor(o).to(self.device)
            subjects = entity_types_[s, :]
            objects = entity_types_[o, :]
            no_type_subjects = torch.all((subjects == -1), dim=-1)
            no_type_objects = torch.all((objects == -1), dim=-1)
            no_type = no_type_subjects | no_type_objects
            scores = self.relation_candidate_retriever(subjects, objects)
            scores[no_type, :] = 0
            type_based_relation_scores.append(scores)
        type_based_relation_scores = torch.cat(type_based_relation_scores, dim=0)

        batch, all_hts_indices = self.construct_graph_input(graph_info)

        local_graph, alternative_hts = graph_info

        if not self.deactivate_graph:
            all_relation_scores = []
            for i in range(0, len(batch), graph_batch_size):
                batch_ = batch[i:i + graph_batch_size]
                relation_scores = self.graph_predictor(local_graph, batch_)
                all_relation_scores.append(relation_scores)
            all_relation_scores = torch.cat(all_relation_scores, dim=0)

            valid_mask = (all_hts_indices != -1)
            selected_scores = all_relation_scores[valid_mask]
            update_indices = all_hts_indices[valid_mask]
            graph_tensors = torch.zeros((output_size, all_relation_scores.size(2)), device=self.device)
            graph_tensors = graph_tensors.scatter(0, update_indices.unsqueeze(1).expand_as(selected_scores), selected_scores)
        else:
            graph_tensors = torch.zeros((output_size, self.num_relations - 1), device=self.device)
        return graph_tensors, type_based_relation_scores

    def forward(self, input_ids, attention_mask, entity_pos, hts, graph_info, no_relation=None):

        num_pairs = sum([len(hts_) for hts_ in hts])


        if self.full_document:
            hidden_state, attention = self.process_longer_input(input_ids, attention_mask)
        else:
            hidden_state, attention = self.process_long_input(input_ids, attention_mask)

        hs, rs, ts, doc_attn, batch_rel = self.get_hrt(hidden_state, attention, entity_pos, hts)

        logits = self.forward_rel(hs, ts, rs, graph_info, no_relation)
        gating_score = torch.zeros((num_pairs, 1), device=self.device)

        return logits, gating_score

    def process_longer_input(self, input_ids, attention_mask, stride=256):
        # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
        n, c = input_ids.size()
        bos_token_id = self.model.config.bos_token_id if self.model.config.bos_token_id is not None else 101
        eos_token_id = self.model.config.eos_token_id if self.model.config.eos_token_id is not None else 102
        start_tokens = torch.tensor([bos_token_id  ]).to(input_ids)
        # TODO: check if this is correct
        end_tokens = torch.tensor([eos_token_id, eos_token_id]).to(input_ids)
        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)
        if c <= 512:
            # if document can fit into the encoder
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )
            sequence_outputs = torch.stack(output[-2][-3:], dim=1)
            sequence_output = sequence_outputs.mean(dim=1)
            attentions = torch.stack(output[-1][-3:], dim=1)
            attention = attentions.mean(dim=1)

        else:
            new_input_ids, new_attention_mask, num_seg = [], [], []
            seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
            start_tokens = torch.tensor([bos_token_id]).to(input_ids)
            # TODO: check if this is correct
            end_tokens = torch.tensor([eos_token_id]).to(input_ids)
            all_segment_spans = []
            for i, l_i in enumerate(seq_len):  # for each batch
                if l_i <= 512:
                    new_input_ids.append(input_ids[i, :512])
                    new_attention_mask.append(attention_mask[i, :512])
                    num_seg.append(1)
                else:  # split the input into two parts: (0, 512) and (end - 512, end)
                    num_seq = math.ceil(l_i / 512)
                    seq_counter = 0
                    input_ids_elem = input_ids[i, 1:-1]
                    attention_mask_elem = attention_mask[i, 1:-1]
                    segment_spans = []
                    start_span = 0
                    end_span = 510
                    while start_span < l_i - 2:
                        segment_spans.append((start_span, min(end_span, l_i-2)))
                        start_span += stride
                        end_span += stride

                    for start_span, end_span in segment_spans:
                        new_input_ids.append(torch.cat([start_tokens, input_ids_elem[start_span: end_span], end_tokens], dim=-1))
                        new_attention_mask.append(torch.cat([torch.ones_like(start_tokens, device=self.device), attention_mask_elem[start_span: end_span], torch.ones_like(end_tokens, device=self.device)], dim=-1))
                        seq_counter += 1
                    num_seg.append(seq_counter)
                    all_segment_spans.append(segment_spans)
                    # input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                    # input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                    # attention_mask1 = attention_mask[i, :512]
                    # attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                    # new_input_ids.extend([input_ids1, input_ids2])
                    # new_attention_mask.extend([attention_mask1, attention_mask2])
                    # num_seg.append(2)

            input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )

            sequence_outputs = torch.stack(output[-2][-3:], dim=1)
            sequence_output = sequence_outputs.mean(dim=1)
            attentions = torch.stack(output[-1][-3:], dim=1)
            attention = attentions.mean(dim=1)

            i = 0
            new_output, new_attention = [], []
            for (n_s, l_i, segment_spans) in zip(num_seg, seq_len, all_segment_spans):
                if n_s == 1:  # 1 segment (no split)
                    output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                    att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                    new_output.append(output)
                    new_attention.append(att)
                else:  # 2 segments (splitted)
                    outputs = []
                    atts = []
                    masks = []
                    left_pad = 0
                    right_pad = c
                    for s in range(n_s):
                        new_output_ = torch.zeros((l_i, sequence_output.size(-1)), device=sequence_output.device)
                        new_att = torch.zeros((attention.size(1), l_i, l_i), device=attention.device)
                        new_mask = torch.zeros((l_i), device=attention.device)
                        seg_start, seg_end = segment_spans[s]
                        segment_length = seg_end - seg_start
                        if s == 0:
                            seq_start = 0
                            seq_end = segment_length + 1
                            seg_end += 1
                        elif s  == n_s - 1:
                            seq_start = len_start
                            seq_end = segment_length + len_start + 1
                            seg_start += 1
                            seg_end += 2
                        else:
                            seq_start = len_start
                            seq_end = segment_length + len_start
                            seg_start += 1
                            seg_end += 1

                        output_ = sequence_output[i + s][seq_start:seq_end]
                        att_ = attention[i + s][:, seq_start:seq_end, seq_start:seq_end]
                        mask_ = attention_mask[i + s][seq_start:seq_end]

                        new_output_[seg_start:seg_end] = output_
                        new_att[:, seg_start:seg_end, seg_start:seg_end] = att_
                        new_mask[seg_start:seg_end] = mask_

                        outputs.append(new_output_)
                        atts.append(new_att)
                        masks.append(new_mask)

                    mask = torch.sum(torch.stack(masks, dim=0), dim=0) + 1e-10
                    output = torch.sum(torch.stack(outputs, dim=0), dim=0) / mask.unsqueeze(-1)
                    atts = torch.stack(atts, dim=0)
                    pair_wise_mask_max = torch.max(mask.unsqueeze(1), mask.unsqueeze(0))
                    att = torch.sum(atts, dim=0) / pair_wise_mask_max.unsqueeze(0)
                    att = att / (att.sum(-1, keepdim=True) + 1e-10)

                    new_output.append(output)
                    new_attention.append(att)

                i += n_s

            sequence_output = torch.stack(new_output, dim=0)
            attention = torch.stack(new_attention, dim=0)

        return sequence_output, attention


    def process_long_input(self, input_ids, attention_mask):
        # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
        n, c = input_ids.size()
        bos_token_id = self.model.config.bos_token_id if self.model.config.bos_token_id is not None else 101
        eos_token_id = self.model.config.eos_token_id if self.model.config.eos_token_id is not None else 102
        start_tokens = torch.tensor([bos_token_id  ]).to(input_ids)
        # TODO: check if this is correct
        end_tokens = torch.tensor([eos_token_id, eos_token_id]).to(input_ids)
        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)
        if c <= 512:
            # if document can fit into the encoder
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )
            sequence_outputs = torch.stack(output[-2][-3:], dim=1)
            sequence_output = sequence_outputs.mean(dim=1)
            attentions = torch.stack(output[-1][-3:], dim=1)
            attention = attentions.mean(dim=1)

        else:
            new_input_ids, new_attention_mask, num_seg = [], [], []
            seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
            for i, l_i in enumerate(seq_len):  # for each batch
                if l_i <= 512:
                    new_input_ids.append(input_ids[i, :512])
                    new_attention_mask.append(attention_mask[i, :512])
                    num_seg.append(1)
                else:  # split the input into two parts: (0, 512) and (end - 512, end)
                    input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                    input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                    attention_mask1 = attention_mask[i, :512]
                    attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                    new_input_ids.extend([input_ids1, input_ids2])
                    new_attention_mask.extend([attention_mask1, attention_mask2])
                    num_seg.append(2)

            input_ids = torch.stack(new_input_ids, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)

            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )

            sequence_outputs = torch.stack(output[-2][-3:], dim=1)
            sequence_output = sequence_outputs.mean(dim=1)
            attentions = torch.stack(output[-1][-3:], dim=1)
            attention = attentions.mean(dim=1)

            i = 0
            new_output, new_attention = [], []
            for (n_s, l_i) in zip(num_seg, seq_len):
                if n_s == 1:  # 1 segment (no split)
                    output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                    att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                    new_output.append(output)
                    new_attention.append(att)
                elif n_s == 2:  # 2 segments (splitted)

                    # first half
                    output1 = sequence_output[i][:512 - len_end]
                    mask1 = attention_mask[i][:512 - len_end]
                    att1 = attention[i][:, :512 - len_end, :512 - len_end]
                    # pad to reserve space for the second half
                    output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                    mask1 = F.pad(mask1, (0, c - 512 + len_end))
                    att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                    # second half
                    output2 = sequence_output[i + 1][len_start:]
                    mask2 = attention_mask[i + 1][len_start:]
                    att2 = attention[i + 1][:, len_start:, len_start:]
                    # pad to reserve space for the first half
                    output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                    mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                    att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])

                    # combine first half and second half
                    mask = mask1 + mask2 + 1e-10
                    output = (output1 + output2) / mask.unsqueeze(-1)
                    att = (att1 + att2)
                    att = att / (att.sum(-1, keepdim=True) + 1e-10)
                    new_output.append(output)
                    new_attention.append(att)
                i += n_s

            sequence_output = torch.stack(new_output, dim=0)
            attention = torch.stack(new_attention, dim=0)

        return sequence_output, attention


    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, entity_pos, hts, y, elems, relation_graph = batch

        gating_label = torch.max(y, dim=-1)[0]

        y_hat, gating_score = self(input_ids, attention_mask, entity_pos, hts, relation_graph, no_relation=gating_label.bool())

        gating_loss = self.gating_loss_fn(gating_score.squeeze(-1), gating_label.float())
        # y_hat = self.reproject(torch.stack([y_hat, y_hat_graph], dim=-1)).squeeze(-1)

        loss = self.loss_fn(y_hat, y) + self.cross_relation_weight * gating_loss
        self.log_dict({"train_loss": loss, "train_gating_loss": gating_loss
                    }, sync_dist=True, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_start(self) -> None:
        if self.dynamic_freeze > 0:
            if self.trainer.current_epoch < self.dynamic_freeze:
                for parameter in self.model.parameters():
                    parameter.requires_grad = False
            else:
                for parameter in self.model.parameters():
                    parameter.requires_grad = True
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, entity_pos, hts, y, elems, relation_graph = batch
        y_hat, gating_score = self(input_ids, attention_mask, entity_pos, hts, relation_graph)

        gating_label = torch.max(y, dim=-1)[0]
        gating_loss = self.gating_loss_fn(gating_score.squeeze(-1), gating_label.float())

        gate_classification = (torch.sigmoid(gating_score) > 0.5).squeeze(-1)

        gate_tp = (gate_classification & gating_label.bool()).sum()
        gate_fp = (gate_classification & ~gating_label.bool()).sum()
        gate_fn = (~gate_classification & gating_label.bool()).sum()

        # y_hat = self.reproject(torch.stack([y_hat, y_hat_graph], dim=-1)).squeeze(-1)

        loss = self.loss_fn(y_hat, y)

        output = {}
        output["rel_pred"] = self.loss_fn.get_label(y_hat, num_labels=self.labels_to_predict)
        scores_topk = self.loss_fn.get_score(y_hat, self.labels_to_predict)
        output["scores"] = scores_topk[0]
        output["topks"] = scores_topk[1]

        loss = loss + self.cross_relation_weight * gating_loss
        self.outputs.append((loss, gating_loss,
                             output, hts, elems, gate_tp, gate_fp, gate_fn, y))

        return loss

    def predict_step(self, batch, batch_idx) -> STEP_OUTPUT:
        input_ids, attention_mask, entity_pos, hts, y, elems, relation_graph = batch
        y_hat, gating_score = self(input_ids, attention_mask, entity_pos, hts, relation_graph)

        prediction = self.loss_fn.get_label(y_hat, num_labels=self.labels_to_predict)

        return prediction, hts, elems

    def on_validation_epoch_end(self):
        loss = torch.stack([x[0] for x in self.outputs]).mean()
        gating_loss = torch.stack([x[1] for x in self.outputs]).mean()
        output = [x[2] for x in self.outputs]
        hts = [x[3] for x in self.outputs]
        elems = [x[4] for x in self.outputs]
        gate_tp = torch.stack([x[5] for x in self.outputs]).sum()
        gate_fp = torch.stack([x[6] for x in self.outputs]).sum()
        gate_fn = torch.stack([x[7] for x in self.outputs]).sum()
        y = torch.cat([x[8] for x in self.outputs], dim=0)
        y = y.bool().cpu().numpy()
        if self.evaluator is None:
            prediction = []
            for output_ in output:
                prediction.append(output_["rel_pred"])
            prediction = torch.cat(prediction, dim=0).cpu().numpy()
            # no_relation = prediction[:, 0]
            prediction[:, 0] = False
            if not self.has_no_label:
                y = np.concatenate([np.zeros((y.shape[0], 1), dtype=bool), y], axis=-1)
            tp = np.sum(prediction & y)
            fp = np.sum(prediction & ~y)
            fn = np.sum(~prediction & y)
            macro_tp = np.sum(prediction & y, axis=0)
            macro_fp = np.sum(prediction & ~y, axis=0)
            macro_fn = np.sum(~prediction & y, axis=0)
            encountered = macro_tp + macro_fp + macro_fn > 0
            macro_precisions = macro_tp / (macro_tp + macro_fp + 1e-10)
            macro_precision = macro_precisions.sum() / (encountered.sum() + 1e-10)
            macro_recalls = macro_tp / (macro_tp + macro_fn + 1e-10)
            macro_recall = macro_recalls.sum() / (encountered.sum() + 1e-10)
            macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-10)

            precision = tp.sum() / (tp.sum() + fp.sum() + 1e-10)
            recall = tp.sum() / (tp.sum() + fn.sum() + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            self.log_dict({"val_loss": loss, "val_gating_loss": gating_loss,
                           "val_precision": precision, "val_recall": recall, "val_f1": f1,
                           "val_gate_precision": gate_tp / (gate_tp + gate_fp + 1e-10),
                           "val_gate_recall": gate_tp / (gate_tp + gate_fn + 1e-10),
                           "val_gate_f1": 2 * gate_tp / (gate_tp + gate_fp + gate_fn + 1e-10),
                            "val_macro_precision": macro_precision, "val_macro_recall": macro_recall, "val_macro_f1": macro_f1


                           }, sync_dist=True, on_epoch=True,)

        else:
            prediction = []
            features = []
            scores = []
            topks = []
            for elem, hts_, output_ in zip(elems, hts, output):
                prediction.append(output_["rel_pred"])
                scores.append(output_["scores"])
                topks.append(output_["topks"])
                for x, y in zip(elem, hts_):
                    feature = {
                        "hts": y,
                        **x
                    }
                    features.append(feature)

            prediction = torch.cat(prediction, dim=0).cpu().numpy()
            topks = torch.cat(topks, dim=0).cpu().numpy()
            scores = torch.cat(scores, dim=0).cpu().numpy()


            official_results, results = self.evaluator.to_official(prediction, features, scores=scores, topks=topks)
            best_re, best_re_ign, _, best_re_macro = self.evaluator.official_evaluate(official_results)

            re_p, re_r, re_f1 = best_re
            re_p_ign, re_r_ign, re_f1_ign = best_re_ign
            re_p_macro, re_r_macro, re_f1_macro = best_re_macro

            gate_precision = gate_tp / (gate_tp + gate_fp + 1e-10)
            gate_recall = gate_tp / (gate_tp + gate_fn + 1e-10)
            gate_f1 = 2 * gate_precision * gate_recall / (gate_precision + gate_recall + 1e-10)

            self.log_dict({"val_loss": loss, "val_gating_loss": gating_loss,
                           "val_precision": re_p, "val_recall": re_r, "val_f1": re_f1,
                              "val_precision_ign": re_p_ign, "val_recall_ign": re_r_ign, "val_f1_ign": re_f1_ign,
                                "val_precision_macro": re_p_macro, "val_recall_macro": re_r_macro, "val_f1_macro": re_f1_macro,
                                "val_gate_precision": gate_precision, "val_gate_recall": gate_recall, "val_gate_f1": gate_f1
                           }, sync_dist=True, on_epoch=True,)
        self.outputs = []


    def configure_optimizers(self):
        params = [
            {'params': self.model.parameters(), 'lr': self.lr_encoder},
            {'params': self.graph_predictor.parameters(), 'lr': self.lr_classifier},
            {'params': self.predictor.parameters(), 'lr': self.lr_classifier},
            {'params': self.subject.parameters(), 'lr': self.lr_classifier},
            {'params': self.object.parameters(), 'lr': self.lr_classifier},
            {'params': self.bilinear.parameters(), 'lr': self.lr_classifier},
            {'params': self.relation_candidate_retriever.parameters(), 'lr': self.lr_classifier},
            {'params': self.no_path_embedding, 'lr': self.lr_classifier}
        ]
        optimizer = torch.optim.AdamW(params)

        params = {
            'optimizer': optimizer,
        }
        if not self.deactivate_scheduler:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps,
                                                        num_training_steps=self.total_steps)
            params['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'step',
            }

        return params



