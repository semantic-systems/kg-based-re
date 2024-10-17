import copy
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import nn, autograd
from torch.nn.utils.rnn import pad_sequence

from torch_scatter import scatter_add
from ULTRA.ultra.base_nbfnet import tasks, layers
from pytorch_lightning import LightningModule
class NBFPL(LightningModule):
    def __init__(self, model_args, train_loader, val_loader, test_loader=None, lr=1e-3):
        super().__init__()
        self.model = NBFAltNet(**model_args)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.lr = lr
        self.dev_outputs = []

    def forward(self, data):
        return self.model(data)

    def get_negatives(self, batch, num_negatives=16):
        target = torch.cat([batch.main_pair_index[1, :].unsqueeze(1), batch.negatives_index.T], dim=1)
        target = target.unsqueeze(-1)
        target = torch.cat((batch.main_pair_index[0, :].unsqueeze(-1).unsqueeze(-1).repeat(1, target.size(1), 1), target), dim=-1)
        target_relations = batch.main_relation
        target_relations = target_relations.unsqueeze(1).repeat(1, target.size(1))
        return batch, target, target_relations
    def training_step(self, batch, batch_idx):
        soft_negative_indices = []
        soft_relations = []
        offset = 0
        max_num_relations = max([x.data.size(0) for x in batch.soft_negatives_indices])
        for x in batch.soft_negatives_indices:
            soft_negative_indices.append(x.data + offset)
            soft_relations.append(torch.cat((x.relations, torch.zeros(max_num_relations - x.relations.size(0), dtype=torch.long))))
            offset += x.num_nodes
        soft_relations = torch.cat(soft_relations, dim=0).to(self.device)
        soft_negative_indices = pad_sequence(soft_negative_indices, batch_first=True, padding_value=-1).to(self.device)
        soft_negatives_mask = soft_negative_indices == -1
        soft_negative_indices[soft_negatives_mask] = 0
        data, batch, target_relations = self.get_negatives(batch)
        batch_mask = batch == -1
        batch[batch_mask] = 0
        output, soft_outputs = self.model(data, batch, soft_indices=soft_negative_indices)
        target = torch.zeros([batch.size(0), batch.size(1)], device=self.device)
        target[:, 0] = 1.0

        output_shape = output.size()
        head_output = output[:, 0]
        head_labels = (target_relations[:, 0].unsqueeze(-1) == torch.arange(self.model.num_output_relation, device=self.device).unsqueeze(0).repeat(output_shape[0], 1))
        head_output = head_output[~head_labels]
        head_labels = head_labels[~head_labels].float()
        output = output.view(-1, output.size(-1))
        relation_mask = target_relations.view(-1).unsqueeze(-1).repeat(1, self.model.num_output_relation) == torch.arange(
            self.model.num_output_relation, device=self.device).unsqueeze(0).repeat(output_shape[0] * output_shape[1], 1)
        output = output[relation_mask].view(output_shape[0], output_shape[1])
        output[batch_mask[:, :, 1]] = -1e9

        soft_outputs_shape = soft_outputs.size()
        soft_outputs = soft_outputs.view(-1, soft_outputs.size(-1))
        relation_mask = soft_relations.view(-1).unsqueeze(-1).repeat(1, self.model.num_output_relation) == torch.arange(
            self.model.num_output_relation, device=self.device).unsqueeze(0).repeat(
            soft_outputs_shape[0] * soft_outputs_shape[1], 1)
        soft_outputs = soft_outputs[relation_mask].view(soft_outputs_shape[0], soft_outputs_shape[1])
        soft_outputs[soft_negatives_mask[:, :]] = -1e9
        soft_targets = torch.zeros_like(soft_outputs)


        loss = self.loss(output, target)
        soft_loss = self.loss(soft_outputs, soft_targets).mean()
        valid_per_dim = torch.sum(~batch_mask[:, :, 1], dim=1)
        loss = (loss[:, 0] + loss[:, 1:].sum(dim=-1) / (valid_per_dim - 1)).mean()
        other_loss = torch.mean(self.loss(head_output, head_labels))
        loss += other_loss
        loss += soft_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_other_loss", other_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_soft_loss", soft_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, orig_batch, batch_idx):
        soft_negative_indices = []
        soft_relations = []
        offset = 0
        for x in orig_batch.soft_negatives_indices:
            soft_negative_indices.append(x.data + offset)
            soft_relations.append(x.relations)
            offset += x.num_nodes

        soft_relations = torch.cat(soft_relations, dim=0).to(self.device)
        soft_negative_indices = pad_sequence(soft_negative_indices, batch_first=True, padding_value=-1).to(self.device)
        soft_negatives_mask = soft_negative_indices == -1
        soft_negative_indices[soft_negatives_mask] = 0
        data, batch, target_relations = self.get_negatives(orig_batch)

        batch_mask = batch == -1
        batch[batch_mask] = 0
        output, soft_outputs = self.model(data, batch, soft_indices=soft_negative_indices)
        target = torch.zeros([batch.size(0), batch.size(1)], device=self.device)
        target[:, 0] = 1.0

        output_shape = output.size()
        head_output = output[:, 0]
        head_labels = (target_relations[:, 0] == torch.arange(self.model.num_output_relation, device=self.device).unsqueeze(0).repeat(output_shape[0], 1))
        head_output = head_output[~head_labels]
        head_labels = head_labels[~head_labels].float()

        output = output.view(-1, output.size(-1))
        relation_mask = target_relations.view(-1).unsqueeze(-1).repeat(1, self.model.num_output_relation) == torch.arange(self.model.num_output_relation, device=self.device).unsqueeze(0).repeat(output_shape[0] * output_shape[1], 1)
        output = output[relation_mask].view(output_shape[0], output_shape[1])
        output[batch_mask[:, :, 1]] = -1e9

        soft_outputs_shape = soft_outputs.size()
        soft_outputs = soft_outputs.view(-1, soft_outputs.size(-1))
        relation_mask = soft_relations.view(-1).unsqueeze(-1).repeat(1, self.model.num_output_relation) == torch.arange(self.model.num_output_relation, device=self.device).unsqueeze(0).repeat(soft_outputs_shape[0] * soft_outputs_shape[1], 1)
        soft_outputs = soft_outputs[relation_mask].view(soft_outputs_shape[0], soft_outputs_shape[1])
        soft_outputs[soft_negatives_mask[:, :]] = -1e9
        soft_targets = torch.zeros_like(soft_outputs)



        ranks = torch.zeros_like(torch.cat((output, soft_outputs), dim=-1), dtype=torch.long, device=self.device)

        for i in range(output.size(0)):
            batched_outputs = torch.cat((output[i], soft_outputs[i]), dim=0)

            # Sort the current batch and get the sorted indices
            _, sorted_indices = torch.sort(batched_outputs, descending=True)

            # Assign ranks based on sorted indices
            ranks[i][sorted_indices] = torch.arange(batched_outputs.size(0), device=self.device)

        alt_ranks = torch.zeros_like(output, dtype=torch.long, device=self.device)

        for i in range(output.size(0)):
            batched_outputs = output[i]

            # Sort the current batch and get the sorted indices
            _, sorted_indices = torch.sort(batched_outputs, descending=True)

            # Assign ranks based on sorted indices
            alt_ranks[i][sorted_indices] = torch.arange(batched_outputs.size(0), device=self.device)

        self.dev_outputs.append({
            "with_soft_rank": ranks[:, 0],
            "without_soft_rank": alt_ranks[:, 0],
            "relations": target_relations[:, 0],
        })

        soft_loss = self.loss(soft_outputs, soft_targets).mean()
        loss = self.loss(output, target)
        valid_per_dim = torch.sum(~batch_mask[:, :, 1], dim=1)
        loss = (loss[:, 0] + loss[:, 1:].sum(dim=-1)/(valid_per_dim-1)).mean()
        other_loss = torch.mean(self.loss(head_output, head_labels))
        loss += other_loss
        loss += soft_loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_other_loss", other_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_soft_loss", soft_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        with_soft_ranks = []
        without_soft_ranks = []
        relations = []
        for x in self.dev_outputs:
            with_soft_ranks.append(x["with_soft_rank"])
            without_soft_ranks.append(x["without_soft_rank"])
            relations.append(x["relations"])
        without_soft_ranks = torch.cat(without_soft_ranks)
        with_soft_ranks = torch.cat(with_soft_ranks)
        relations = torch.cat(relations)
        hits_at_1 = (with_soft_ranks == 0).float().mean()
        hits_at_3 = (with_soft_ranks < 3).float().mean()
        hits_at_10 = (with_soft_ranks < 10).float().mean()
        MRR = (1.0 / (with_soft_ranks.float() + 1)).mean()

        non_soft_hits_at_1 = (without_soft_ranks == 0).float().mean()
        non_soft_hits_at_3 = (without_soft_ranks < 3).float().mean()
        non_soft_hits_at_10 = (without_soft_ranks < 10).float().mean()
        non_soft_MRR = (1.0 / (without_soft_ranks.float() + 1)).mean()

        macro_with_soft_ranks = defaultdict(list)
        macro_without_soft_ranks = defaultdict(list)

        for with_soft_rank, without_soft_rank, relation in zip(with_soft_ranks, without_soft_ranks, relations):
            macro_with_soft_ranks[relation.item()].append(with_soft_rank)
            macro_without_soft_ranks[relation.item()].append(without_soft_rank)

        macro_hits_at_1 = []
        macro_hits_at_3 = []
        macro_hits_at_10 = []
        macro_MRR = []

        for relation in macro_with_soft_ranks:
            macro_hits_at_1.append((torch.tensor(macro_with_soft_ranks[relation]) == 0).float().mean())
            macro_hits_at_3.append((torch.tensor(macro_with_soft_ranks[relation]) < 3).float().mean())
            macro_hits_at_10.append((torch.tensor(macro_with_soft_ranks[relation]) < 10).float().mean())
            macro_MRR.append((1.0 / (torch.tensor(macro_with_soft_ranks[relation]).float() + 1)).mean())

        macro_hits_at_1 = torch.stack(macro_hits_at_1).mean()
        macro_hits_at_3 = torch.stack(macro_hits_at_3).mean()
        macro_hits_at_10 = torch.stack(macro_hits_at_10).mean()
        macro_MRR = torch.stack(macro_MRR).mean()

        macro_non_soft_hits_at_1 = []
        macro_non_soft_hits_at_3 = []
        macro_non_soft_hits_at_10 = []
        macro_non_soft_MRR = []

        for relation in macro_without_soft_ranks:
            macro_non_soft_hits_at_1.append((torch.tensor(macro_without_soft_ranks[relation]) == 0).float().mean())
            macro_non_soft_hits_at_3.append((torch.tensor(macro_without_soft_ranks[relation]) < 3).float().mean())
            macro_non_soft_hits_at_10.append((torch.tensor(macro_without_soft_ranks[relation]) < 10).float().mean())
            macro_non_soft_MRR.append((1.0 / (torch.tensor(macro_without_soft_ranks[relation]).float() + 1)).mean())

        macro_non_soft_hits_at_1 = torch.stack(macro_non_soft_hits_at_1).mean()
        macro_non_soft_hits_at_3 = torch.stack(macro_non_soft_hits_at_3).mean()
        macro_non_soft_hits_at_10 = torch.stack(macro_non_soft_hits_at_10).mean()
        macro_non_soft_MRR = torch.stack(macro_non_soft_MRR).mean()



        self.log("hits_at_1", hits_at_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("hits_at_3", hits_at_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("hits_at_10", hits_at_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log("MRR", MRR, on_step=False, on_epoch=True, prog_bar=True)
        self.log("non_soft_hits_at_1", non_soft_hits_at_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("non_soft_hits_at_3", non_soft_hits_at_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("non_soft_hits_at_10", non_soft_hits_at_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log("non_soft_MRR", non_soft_MRR, on_step=False, on_epoch=True, prog_bar=True)
        self.log("macro_hits_at_1", macro_hits_at_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("macro_hits_at_3", macro_hits_at_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("macro_hits_at_10", macro_hits_at_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log("macro_MRR", macro_MRR, on_step=False, on_epoch=True, prog_bar=True)
        self.log("macro_non_soft_hits_at_1", macro_non_soft_hits_at_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("macro_non_soft_hits_at_3", macro_non_soft_hits_at_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("macro_non_soft_hits_at_10", macro_non_soft_hits_at_10, on_step=False, on_epoch=True, prog_bar=True)
        self.log("macro_non_soft_MRR", macro_non_soft_MRR, on_step=False, on_epoch=True, prog_bar=True)

        self.dev_outputs = []

    def test_step(self, batch, batch_idx):
        data, batch, target_relations = self.get_negatives(batch)
        output = self.model(data, batch)
        target = torch.zeros([batch.size(0), batch.size(1)], device=self.device)
        target[:, 0] = 1.0

        output_shape = output.size()
        output = output.view(-1, output.size(-1))
        relation_mask = target_relations.view(-1).unsqueeze(-1).repeat(1, self.model.num_relation) == torch.arange(
            self.model.num_relation, device=self.device).unsqueeze(0).repeat(output_shape[0] * output_shape[1], 1)
        output = output[relation_mask].view(output_shape[0], output_shape[1])

        loss = self.loss(output, target)
        loss = (loss[:, 0] + loss[:, 1:].mean(dim=-1)).mean()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader



class NBFAltNet(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_relation, num_output_relation, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, remove_one_hop=False, num_beam=10, path_topk=10, num_rep: int = 3, score=False):
        super(NBFAltNet, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.num_output_relation = num_output_relation
        self.short_cut = short_cut  # whether to use residual connections between GNN layers
        self.concat_hidden = concat_hidden  # whether to compute final states as a function of all layer outputs or last
        self.remove_one_hop = remove_one_hop  # whether to dynamically remove one-hop edges from edge_index
        self.num_beam = num_beam
        self.path_topk = path_topk

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], 2 * num_relation,
                                                                self.dims[0], message_func, aggregate_func, layer_norm,
                                                                activation, dependent))

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1])

        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        self.query = nn.Embedding(2 * num_relation, input_dim)
        self.start_query = nn.Embedding(num_rep, input_dim)
        self.num_rep = num_rep
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, feature_dim))
        self.mlp = nn.Sequential(*mlp)
        self.score = score
        self.scorer = nn.Linear(feature_dim, num_output_relation)

    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        # we remove training edges (we need to predict them at training time) from the edge index
        # think of it as a dynamic edge dropout
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        # r_index_ext = torch.cat([r_index, r_index + self.num_relation // 2], dim=-1)
        # if self.remove_one_hop:
        # we remove all existing immediate edges between heads and tails in the batch
        edge_index = data.edge_index
        easy_edge = torch.stack([h_index_ext, t_index_ext]).flatten(1)
        index = tasks.edge_match(edge_index, easy_edge)[0]
        mask = ~index_to_mask(index, data.num_edges)
        # else:
        #     # we remove existing immediate edges between heads and tails in the batch with the given relation
        #     edge_index = torch.cat([data.edge_index, data.edge_type[:,0].int().unsqueeze(0)])
        #     # note that here we add relation types r_index_ext to the matching query
        #     easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
        #     easy_edge = easy_edge[:, torch.any(easy_edge != -1, dim=0)]
        #     index = tasks.edge_match(edge_index, easy_edge)[0]
        #     mask = ~index_to_mask(index, data.num_edges)

        data = copy.copy(data)
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def negative_sample_to_tail(self, h_index, t_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        return new_h_index, new_t_index

    def bellmanford(self, data, h_index, query, separate_grad=False):
        batch_size = len(h_index)

        # initialize queries (relation types of the given triples)
        query = self.start_query(query)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        assert index.size() == query.size()
        assert torch.max(index) < boundary.size(1)
        assert not torch.any(index == -1)

        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary
        path_boundary = torch.zeros(batch_size, data.num_nodes, device=h_index.device)
        path_boundary = path_boundary.scatter(1, index, torch.ones_like(index, device=index.device).float())


        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden, path_boundary = layer(layer_input, query, boundary, path_boundary, data.edge_index, data.edge_type, size, edge_weight, data.edge_attr)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens, dim=-1)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
            "path_boundary": path_boundary,
        }

    def forward(self, data, batch, soft_indices=None):
        original_size = batch.size(0)
        batch = batch.repeat(self.num_rep, 1, 1)
        query = torch.arange(self.num_rep, device=batch.device)
        query = torch.repeat_interleave(query, batch.shape[0] // self.num_rep, dim=0)

        h_index, t_index = batch.unbind(-1)
        if self.training and self.remove_one_hop:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index = self.negative_sample_to_tail(h_index, t_index)
        assert (h_index[:, [0]] == h_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], query)  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        scoped_feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        final_features = []
        for i in range(self.num_rep):
            final_features.append(scoped_feature[i * original_size:(i + 1) * original_size, :, :])
        final_features = torch.mean(torch.stack(final_features), dim=0)
        features = self.mlp(final_features).squeeze(-1)
        if self.score:
            if soft_indices is not None:
                soft_feature = feature.gather(1, soft_indices.unsqueeze(-1).expand(-1, -1, feature.shape[-1]))
                final_features = []
                for i in range(self.num_rep):
                    final_features.append(soft_feature[i * original_size:(i + 1) * original_size, :, :])
                final_features = torch.mean(torch.stack(final_features), dim=0)
                soft_feature = self.mlp(final_features).squeeze(-1)
                return self.scorer(features), self.scorer(soft_feature)

            return self.scorer(features)
        if "path_boundary" in output:
            final_path_boundaries = []
            for i in range(self.num_rep):
                path_boundary = output["path_boundary"][i * original_size:(i + 1) * original_size, :]
                path_boundary = path_boundary.unsqueeze(-1)
                path_index = t_index[i * original_size:(i + 1) * original_size, :].unsqueeze(-1).expand(-1, -1, path_boundary.shape[-1])
                path_boundary = path_boundary.gather(1, path_index)
                final_path_boundaries.append(path_boundary)
            path_boundary = torch.max(torch.stack(final_path_boundaries), dim=0).values
            return features, path_boundary
        return features

    def visualize(self, data, batch):
        assert batch.shape == (1, 3)
        h_index, t_index, r_index = batch.unbind(-1)

        output = self.bellmanford(data, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        edge_weights = output["edge_weights"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_grads = autograd.grad(score, edge_weights)
        distances, back_edges = self.beam_search_distance(data, edge_grads, h_index, t_index, self.num_beam)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)

        return paths, weights

    @torch.no_grad()
    def beam_search_distance(self, data, edge_grads, h_index, t_index, num_beam=10):
        # beam search the top-k distance from h to t (and to every other node)
        num_nodes = data.num_nodes
        input = torch.full((num_nodes, num_beam), float("-inf"), device=h_index.device)
        input[h_index, 0] = 0
        edge_mask = data.edge_index[0, :] != t_index

        distances = []
        back_edges = []
        for edge_grad in edge_grads:
            # we don't allow any path goes out of t once it arrives at t
            node_in, node_out = data.edge_index[:, edge_mask]
            relation = data.edge_type[edge_mask]
            edge_grad = edge_grad[edge_mask]

            message = input[node_in] + edge_grad.unsqueeze(-1) # (num_edges, num_beam)
            # (num_edges, num_beam, 3)
            msg_source = torch.stack([node_in, node_out, relation], dim=-1).unsqueeze(1).expand(-1, num_beam, -1)

            # (num_edges, num_beam)
            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            # pick the first occurrence as the ranking in the previous node's beam
            # this makes deduplication easier later
            # and store it in msg_source
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=message.device) / (num_beam + 1)
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1) # (num_edges, num_beam, 4)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort messages w.r.t. node_out
            message = message[order].flatten() # (num_edges * num_beam)
            msg_source = msg_source[order].flatten(0, -2) # (num_edges * num_beam, 4)
            size = node_out.bincount(minlength=num_nodes)
            msg2out = size_to_index(size[node_out_set] * num_beam)
            # deduplicate messages that are from the same source and the same beam
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=message.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = msg2out.bincount(minlength=len(node_out_set))

            if not torch.isinf(message).all():
                # take the topk messages from the neighborhood
                # distance: (len(node_out_set) * num_beam)
                distance, rel_index = scatter_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                # store msg_source for backtracking
                back_edge = msg_source[abs_index] # (len(node_out_set) * num_beam, 4)
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                # scatter distance / back_edge back to all nodes
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_nodes) # (num_nodes, num_beam)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_nodes) # (num_nodes, num_beam, 4)
            else:
                distance = torch.full((num_nodes, num_beam), float("-inf"), device=message.device)
                back_edge = torch.zeros(num_nodes, num_beam, 4, dtype=torch.long, device=message.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        # backtrack distances and back_edges to generate the paths
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def size_to_index(size):
    range = torch.arange(len(size), device=size.device)
    index2sample = range.repeat_interleave(size)
    return index2sample


def multi_slice_mask(starts, ends, length):
    values = torch.cat([torch.ones_like(starts), -torch.ones_like(ends)])
    slices = torch.cat([starts, ends])
    mask = scatter_add(values, slices, dim=0, dim_size=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def scatter_extend(data, size, input, input_size):
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = torch.zeros(new_cum_size[-1], *data.shape[1:], dtype=data.dtype, device=data.device)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = input
    return new_data, new_size


def scatter_topk(input, size, k, largest=True):
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (input.ndim - 1))

    mask = ~torch.isinf(input)
    max = input[mask].max().item()
    min = input[mask].min().item()
    safe_input = input.clamp(2 * min - max, 2 * max - min)
    offset = (max - min) * 4
    if largest:
        offset = -offset
    input_ext = safe_input + offset * index2graph
    index_ext = input_ext.argsort(dim=0, descending=largest)
    num_actual = size.clamp(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        # special case: size < k, pad with the last valid index
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = scatter_extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask] # (N * k, ...)
    value = input.gather(0, index)
    if isinstance(k, torch.Tensor) and k.shape == size.shape:
        value = value.view(-1, *input.shape[1:])
        index = index.view(-1, *input.shape[1:])
        index = index - (size.cumsum(0) - size).repeat_interleave(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *input.shape[1:])
        index = index.view(-1, k, *input.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index