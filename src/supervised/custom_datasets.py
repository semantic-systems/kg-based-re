import copy
import json
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Set

import graph_tool.all as gt
import pytorch_lightning as pl
import torch
# from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_geometric.data import Data, Batch

from src.supervised.utils.docred.entity_types import get_class_superclasses, get_valid_types_refined, SPECIAL_ENTITIES


def create_meta_graph(hts):
    return

def custom_collate_fn(batch):
    input_ids = [elem["input_ids"] for elem in batch]
    entity_indices = deepcopy([elem["entity_indices"] for elem in batch])
    y = torch.cat([torch.stack(elem["relation_labels"], 0) for elem in batch], 0).float()
    hts = deepcopy([elem["hts"] for elem in batch])
    local_graphs = [elem["local_graph"] for elem in batch]
    elems = [elem["elem"] for elem in batch]



    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-1)
    attention_mask = (input_ids != -1).int()
    input_ids[input_ids == -1] = 0

    local_graph = Batch.from_data_list(local_graphs)
    alternative_hts = []
    offset = 0
    for elem, local_graph_ in zip(hts, local_graphs):
        for h, t in elem:
            alternative_hts.append((h + offset, t + offset))
        offset += local_graph_.num_nodes

    graph_info = (local_graph, alternative_hts)
    return input_ids, attention_mask, entity_indices, hts, y, elems, graph_info


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, examples, graph, graph_relation_indices, num_hops=2, limit=50,
                 downsample=False, remove_direct_links=False, precached_neighborhood=None,
                 relations_filter_set=None):
        self.examples = examples
        self.graph = graph
        self.limit = limit
        self.graph_relation_indices = graph_relation_indices
        self.valid_pids = None
        if isinstance(graph_relation_indices, dict) and "valid_props" in graph_relation_indices and "prop_idx" in graph_relation_indices:
            self.valid_pids = set(graph_relation_indices["valid_props"])
            self.graph_relation_indices = {int(x): y for x,y in graph_relation_indices["prop_idx"].items()}
        self.num_hops = num_hops
        self.remove_direct_links = remove_direct_links
        self.downsample = downsample
        self.precached_neighborhood = precached_neighborhood
        self.relations_filter_set = relations_filter_set


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = copy.deepcopy(self.examples[idx])
        all_identifiers = example["all_identifiers"]
        entity_index = example["entity_index"]
        direct_links = set()
        for (s, o), target in zip(example["hts"], example["relation_labels"]):
            if torch.any(target == 1):
                direct_links.add((s, o))
        if self.precached_neighborhood is not None:
            triples = set()
            for identifier in all_identifiers:
                triples.update(self.precached_neighborhood.get(identifier, []))
        else:
            if self.downsample:
                sampled_all_identifiers = set(random.sample(all_identifiers, min(len(all_identifiers), 30)))
            else:
                sampled_all_identifiers = all_identifiers
            triples = self.get_two_hop(sampled_all_identifiers)
        occurs_in_triples = defaultdict(int)
        for s, p, o in triples:
            occurs_in_triples[s] += 1
            occurs_in_triples[o] += 1
        nodes_counter = max([max(v) for v in entity_index.values()]) + 1
        original_nodes = set(entity_index.keys())
        edge_index = []
        edge_type = []
        for s, p, o in triples:
            if (s not in original_nodes and o not in original_nodes) and (occurs_in_triples[s] < 3 or occurs_in_triples[o] < 3):
                continue

            if s not in entity_index:
                entity_index[s].append(nodes_counter)
                nodes_counter += 1
            if o not in entity_index:
                entity_index[o].append(nodes_counter)
                nodes_counter += 1
            for x in entity_index[s]:
                for y in entity_index[o]:
                    if x != y:
                        if self.remove_direct_links and (x, y) in direct_links:
                            continue
                        edge_index.append((x, y))
                        edge_type.append(p)

        if not edge_index:
            edge_index = torch.tensor([0, 0]).unsqueeze(-1)
            edge_type = torch.tensor([0])
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.int64)
        num_nodes = nodes_counter
        local_graph = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes)
        example["local_graph"] = local_graph
        return example

    def get_two_hop(self, qids: Set[int]):
        triples = set()
        other_qids = set()

        for qid in qids:
            if not isinstance(qid, int):
                qid = int(qid[1:])
            if qid < self.graph.num_vertices():
                out_edges = self.graph.get_out_edges(qid, [self.graph.ep["pid"]])
                in_edges = self.graph.get_in_edges(qid, [self.graph.ep["pid"]])
                if self.valid_pids is not None:
                    out_edges = [x for x in out_edges if x[2] in self.valid_pids]
                    in_edges = [x for x in in_edges if x[2] in self.valid_pids]
                if self.relations_filter_set is not None:
                    out_edges = [x for x in out_edges if x[2] in self.relations_filter_set]
                    in_edges = [x for x in in_edges if x[2] in self.relations_filter_set]

                out_edges = out_edges[:self.limit]
                in_edges = in_edges[:self.limit]
                # if self.downsample:
                #     out_edges = random.sample(list(out_edges), min(len(out_edges), 10))
                #     in_edges = random.sample(list(in_edges), min(len(in_edges), 10))
                for edge in out_edges:
                    head, tail, pid = edge
                    triples.add((qid, self.graph_relation_indices[pid], tail))
                    triples.add((tail, self.graph_relation_indices[pid] + len(self.graph_relation_indices), qid))
                    other_qids.add(tail)
                for edge in in_edges:
                    head, tail, pid = edge
                    triples.add((head, self.graph_relation_indices[pid], qid))
                    triples.add((qid, self.graph_relation_indices[pid] + len(self.graph_relation_indices), head))
                    other_qids.add(head)
        existing_qids = qids
        for i in range(self.num_hops - 1):
            other_qids = other_qids.difference(existing_qids)
            new_other_qids = set()
            for other_qid in other_qids:
                out_edges = self.graph.get_out_edges(other_qid, [self.graph.ep["pid"]])
                in_edges = self.graph.get_in_edges(other_qid, [self.graph.ep["pid"]])
                if self.valid_pids is not None:
                    out_edges = [x for x in out_edges if x[2] in self.valid_pids]
                    in_edges = [x for x in in_edges if x[2] in self.valid_pids]
                if self.relations_filter_set is not None:
                    out_edges = [x for x in out_edges if x[2] in self.relations_filter_set]
                    in_edges = [x for x in in_edges if x[2] in self.relations_filter_set]
                out_edges = out_edges[:self.limit]
                in_edges = in_edges[:self.limit]
                if self.downsample:
                    out_edges = random.sample(list(out_edges), min(len(out_edges), 10))
                    in_edges = random.sample(list(in_edges), min(len(in_edges), 10))
                for edge in out_edges:
                    head, tail, pid = edge
                    triples.add((other_qid, self.graph_relation_indices[pid], tail))
                    triples.add((tail, self.graph_relation_indices[pid] + len(self.graph_relation_indices), other_qid))
                    new_other_qids.add(tail)
                for edge in in_edges:
                    head, tail, pid = edge
                    triples.add((head, self.graph_relation_indices[pid], other_qid))
                    triples.add((other_qid, self.graph_relation_indices[pid] + len(self.graph_relation_indices), head))
                    new_other_qids.add(head)
            existing_qids.update(other_qids)
            other_qids = new_other_qids
        return triples


class Dataset(pl.LightningDataModule):
    def __init__(self, tokenizer, relation_dict, max_length: int = 512, batch_size: int = 8, num_workers=4,
                 add_no_relation=False, only_existing_triples=False, num_hops=2, downsample=False, remove_direct_links=False):
        super().__init__()
        self.train_dataloader_ = None
        self.val_dataloader_ = None
        self.test_dataloader_ = None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.add_no_relation = add_no_relation
        self.superclasses_dict = get_class_superclasses()
        self.valid_types, self.types_index = get_valid_types_refined()

        self.relation_dict = relation_dict
        self.num_relations = len(relation_dict) + add_no_relation
        self.num_workers = num_workers

        self.val_path = None
        self.train_path = None
        self.test_path = None
        self.graph = None
        self.graph_relation_indices = None
        self.only_existing_triples = only_existing_triples
        self.num_hops = num_hops
        self.downsample = downsample
        self.remove_direct_links = remove_direct_links
        self.relations_filter_set = None


    def add_entity_markers(self, example, entity_start, entity_end):
        ''' add entity marker (*) at the end and beginning of entities. '''

        sents = []
        sent_map = []
        sent_pos = []

        sent_start = 0
        for i_s, sent in enumerate(example['sents']):
            # add * marks to the beginning and end of entities
            new_map = {}

            for i_t, token in enumerate(sent):
                tokens_wordpiece = self.tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)

            sent_end = len(sents)
            # [sent_start, sent_end)
            sent_pos.append((sent_start, sent_end,))
            sent_start = sent_end

            # update the start/end position of each token.
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        return sents, sent_map, sent_pos

    def prune_triples(self, all_triples, main_qids, entities_in_subgraph):
        last_length = len(all_triples)
        new_length = 0
        while new_length != last_length:
            triples_of_entity = defaultdict(set)
            for (s, p, o) in all_triples:
                triples_of_entity[s].add((s, p, o))
                triples_of_entity[o].add((s, p, o))

            for entity, triples in triples_of_entity.items():
                if entity in main_qids:
                    continue
                if len(triples) == 1:
                    all_triples.difference_update(triples)

            last_length = new_length
            new_length = len(all_triples)
        return all_triples

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_

    def test_dataloader(self):
        return self.test_dataloader_

    def setup(self, stage=None):
        if self.val_dataloader_ is not None and self.train_dataloader_ is not None and self.test_dataloader_ is not None:
            return
        self.prepare_data()

        self.val_dataloader_ = self.load_dataset(self.val_path.replace(".json", "_preprocessed.json"), shuffle=False)
        self.train_dataloader_ = self.load_dataset(self.train_path.replace(".json", "_preprocessed.json"), downsample=self.downsample)
        self.test_dataloader_ = self.load_dataset(self.test_path.replace(".json", "_preprocessed.json"), test=True)

    def prepare_dataset(self, path: str):
        raise NotImplementedError

    def prepare_data(self) -> None:
        if not os.path.exists(self.val_path.replace(".json", "_preprocessed.json")):
            self.prepare_dataset(self.val_path)

        if not os.path.exists(self.test_path.replace(".json", "_preprocessed.json")):
            self.prepare_dataset(self.test_path)

        if not os.path.exists(self.train_path.replace(".json", "_preprocessed.json")):
            self.prepare_dataset(self.train_path)

    def load_dataset(self, path: str, test=False, shuffle=True, downsample=False):
        data = json.load(open(path))
        num_entities = 0
        num_with_qids = 0
        examples = []
        for idx, elem in enumerate(tqdm(data)):
            entity_index = defaultdict(list)
            all_identifiers = set()
            no_qid = 0
            for idx, item in enumerate(elem["entity_identifiers"]):
                if not item:
                    new_qid = f"no_qid_{no_qid}"
                    no_qid += 1
                    entity_index[new_qid].append(idx)
                else:
                    num_with_qids += 1
                    for identifier in set(item):
                        if isinstance(identifier, str):
                            identifier = int(identifier[1:])
                        entity_index[identifier].append(idx)
                        all_identifiers.add(identifier)
                num_entities += 1

            entity_indices = []
            entity_sentence = []
            for v in elem["entity_indices"]:
                new_v = []
                already_seen = set()
                for x in v:
                    if x[1] not in already_seen and x[1] < self.max_length:
                        new_v.append(x)
                        already_seen.add(x[1])
                entity_indices.append([x[1] for x in new_v])
                entity_sentence.append([x[0] for x in new_v])

            invalid_entity_indices = [len(v) == 0 for v in entity_indices]

            filtered_relation_labels = defaultdict(list)
            for relation_label in elem["relation_labels"]:
                if not invalid_entity_indices[relation_label[0]] and not invalid_entity_indices[relation_label[1]]:
                    filtered_relation_labels[(relation_label[0], relation_label[1])].append(relation_label[2])


            all_relation_labels = []
            hts = []
            negative_hts = []
            negative_relation_labels = []
            for entity_1_idx, entity_1 in enumerate(entity_indices):
                for entity_2_idx, entity_2 in enumerate(entity_indices):
                    if entity_1_idx == entity_2_idx:
                        continue
                    all_labels = torch.tensor([0] * self.num_relations)
                    if (entity_1_idx, entity_2_idx) in filtered_relation_labels:
                        for label in filtered_relation_labels[(entity_1_idx, entity_2_idx)]:
                            all_labels[label + self.add_no_relation] = 1
                        all_relation_labels.append(all_labels)
                        hts.append((entity_1_idx, entity_2_idx))
                    elif self.only_existing_triples:
                        continue
                    else:
                        negative_relation_labels.append(all_labels)
                        negative_hts.append((entity_1_idx, entity_2_idx))
            all_relation_labels.extend(negative_relation_labels)
            hts.extend(negative_hts)
            if not all_relation_labels:
                continue

            if filtered_relation_labels:
                elem["relation_labels"] = all_relation_labels
            else:
                continue

            elem["hts"] = hts
            elem["entity_indices"] = entity_indices
            elem["entity_sentence"] = entity_sentence
            input_ids = elem["input_ids"]
            input_ids = input_ids[:self.max_length - 2]
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
            elem["input_ids"] = torch.tensor(input_ids)
            elem["entity_index"] = entity_index
            elem["all_identifiers"] = all_identifiers
            elem["elem"] = elem.get("elem", None)
            examples.append(elem)

        if os.path.exists(path.replace(".json", "_cache.pkl")):
            precached = pickle.load(open(path.replace(".json", "_cache.pkl"), "rb"))
            precached = {x: set(y) for x, y in precached.items()}
        else:
            precached = None

        print(f"Ratio of entities with qids: {num_with_qids / num_entities:.2f}")

        return DataLoader(CustomDataset(examples, self.graph, self.graph_relation_indices,
                                        num_hops=self.num_hops, downsample=downsample, precached_neighborhood=precached,
                                        remove_direct_links=self.remove_direct_links,
                                        relations_filter_set=self.relations_filter_set), shuffle=shuffle, batch_size=self.batch_size, collate_fn=partial(custom_collate_fn),
                          num_workers=self.num_workers, )




class DocREDDataset(Dataset):
    def __init__(self, tokenizer, relation_dict, max_length: int = 512, batch_size: int = 8, num_workers=4, use_gold=True,
                 add_no_relation=False, remove_direct_links=False, use_wikidata5m=False):
        super().__init__(tokenizer, relation_dict, max_length, batch_size, num_workers, add_no_relation,
                         remove_direct_links=remove_direct_links)
        if use_gold:
            self.train_path = "data/docred/train_annotated.json"
        else:
            self.train_path = "data/docred/train_distant_qid.json"
        self.val_path = "data/docred/dev.json"
        self.test_path = "data/docred/test.json"

        if use_wikidata5m:
            self.graph_relation_indices = json.load(open("data/wikidata5m_transductive/prop_idx.json"))
            self.graph = gt.load_graph("data/wikidata5m_graph.gt")
        else:
            self.graph_relation_indices = {int(x): v for x, v in
                                           json.load(open("data/docred/prop_dict_all.json")).items()}
            self.graph = gt.load_graph("data/docred/full_graph_all.gt")

    def prepare_dataset(self, path: str):
        data = json.load(open(path))

        examples = []

        for elem in tqdm(data):
            entity_start = set()
            entity_end = set()
            entity_indices = defaultdict(list)
            entity_identifiers = []
            for idx, entity in enumerate(elem["vertexSet"]):
                qids = set()
                for occurrence in entity:
                    entity_start.add((occurrence["sent_id"], occurrence["pos"][0]))
                    entity_end.add((occurrence["sent_id"], occurrence["pos"][1] - 1))
                    entity_indices[idx].append((occurrence["sent_id"], occurrence["pos"][0]))
                    if occurrence["type"] in SPECIAL_ENTITIES:
                        continue
                    if not isinstance(occurrence["qid"], list):
                        if occurrence["qid"] is not None and occurrence["qid"].startswith("Q") and occurrence["qid"][1:].isdigit():
                            qids.add(occurrence["qid"])
                    else:
                        raw_qids = occurrence["qid"]
                        raw_qids = [x for x in raw_qids if x[0].startswith("Q") and x[0][1:].isdigit()]
                        for qid, score in raw_qids[:3]:
                            qids.add(qid)
                entity_identifiers.append(list(qids))

            sents, sent_map, sent_pos = self.add_entity_markers(elem, entity_start, entity_end)

            sent_map = [{k: v + 1 for k, v in sent_map_.items()} for sent_map_ in sent_map]
            input_ids = self.tokenizer.convert_tokens_to_ids(sents)

            entity_indices = {k: list(sorted([[entity[0], sent_map[entity[0]][entity[1]]] for entity in v])) for k, v in entity_indices.items()}

            final_entity_indices = []
            for idx in range(len(entity_indices)):
                final_entity_indices.append(entity_indices[idx])

            relation_labels = []
            for label in elem.get("labels", []):
                head = label["h"]
                tail = label["t"]
                relation_labels.append((head, tail, self.relation_dict[label["r"]]))

            example = {
                "input_ids": input_ids,
                "entity_indices": final_entity_indices,
                "relation_labels": relation_labels,
                "entity_identifiers": entity_identifiers,
                "elem": elem
            }
            examples.append(example)

        json.dump(examples, open(path.replace(".json", "_preprocessed.json"), "w"))



class ReDocREDDataset(Dataset):
    def __init__(self, tokenizer, relation_dict, max_length: int = 512, batch_size: int = 8, num_workers=4, use_gold=True,
                 add_no_relation=False, remove_direct_links=False, use_wikidata5m=False):
        super().__init__(tokenizer, relation_dict, max_length, batch_size, num_workers, add_no_relation,
                         remove_direct_links=remove_direct_links)
        if use_gold:
            self.train_path = "data/docred/train_revised_wiki_qid.json"
        else:
            self.train_path = "data/docred/train_distant_qid.json"
        self.val_path = "data/docred/dev_revised_wiki_qid.json"
        self.test_path = "data/docred/test_revised_wiki_qid.json"

        if use_wikidata5m:
            self.graph_relation_indices = json.load(open("data/wikidata5m_transductive/prop_idx.json"))
            self.graph = gt.load_graph("data/wikidata5m_graph.gt")
        else:
            self.graph_relation_indices = {int(x): v for x, v in
                                           json.load(open("data/docred/prop_dict_all.json")).items()}
            self.graph = gt.load_graph("data/docred/full_graph_all.gt")

    def prepare_dataset(self, path: str):
        data = json.load(open(path))

        examples = []

        for elem in tqdm(data):
            entity_start = set()
            entity_end = set()
            entity_indices = defaultdict(list)
            entity_identifiers = []
            for idx, entity in enumerate(elem["vertexSet"]):
                qids = set()
                for occurrence in entity:
                    entity_start.add((occurrence["sent_id"], occurrence["pos"][0]))
                    entity_end.add((occurrence["sent_id"], occurrence["pos"][1] - 1))
                    entity_indices[idx].append((occurrence["sent_id"], occurrence["pos"][0]))
                    if occurrence["type"] in SPECIAL_ENTITIES:
                        continue
                    if not isinstance(occurrence["qid"], list):
                        if occurrence["qid"] is not None and occurrence["qid"].startswith("Q") and occurrence["qid"][1:].isdigit():
                            qids.add(occurrence["qid"])
                    else:
                        raw_qids = occurrence["qid"]
                        raw_qids = [x for x in raw_qids if x[0].startswith("Q") and x[0][1:].isdigit()]
                        for qid, score in raw_qids[:3]:
                            qids.add(qid)
                entity_identifiers.append(list(qids))

            sents, sent_map, sent_pos = self.add_entity_markers(elem, entity_start, entity_end)

            sent_map = [{k: v + 1 for k, v in sent_map_.items()} for sent_map_ in sent_map]
            input_ids = self.tokenizer.convert_tokens_to_ids(sents)

            entity_indices = {k: list(sorted([[entity[0], sent_map[entity[0]][entity[1]]] for entity in v])) for k, v in entity_indices.items()}

            final_entity_indices = []
            for idx in range(len(entity_indices)):
                final_entity_indices.append(entity_indices[idx])

            relation_labels = []
            for label in elem.get("labels", []):
                head = label["h"]
                tail = label["t"]
                relation_labels.append((head, tail, self.relation_dict[label["r"]]))

            example = {
                "input_ids": input_ids,
                "entity_indices": final_entity_indices,
                "relation_labels": relation_labels,
                "entity_identifiers": entity_identifiers,
                "elem": elem
            }
            examples.append(example)

        json.dump(examples, open(path.replace(".json", "_preprocessed.json"), "w"))

class DWIEDataset(Dataset):
    def __init__(self, tokenizer, relation_dict, max_length: int = 512, batch_size: int = 8, num_workers=4,
                 add_no_relation=False, remove_direct_links=False, filter_relations=False):
        super().__init__(tokenizer, relation_dict, max_length, batch_size, num_workers, add_no_relation, downsample=True,
                         remove_direct_links=remove_direct_links)
        self.train_path = "data/dwie/train_qid.json"
        self.val_path = "data/dwie/dev_qid.json"
        self.test_path = "data/dwie/test_qid.json"
        self.graph = gt.load_graph("data/dwie/full_graph.gt")
        if not os.path.exists("data/dwie/prop_dict.json"):
            self.graph_relation_indices = {}
            for idx, edge in enumerate(self.graph.edges()):
                pid = self.graph.ep["pid"][edge]
                if pid not in self.graph_relation_indices:
                    self.graph_relation_indices[pid] = len(self.graph_relation_indices)
            json.dump(self.graph_relation_indices, open("data/dwie/prop_dict.json", "w"))
        else:
            self.graph_relation_indices = {int(x):v for x, v in json.load(open("data/dwie/prop_dict.json")).items()}
        if filter_relations:
            self.relations_filter_set = set(json.load(open("data/dwie/filtered_relations_one_hop.json")))

    def prepare_dataset(self, path: str):
        data = json.load(open(path))

        examples = []

        for elem in tqdm(data):
            entity_start = set()
            entity_end = set()
            entity_indices = defaultdict(list)
            entity_identifiers = []
            sent = elem["text"]
            if len(elem["mentions"]) == 0:
                continue
            new_sent = [sent[:elem["mentions"][0]["begin"]]]
            for idx, mention in enumerate(elem["mentions"]):

                if "candidates" in mention and mention["candidates"] and  mention["candidates"][0].startswith("Q"):
                    wikipedia_id = mention["candidates"][0]
                else:
                    wikipedia_id = mention["concept"]

                concept_id = mention["concept"]
                begin_index = len(new_sent)
                end_index= len(new_sent)
                new_sent.append(sent[mention["begin"]:mention["end"]])
                if idx == len(elem["mentions"]) - 1:
                    new_sent.append(sent[mention["end"]:])
                else:
                    new_sent.append(sent[mention["end"]:elem["mentions"][idx + 1]["begin"]])
                entity_start.add((0, begin_index))
                entity_end.add((0, end_index))
                entity_indices[concept_id].append([(0, begin_index), wikipedia_id])

            new_entity_indices = defaultdict(list)
            counter = 0
            remap = {}
            for idx in sorted(entity_indices.keys()):
                entity_identifiers_ = []
                entity_indices_ = []
                for (sent_id, begin_index), wikipedia_id in entity_indices[idx]:
                    if not isinstance(wikipedia_id, int):
                        entity_identifiers_.append(wikipedia_id)
                    entity_indices_.append((sent_id, begin_index))
                new_entity_indices[counter] = entity_indices_
                entity_identifiers.append(entity_identifiers_)
                remap[idx] = counter
                counter += 1

            entity_indices = new_entity_indices
            sents, sent_map, sent_pos = self.add_entity_markers({
                "sents": [new_sent]
            }, entity_start, entity_end)

            sent_map = [{k: v + 1 for k, v in sent_map_.items()} for sent_map_ in sent_map]
            input_ids = self.tokenizer.convert_tokens_to_ids(sents)

            entity_indices = {k: list(sorted([[entity[0], sent_map[entity[0]][entity[1]]] for entity in v])) for k, v in entity_indices.items()}

            final_entity_indices = []
            for idx in range(len(entity_indices)):
                final_entity_indices.append(entity_indices[idx])

            relation_labels = []
            found_in_training_labels = []
            for label in elem.get("relations", []):
                head = label["s"]
                tail = label["o"]
                if head in remap and tail in remap:
                    relation = label["p"]
                    relation_labels.append((remap[head], remap[tail], self.relation_dict[relation]))
                    found_in_training_labels.append(label.get("encountered", False))


            if not relation_labels:
                continue
            example = {
                "input_ids": input_ids,
                "entity_indices": final_entity_indices,
                "relation_labels": relation_labels,
                "found_in_training_labels": found_in_training_labels,
                "entity_identifiers": entity_identifiers,
                "elem": None
            }
            examples.append(example)

        json.dump(examples, open(path.replace(".json", "_preprocessed.json"), "w"))

    @classmethod
    def get_relation_dict(cls):
        relation_dict = json.load(open("data/dwie/relation_dict.json"))
        return relation_dict

class BioRELDataset(Dataset):

    @classmethod
    def get_relation_dict(cls):
        train_path = "data/biorel/train.json"
        val_path = "data/biorel/dev.json"
        test_path = "data/biorel/test.json"
        relation_dict = {}
        for item in json.load(open(train_path)):
            if item["relation"] not in relation_dict:
                relation_dict[item["relation"]] = len(relation_dict)

        for item in json.load(open(val_path)):
            if item["relation"] not in relation_dict:
                relation_dict[item["relation"]] = len(relation_dict)

        for item in json.load(open(test_path)):
            if item["relation"] not in relation_dict:
                relation_dict[item["relation"]] = len(relation_dict)

        return relation_dict
    def __init__(self, tokenizer, max_length: int = 512, batch_size: int = 8, num_workers=4,
                 add_no_relation=False, remove_direct_links=False):
        train_path = "data/biorel/train.json"
        val_path = "data/biorel/dev.json"
        test_path = "data/biorel/test.json"

        relation_dict = self.get_relation_dict()


        super().__init__(tokenizer, relation_dict, max_length, batch_size, num_workers, add_no_relation,
                         only_existing_triples=True, num_hops=2, remove_direct_links=remove_direct_links)


        self.graph = gt.load_graph("data/biorel/full_graph_3_hop.gt")
        self.identifier_mapping = json.load(open("data/biorel/identifier_mapping.json"))
        entity_indices = json.load(open("data/biorel/node_index.json"))
        self.entity_indices = defaultdict(set)
        for identifier, values in self.identifier_mapping.items():
            for value in values:
                if value in entity_indices:
                    self.entity_indices[identifier].add(entity_indices[value])
        self.graph_relation_indices = json.load(open("data/biorel/relation_index.json"))
        self.graph_relation_indices = {idx: idx for idx in range(len(self.graph_relation_indices))}

        self.val_path = val_path
        self.train_path = train_path
        self.test_path = test_path

    def prepare_dataset(self, path: str):
        data = json.load(open(path))

        examples = []

        for elem in tqdm(data):
            sentence = elem["sentence"]

            head_entity = elem["head"]
            tail_entity = elem["tail"]
            head_cui = head_entity["CUI"]
            tail_cui = tail_entity["CUI"]

            entity_start = set()
            entity_end = set()
            entity_indices = defaultdict(list)
            entity_identifiers = []
            sent = []
            if head_entity["start"] < tail_entity["start"]:
                sent.append(sentence[:head_entity["start"]])
                sent.append(sentence[head_entity["start"]:head_entity["start"] + head_entity["length"]])
                sent.append(sentence[head_entity["start"] + head_entity["length"]:tail_entity["start"]])
                sent.append(sentence[tail_entity["start"]:tail_entity["start"] + tail_entity["length"]])
                sent.append(sentence[tail_entity["start"] + tail_entity["length"]:])
                entity_start.add((0, 1))
                entity_end.add((0, 1))
                entity_start.add((0, 3))
                entity_end.add((0, 3))
                entity_indices[0].append((0, 1))
                entity_indices[1].append((0, 3))
                entity_identifiers.append(list(self.entity_indices.get(head_cui, [])))
                entity_identifiers.append(list(self.entity_indices.get(tail_cui, [])))
            else:
                sent.append(sentence[:tail_entity["start"]])
                sent.append(sentence[tail_entity["start"]:tail_entity["start"] + tail_entity["length"]])
                sent.append(sentence[tail_entity["start"] + tail_entity["length"]:head_entity["start"]])
                sent.append(sentence[head_entity["start"]:head_entity["start"] + head_entity["length"]])
                sent.append(sentence[head_entity["start"] + head_entity["length"]:])
                entity_start.add((0, 1))
                entity_end.add((0, 1))
                entity_start.add((0, 3))
                entity_end.add((0, 3))
                entity_indices[1].append((0, 1))
                entity_indices[0].append((0, 3))
                entity_identifiers.append(list(self.entity_indices.get(tail_cui, [])))
                entity_identifiers.append(list(self.entity_indices.get(head_cui, [])))



            elem["sents"] = [sent]

            sents, sent_map, sent_pos = self.add_entity_markers(elem, entity_start, entity_end)

            sent_map = [{k: v + 1 for k, v in sent_map_.items()} for sent_map_ in sent_map]
            input_ids = self.tokenizer.convert_tokens_to_ids(sents)

            entity_indices = {k: list(sorted([[entity[0], sent_map[entity[0]][entity[1]]] for entity in v])) for k, v in entity_indices.items()}

            final_entity_indices = []
            for idx in range(len(entity_indices)):
                final_entity_indices.append(entity_indices[idx])

            relation_labels = [
                (0, 1, self.relation_dict[elem["relation"]])
            ]

            example = {
                "input_ids": input_ids,
                "entity_indices": final_entity_indices,
                "relation_labels": relation_labels,
                "entity_identifiers": entity_identifiers
            }
            examples.append(example)

        json.dump(examples, open(path.replace(".json", "_preprocessed.json"), "w"))


# class ADEDataset(Dataset):
#
#     @classmethod
#     def get_relation_dict(cls):
#         train_path = "data/biorel/train.json"
#         val_path = "data/biorel/dev.json"
#         test_path = "data/biorel/test.json"
#         relation_dict = {}
#         for item in json.load(open(train_path)):
#             if item["relation"] not in relation_dict:
#                 relation_dict[item["relation"]] = len(relation_dict)
#
#         for item in json.load(open(val_path)):
#             if item["relation"] not in relation_dict:
#                 relation_dict[item["relation"]] = len(relation_dict)
#
#         for item in json.load(open(test_path)):
#             if item["relation"] not in relation_dict:
#                 relation_dict[item["relation"]] = len(relation_dict)
#
#         return relation_dict
#     def __init__(self, tokenizer, max_length: int = 512, batch_size: int = 8, num_workers=4,
#                  add_no_relation=False):
#
#         dataset_class = load_dataset("ade-benchmark-corpus/ade_corpus_v2", name="Ade_corpus_v2_classification")
#         dataset_ade = load_dataset("ade-benchmark-corpus/ade_corpus_v2", name="Ade_corpus_v2_drug_ade_relation")
#         dataset_dosage = load_dataset("ade-benchmark-corpus/ade_corpus_v2", name="Ade_corpus_v2_drug_dosage_relation")
#         train_path = "data/biorel/train.json"
#         val_path = "data/biorel/dev.json"
#         test_path = "data/biorel/test.json"
#
#         relation_dict = self.get_relation_dict()
#
#
#         super().__init__(tokenizer, relation_dict, max_length, batch_size, num_workers, add_no_relation,
#                          only_existing_triples=True, num_hops=2)
#
#
#         self.graph = gt.load_graph("data/biorel/full_graph_3_hop.gt")
#         self.identifier_mapping = json.load(open("data/biorel/identifier_mapping.json"))
#         entity_indices = json.load(open("data/biorel/node_index.json"))
#         self.entity_indices = defaultdict(set)
#         for identifier, values in self.identifier_mapping.items():
#             for value in values:
#                 if value in entity_indices:
#                     self.entity_indices[identifier].add(entity_indices[value])
#         self.graph_relation_indices = json.load(open("data/biorel/relation_index.json"))
#         self.graph_relation_indices = {idx: idx for idx in range(len(self.graph_relation_indices))}
#
#         self.val_path = val_path
#         self.train_path = train_path
#         self.test_path = test_path
#
#     def prepare_dataset(self, path: str):
#         data = json.load(open(path))
#
#         examples = []
#
#         for elem in tqdm(data):
#             sentence = elem["sentence"]
#
#             head_entity = elem["head"]
#             tail_entity = elem["tail"]
#             head_cui = head_entity["CUI"]
#             tail_cui = tail_entity["CUI"]
#
#             entity_start = set()
#             entity_end = set()
#             entity_indices = defaultdict(list)
#             entity_identifiers = []
#             sent = []
#             if head_entity["start"] < tail_entity["start"]:
#                 sent.append(sentence[:head_entity["start"]])
#                 sent.append(sentence[head_entity["start"]:head_entity["start"] + head_entity["length"]])
#                 sent.append(sentence[head_entity["start"] + head_entity["length"]:tail_entity["start"]])
#                 sent.append(sentence[tail_entity["start"]:tail_entity["start"] + tail_entity["length"]])
#                 sent.append(sentence[tail_entity["start"] + tail_entity["length"]:])
#                 entity_start.add((0, 1))
#                 entity_end.add((0, 1))
#                 entity_start.add((0, 3))
#                 entity_end.add((0, 3))
#                 entity_indices[0].append((0, 1))
#                 entity_indices[1].append((0, 3))
#                 entity_identifiers.append(list(self.entity_indices.get(head_cui, [])))
#                 entity_identifiers.append(list(self.entity_indices.get(tail_cui, [])))
#             else:
#                 sent.append(sentence[:tail_entity["start"]])
#                 sent.append(sentence[tail_entity["start"]:tail_entity["start"] + tail_entity["length"]])
#                 sent.append(sentence[tail_entity["start"] + tail_entity["length"]:head_entity["start"]])
#                 sent.append(sentence[head_entity["start"]:head_entity["start"] + head_entity["length"]])
#                 sent.append(sentence[head_entity["start"] + head_entity["length"]:])
#                 entity_start.add((0, 1))
#                 entity_end.add((0, 1))
#                 entity_start.add((0, 3))
#                 entity_end.add((0, 3))
#                 entity_indices[1].append((0, 1))
#                 entity_indices[0].append((0, 3))
#                 entity_identifiers.append(list(self.entity_indices.get(tail_cui, [])))
#                 entity_identifiers.append(list(self.entity_indices.get(head_cui, [])))
#
#
#
#             elem["sents"] = [sent]
#
#             sents, sent_map, sent_pos = self.add_entity_markers(elem, entity_start, entity_end)
#
#             sent_map = [{k: v + 1 for k, v in sent_map_.items()} for sent_map_ in sent_map]
#             input_ids = self.tokenizer.convert_tokens_to_ids(sents)
#
#             entity_indices = {k: list(sorted([sent_map[entity[0]][entity[1]] for entity in v])) for k, v in entity_indices.items()}
#
#             final_entity_indices = []
#             for idx in range(len(entity_indices)):
#                 final_entity_indices.append(entity_indices[idx])
#
#             relation_labels = [
#                 (0, 1, self.relation_dict[elem["relation"]])
#             ]
#
#             example = {
#                 "input_ids": input_ids,
#                 "entity_indices": final_entity_indices,
#                 "relation_labels": relation_labels,
#                 "entity_identifiers": entity_identifiers
#             }
#             examples.append(example)
#
#         json.dump(examples, open(path.replace(".json", "_preprocessed.json"), "w"))

if __name__ == "__main__":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    relation_dict = DWIEDataset.get_relation_dict()
    dataset = DWIEDataset(tokenizer, relation_dict)
    dataset.setup()
    for batch in dataset.train_dataloader():
        blah = 3
