import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import jsonlines
import torch
from torch.utils.data import DataLoader, Dataset
import graph_tool.all as gt
import torch_geometric as tg
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, dataset, edge_mapping, two_hop_neighborhoods_data, remove_direct_link=False,
                 add_inverse_relations=False, empty_graph=False):
        self.dataset = dataset
        self.edge_mapping = edge_mapping
        self.two_hop_neighborhoods_data = two_hop_neighborhoods_data
        self.remove_direct_link = remove_direct_link
        self.add_inverse_relations = add_inverse_relations
        self.empty_graph = empty_graph

    def create_graph(self, head_qids, tail_qids, edge_mapping):
        occurrence_counter = defaultdict(int)
        head_neighborhoods = []
        node_mapping = {}
        for head_qid in head_qids:
            head_neighborhood = self.two_hop_neighborhoods_data.get(head_qid, [])
            node_mapping[head_qid] = 0
            for s, p, o in head_neighborhood:
                occurrence_counter[s] += 1
                occurrence_counter[o] += 1
            if head_neighborhood:
                head_neighborhoods.append(head_neighborhood)


        tail_neighborhoods = []
        for tail_qid in tail_qids:
            tail_neighborhood = self.two_hop_neighborhoods_data.get(tail_qid, [])
            if tail_qid not in node_mapping:
                node_mapping[tail_qid] = len(node_mapping)
            for s, p, o in tail_neighborhood:
                occurrence_counter[s] += 1
                occurrence_counter[o] += 1
            if tail_neighborhood:
                tail_neighborhoods.append(tail_neighborhood)


        edge_index = []
        edge_type = []
        if self.empty_graph:
            head_neighborhoods = []
            tail_neighborhoods = []
        for head_neighborhood in head_neighborhoods:
            for s, p, o in head_neighborhood:
                if self.remove_direct_link and s == head_qid and o == tail_qid:
                    continue
                if occurrence_counter[s] < 2 or occurrence_counter[o] < 2:
                    continue
                if p not in edge_mapping:
                    continue
                if s not in node_mapping:
                    node_mapping[s] = len(node_mapping)
                if o not in node_mapping:
                    node_mapping[o] = len(node_mapping)
                edge_index.append((node_mapping[s], node_mapping[o]))
                edge_type.append(edge_mapping[p])
                if self.add_inverse_relations:
                    edge_index.append((node_mapping[o], node_mapping[s]))
                    edge_type.append(edge_mapping[p] + len(edge_mapping))


        for tail_neighborhood in tail_neighborhoods:
            for s, p, o in tail_neighborhood:
                if self.remove_direct_link and s == head_qid and o == tail_qid:
                    continue
                if occurrence_counter[s] < 2 or occurrence_counter[o] < 2:
                    continue
                if p not in edge_mapping:
                    continue
                if s not in node_mapping:
                    node_mapping[s] = len(node_mapping)
                if o not in node_mapping:
                    node_mapping[o] = len(node_mapping)
                edge_index.append((node_mapping[s], node_mapping[o]))
                edge_type.append(edge_mapping[p])
                if self.add_inverse_relations:
                    edge_index.append((node_mapping[o], node_mapping[s]))
                    edge_type.append(edge_mapping[p] + len(edge_mapping))

        if not edge_index:
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            edge_type = torch.zeros(1, dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_type, dtype=torch.long)
        graph = tg.data.Data(edge_index=edge_index, edge_type=edge_type, num_nodes=len(node_mapping))
        return graph, node_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        head_id, tail_id = item[4]
        if isinstance(head_id, str):
            head_ids = [head_id]
        else:
            head_ids = head_id
        if isinstance(tail_id, str):
            tail_ids = [tail_id]
        else:
            tail_ids = tail_id
        new_head_ids = []
        for head_id in head_ids:
            if not head_id.startswith("Q") and not head_id[1:].isdigit():
                head_id = "none"
            else:
                head_id = int(head_id[1:])
            new_head_ids.append(head_id)
        head_ids = new_head_ids
        new_tail_ids = []
        for tail_id in tail_ids:
            if not tail_id.startswith("Q") and not tail_id[1:].isdigit():
                tail_id = "none"
            else:
                tail_id = int(tail_id[1:])
            new_tail_ids.append(tail_id)
        tail_ids = new_tail_ids
        relations = item[2]
        relation_nodes = []
        queries = []

        graph, node_mapping = self.create_graph(head_ids, tail_ids, self.edge_mapping)

        for _, relation in relations:
            relation = int(relation[1:])
            relation_nodes.append(self.edge_mapping[relation])
            queries.append([node_mapping[head_ids[0]], node_mapping[tail_ids[0]], self.edge_mapping[relation]])

        return item[0], item[1], item[2], item[3], graph, queries, relation_nodes


def custom_collate_fn(batch, tokenizer, relation_graph):
    labels = []
    texts = []
    graphs = []
    elements_per_item = []
    batch_relations = []
    all_indices = []
    counter = 0
    node_offset = 0
    queries = []
    relation_nodes = []
    for idx, item in enumerate(batch):
        labels = labels + item[1]
        all_indices.append(item[3])
        for elem in item[0]:
            texts.append(elem[0][0])
            counter += 1
        graph = item[4]
        graphs.append(graph)
        for x in item[5]:
            x[0] += node_offset
            x[1] += node_offset
            queries.append(x)
        node_offset += graph.num_nodes
        batch_relations.append(item[2])
        elements_per_item.append(len(item[0]))
    graphs = tg.data.Batch.from_data_list(graphs)
    graphs.num_relations = relation_graph.num_nodes
    graphs.relation_graph = relation_graph
    queries = torch.tensor(queries, dtype=torch.long)
    relation_nodes = torch.tensor(relation_nodes, dtype=torch.long)
    maximum_elements = max(elements_per_item)
    assert all([x == maximum_elements for x in elements_per_item])
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    labels = torch.tensor(labels, dtype=torch.float)

    return encoded, labels, maximum_elements, batch_relations, graphs, queries, relation_nodes

class RelationExtractionDataset(pl.LightningDataModule):
    def __init__(self, dataset_name: str, tokenizer, args: dict = None):
        if args is None:
            args = {}
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        main_dataset = dataset_name.split("/")[0]

        self.type_labels = json.load(open("types_to_label.json"))
        self.other_properties = args.get("other_properties", 5)
        self.hard_other_properties = args.get("hard_other_properties", 5)
        self.tokenizer = tokenizer
        self.num_workers = args.get("num_workers", 2)
        self.include_descriptions = args.get("include_descriptions", False)
        self.include_types = args.get("include_types", False)
        self.use_predicted_candidates = args.get("use_predicted_candidates", False)
        self.use_all_predicted_candidates = args.get("use_all_predicted_candidates", False)
        assert (self.use_all_predicted_candidates and self.use_predicted_candidates) or not self.use_all_predicted_candidates
        if self.use_all_predicted_candidates:
            assert not self.include_types
        self.remove_direct_link = args.get("remove_direct_link", False)
        self.add_inverse_relations = args.get("add_inverse_relations", False)
        self.use_filtered_meta_graph = args.get("use_filtered_meta_graph", False)
        self.empty_graph = args.get("empty_graph", False)


        self.relation_meta_graphs = self.initialize_meta_graphs(main_dataset)
        self.two_hop_neighborhoods_data = self.load_neighborhoods(f"{main_dataset}/two_hop_neighborhoods_data.json")

        self.rel_to_description = {item["wikidata_id"]:item["en_description"] for item in jsonlines.open("rel_id_title_description_cleaned.jsonl")}

        self.type_descriptions = json.load(open("types_to_description_alt.json"))

        if Path("entity_types_reparsed.pickle").exists():
            self.types_dictionary = pickle.load(open("entity_types_reparsed.pickle", "rb"))
        else:
            self.types_dictionary = pickle.load(open("entity_types.pickle", "rb"))
            types_index = pickle.load(open("entity_types_index.pickle", "rb"))
            inverse_types_index = {v: k for k, v in types_index.items()}
            self.types_dictionary = {key: [inverse_types_index[x] for x in value] for key, value in self.types_dictionary.items()}
            pickle.dump(self.types_dictionary, open("entity_types_reparsed.pickle", "wb"))
    @staticmethod
    def load_neighborhoods(path):
        pickle_path = f"{path.replace('.json', '.pickle')}"
        if Path(pickle_path).exists():
            return pickle.load(open(pickle_path, "rb"))
        else:
            data = json.load(open(path))
            new_data = {}
            for key, value in data.items():
                key = int(key[1:])
                triples = []
                for item in value:
                    s, p, o = item
                    if not (s.startswith("Q") and o.startswith("Q") and p.startswith("P") and s[1:].isdigit() and o[1:].isdigit() and p[1:].isdigit()):
                        continue
                    s = int(s[1:])
                    o = int(o[1:])
                    p = int(p[1:])
                    triples.append((s, p, o))
                new_data[key] = triples
            pickle.dump(new_data, open(pickle_path, "wb"))
            return data

    def initialize_meta_graphs(self, dataset_name):
        path = f"{dataset_name}/meta_graphs"
        suffix = "_inv" if self.add_inverse_relations else ""
        if self.use_filtered_meta_graph:
            tmp_path = f"{dataset_name}/meta_graphs/meta_graphs_new{suffix}.pickle"
        else:
            tmp_path = f"{dataset_name}/meta_graphs/meta_graphs{suffix}.pickle"

        if Path(tmp_path).exists():
            return pickle.load(open(tmp_path, "rb"))
        else:
            edge_mapping = {}
            edge_index = []
            edge_types = []
            edges_found = set()
            if Path(path).exists():
                for item in tqdm(Path(path).iterdir(), total=len(list(Path(path).iterdir()))):
                    if item.is_file() and item.suffix == ".gt":
                        new_in_file_path =  "new" in item.stem
                        if self.use_filtered_meta_graph and not new_in_file_path:
                            continue
                        elif not self.use_filtered_meta_graph and new_in_file_path:
                            continue
                        graph = gt.load_graph(str(item))

                        if self.use_filtered_meta_graph:
                            pid = item.stem[:-4]
                        else:
                            pid = item.stem
                        if int(pid[1:]) not in edge_mapping:
                            edge_mapping[int(pid[1:])] = len(edge_mapping)
                        for edge in graph.edges():
                            s, o = edge
                            s = int(s)
                            o = int(o)
                            edge_type = graph.ep["edge_type"][edge]
                            if (s, o, edge_type) in edges_found:
                                continue
                            edges_found.add((s, o, edge_type))
                            if s not in edge_mapping:
                                edge_mapping[s] = len(edge_mapping)
                            if o not in edge_mapping:
                                edge_mapping[o] = len(edge_mapping)
                            edge_index.append((edge_mapping[s], edge_mapping[o]))
                            edge_types.append(graph.ep["edge_type"][edge])
            for pid, node in edge_mapping.items():
                if not (pid, pid, 0) in edges_found:
                    edge_index.append((node, node))
                    edge_types.append(0)
                if not (pid, pid, 3) in edges_found:
                    edge_index.append((node, node))
                    edge_types.append(3)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            orig_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_index = torch.clone(orig_edge_index)
            num_nodes = len(edge_mapping)

            if self.add_inverse_relations:

                # Add inverse_nodes
                new_edge_index = edge_index + num_nodes
                edge_type_map = {
                    0: 3,
                    1: 2,
                    2: 1,
                    3: 0
                }
                new_edge_type = torch.tensor([edge_type_map[int(x)] for x in edge_type])
                edge_index = torch.cat([edge_index, new_edge_index], dim=1)
                edge_type = torch.cat([edge_type, new_edge_type], dim=0)

                new_edge_index = torch.clone(orig_edge_index)
                new_edge_index[0, :] += num_nodes
                edge_type_map = {
                    0: 2,
                    1: 3,
                    3: 1,
                    2: 0
                }
                new_edge_type = torch.tensor([edge_type_map[int(x)] for x in edge_type])
                edge_index = torch.cat([edge_index, new_edge_index], dim=1)
                edge_type = torch.cat([edge_type, new_edge_type], dim=0)

                new_edge_index = torch.clone(orig_edge_index)
                new_edge_index[1, :] += num_nodes
                edge_type_map = {
                    0: 1,
                    1: 0,
                    2: 3,
                    3: 2
                }
                new_edge_type = torch.tensor([edge_type_map[int(x)] for x in edge_type])
                edge_index = torch.cat([edge_index, new_edge_index], dim=1)
                edge_type = torch.cat([edge_type, new_edge_type], dim=0)

            final_graph = tg.data.Data(edge_index=edge_index, edge_type=edge_type, num_nodes=2 * len(edge_mapping)), edge_mapping
            pickle.dump(final_graph, open(tmp_path, "wb"))
            graphs = final_graph
        return graphs


    def prepare_data(self):
        pass
    def create_representation_with_types(self, text, head_qid, head_text, tail_qid, tail_text, property):
        head_types = self.types_dictionary.get(head_qid, [])
        tail_types = self.types_dictionary.get(tail_qid, [])
        head_type_descriptions = " and ".join(
            [self.type_labels[x] for x in head_types if x in self.type_labels])
        tail_type_descriptions = " and ".join(
            [self.type_labels[x] for x in tail_types if x in self.type_labels])
        if not head_type_descriptions:
            head_type_descriptions = "Any"
        if not tail_type_descriptions:
            tail_type_descriptions = "Any"
        property_description = self.rel_to_description[property[1]]

        if self.include_types:
            output_text = f"[CLS] Given the Head Entity : {head_text} with Types : {head_type_descriptions}, Tail Entity : {tail_text} with Types : {tail_type_descriptions} and Context : {text}, the context expresses the relation [SEP] {property[0]}"
        else:
            output_text = f"[CLS] Given the Head Entity : {head_text}, Tail Entity : {tail_text} and Context : {text}, the context expresses the relation [SEP] {property[0]}"
        if self.include_descriptions:
            output_text += f" defined as {property_description} [SEP]"
        else:
            output_text += f" [SEP]"


        head_text_start = output_text.find(f"Head Entity : {head_text}") + len(f"Head Entity : ")

        head_text_span = (head_text_start, head_text_start + len(head_text))
        tail_text_start = output_text.find(f"Tail Entity : {tail_text}") + len(f"Tail Entity : ")
        tail_text_span = (tail_text_start, tail_text_start + len(tail_text))
        assert output_text[head_text_span[0]:head_text_span[1]] == head_text
        assert output_text[tail_text_span[0]:tail_text_span[1]] == tail_text

        return output_text, head_text_span, tail_text_span

    def create_representations(self, head_text, tail_text, text, elem, relation):
        head_id = elem["head_id"]
        tail_id = elem["tail_id"]
        head_ids = []
        tail_ids = []
        if self.use_predicted_candidates:
            if self.use_all_predicted_candidates:
                head_ids = elem["head_predictions"]
                tail_ids = elem["tail_predictions"]
            elif "head_prediction" in elem and "tail_prediction" in elem:
                head_id = elem["head_prediction"]
                tail_id = elem["tail_prediction"]
        if head_ids:
            head_id = head_ids
            elem["head_id"] = head_id
        if tail_ids:
            tail_id = tail_ids
            elem["tail_id"] = tail_id
        text_representation = self.create_representation_with_types(text, head_id, head_text, tail_id,
                                                                    tail_text, relation)
        #type_tuple = self.create_type_representation(elem["head_id"], elem["tail_id"])
        type_tuple = None
        return text_representation, type_tuple
    def create_graph(self, head_qid, tail_qid, pid):
        meta_graph = self.relation_meta_graphs[pid]
        head_neighborhood = self.two_hop_neighborhoods_data.get(head_qid, [])
        tail_neighborhood = self.two_hop_neighborhoods_data.get(tail_qid, [])
        if not head_neighborhood or not tail_neighborhood:
            return None, None
        edge_index = []
        edge_type = []
        node_mapping = {}
        for s, p, o in head_neighborhood:
            p = int(p[1:])
            if p not in meta_graph.edge_mapping:
                continue
            if s not in node_mapping:
                node_mapping[s] = len(node_mapping)
            if o not in node_mapping:
                node_mapping[o] = len(node_mapping)
            edge_index.append((node_mapping[s], node_mapping[o]))
            edge_type.append(meta_graph.edge_mapping[p])

        for s, p, o in tail_neighborhood:
            p = int(p[1:])
            if p not in meta_graph.edge_mapping:
                continue
            if s not in node_mapping:
                node_mapping[s] = len(node_mapping)
            if o not in node_mapping:
                node_mapping[o] = len(node_mapping)
            edge_index.append((node_mapping[s], node_mapping[o]))
            edge_type.append(meta_graph.edge_mapping[p])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        graph = tg.data.Data(edge_index=edge_index, edge_type=edge_type, node_mapping=node_mapping)
        return graph, meta_graph

    def convert_dataset(self, dataset, all_relations, training=False, include_hard=True):
        label_dict = {x[1]:x[0] for x in all_relations}
        final_dataset = []
        for idx, item in enumerate(dataset):
            for idx_, elem in enumerate(item["triplets"]):
                text = " ".join(elem["tokens"])
                if elem["head"] and elem["tail"]:
                    head_text = " ".join(elem["tokens"][elem["head"][0]:elem["head"][-1] + 1])
                    tail_text = " ".join(elem["tokens"][elem["tail"][0]:elem["tail"][-1] + 1])
                    property = (elem["label"], elem["label_id"])
                    allowed_properties = [x for x in all_relations if x != property]
                    if training:
                        random_other_properties = []
                        if include_hard:
                            hard_relations = [x for x in self.similarity_to_other_properties[property[1]][:20]]
                            distribution = [x[1] + 0.0001 for x in hard_relations]
                            normalizer = sum(distribution)
                            distribution = [x / normalizer for x in distribution]
                            sampled_indices = np.random.choice(np.arange(0, len(distribution)), size=self.hard_other_properties,p=distribution, replace=False)
                            hard_relations = [(label_dict[relation], relation) for relation, _ in hard_relations]
                            random_other_properties += [hard_relations[x] for x in sampled_indices]
                            allowed_properties = [x for x in allowed_properties if x not in random_other_properties]
                        random_other_properties += random.sample(allowed_properties, self.other_properties)
                        all_representations = [self.create_representations(head_text, tail_text, text, elem, property)] + \
                                              [self.create_representations(head_text, tail_text, text, elem, relation) for relation in random_other_properties]
                        labels = [1] + [0] * len(random_other_properties)
                        relations = [property] + random_other_properties
                    else:
                        all_representations = []
                        labels = []
                        relations = []
                        for relation in all_relations:
                            all_representations.append(self.create_representations(head_text, tail_text, text, elem, relation))
                            if relation == property:
                                labels.append(1)
                            else:
                                labels.append(0)
                        relations += all_relations

                    final_dataset.append((all_representations, labels, relations, (idx, idx_), (elem["head_id"], elem["tail_id"])))
        return CustomDataset(final_dataset, self.relation_meta_graphs[1], self.two_hop_neighborhoods_data, self.remove_direct_link,
                             self.add_inverse_relations, empty_graph=self.empty_graph)

    def structure_dataset_split(self, dataset: list):
        all_relations = set()
        for item in dataset:
            for triplet in item["triplets"]:
                all_relations.add((triplet["label"], triplet["label_id"]))
        all_relations = sorted(list(all_relations))
        return dataset, all_relations
    def load_dataset(self, folder_name: str):
        train = []
        val = []
        test = []
        for item in jsonlines.open(f"{folder_name}/train_mapped.jsonl"):
            train.append(item)
        for item in jsonlines.open(f"{folder_name}/dev_mapped.jsonl"):
            val.append(item)
        test_dataset_name = "test_mapped.jsonl"
        if self.use_predicted_candidates:
            test_dataset_name = "test_mapped_predicted.jsonl"
        for item in jsonlines.open(f"{folder_name}/{test_dataset_name}"):
            test.append(item)
        return self.structure_dataset_split(train), self.structure_dataset_split(val), self.structure_dataset_split(test)

    def setup(self, stage=None):
        (train, train_relations), (val, val_relations), (test, test_relations) = self.load_dataset(self.dataset_name)
        self.similarity_to_other_properties = {}
        self._train_dataset = train
        self.train_relations = train_relations
        self._val_dataset = val
        self.val_relations = val_relations
        self._test_dataset = test
        self.test_relations = test_relations
        self.train_dataset = self.convert_dataset(train, train_relations, training=True)
        self.val_dataset = self.convert_dataset(val, val_relations)
        self.test_dataset = self.convert_dataset(test, test_relations)
    def train_dataloader(self):
        train_dataset = self.train_dataset
        return DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True,
                          collate_fn=lambda batch: custom_collate_fn(batch, self.tokenizer, self.relation_meta_graphs[0]),
                          num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args["batch_size"], shuffle=False,
                          collate_fn=lambda batch: custom_collate_fn(batch, self.tokenizer, self.relation_meta_graphs[0]),
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args["batch_size"], shuffle=False,
                          collate_fn=lambda batch: custom_collate_fn(batch, self.tokenizer, self.relation_meta_graphs[0]),
                          num_workers=self.num_workers)


if __name__ == "__main__":
    dataset = RelationExtractionDataset("wiki/unseen_5_seed_0", None, {"batch_size": 2}     )
    dataset.setup()
    for item in dataset.train_dataloader():
        print(item)
        break
