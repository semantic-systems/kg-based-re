import copy
import json
import pickle
from collections import defaultdict
from typing import Set

import torch
from tqdm import tqdm
import graph_tool.all as gt


candidates_dict = {}
entity_descriptions = {}
def create_examples(item):
    global candidates_dict
    global entity_descriptions
    examples = []
    previous_candidates = []
    for entity in item["entities"]:
        if entity["qid"] is None:
            continue

        candidates = candidates_dict.get(entity["mention"], candidates_dict.get(entity["mention"].lower(), []))
        candidates = [x[0] for x in candidates]

        candidate_qids = []
        candidates_found = set()
        for candidate in candidates[:30]:
            candidate_content = entity_descriptions.get(candidate, None)
            if not candidate_content or candidate is None or candidate in candidates_found:
                continue

            candidate_qids.append(candidate)
            candidates_found.add(candidate)
        examples.append({
            "candidate_qids": candidate_qids,
            "previous_candidates": copy.deepcopy(previous_candidates),
        })
        previous_candidates.append(entity["qid"])

    return examples

def init_worker():
    graph_obj, graph_relation_indices_obj = load_files()
    global graph_worker
    global graph_relation_indices_worker
    graph_worker = graph_obj
    graph_relation_indices_worker = graph_relation_indices_obj
    print(f"Worker initialized with data_objects")


def get_two_hop(qids: Set[int], limit=50, num_hops=2):
    global graph_worker
    global graph_relation_indices_worker
    triples = set()
    other_qids = set()

    for qid in qids:
        if not isinstance(qid, int):
            qid = int(qid[1:])
        if qid < graph_worker.num_vertices():
            out_edges = graph_worker.get_out_edges(qid, [graph_worker.ep["pid"]])
            in_edges = graph_worker.get_in_edges(qid, [graph_worker.ep["pid"]])
            out_edges = out_edges[:limit]
            in_edges = in_edges[:limit]
            for edge in out_edges:
                head, tail, pid = edge
                triples.add((qid, graph_relation_indices_worker[pid], tail))
                triples.add((tail, graph_relation_indices_worker[pid] + len(graph_relation_indices_worker), qid))
                other_qids.add(tail)
            for edge in in_edges:
                head, tail, pid = edge
                triples.add((head, graph_relation_indices_worker[pid], qid))
                triples.add((qid, graph_relation_indices_worker[pid] + len(graph_relation_indices_worker), head))
                other_qids.add(head)

    existing_qids = qids
    for i in range(num_hops - 1):
        other_qids = other_qids.difference(existing_qids)
        new_other_qids = set()
        for other_qid in other_qids:
            out_edges = graph_worker.get_out_edges(other_qid, [graph_worker.ep["pid"]])
            in_edges = graph_worker.get_in_edges(other_qid, [graph_worker.ep["pid"]])
            out_edges = out_edges[:limit]
            in_edges = in_edges[:limit]
            for edge in out_edges:
                head, tail, pid = edge
                triples.add((other_qid, graph_relation_indices_worker[pid], tail))
                triples.add((tail, graph_relation_indices_worker[pid] + len(graph_relation_indices_worker), other_qid))
                new_other_qids.add(tail)
            for edge in in_edges:
                head, tail, pid = edge
                triples.add((head, graph_relation_indices_worker[pid], other_qid))
                triples.add((other_qid, graph_relation_indices_worker[pid] + len(graph_relation_indices_worker), head))
                new_other_qids.add(head)
        existing_qids.update(other_qids)
        other_qids = new_other_qids
    return triples

def prepare_full_examples(examples):
    all_identifiers = set()
    for example in examples:
        all_identifiers.update(example["all_identifiers"])
    all_identifiers = list(all_identifiers)
    pool = torch.multiprocessing.Pool(14, initializer=init_worker)
    final_triples = pool.imap(prepare_full_example, tqdm(all_identifiers, total=len(all_identifiers)))
    all_final_triples = {}
    for identifier, triples in zip(all_identifiers, final_triples):
        all_final_triples[identifier] = triples
    return all_final_triples
def prepare_full_example(identifier):
    triples = get_two_hop({identifier})

    occurs_in_triples = defaultdict(int)
    for s, p, o in triples:
        occurs_in_triples[s] += 1
        occurs_in_triples[o] += 1
    original_nodes = {identifier}

    edge_index = []
    edge_type = []
    final_triples = []
    for s, p, o in triples:
        final_triples.append((s, p, o))
    return final_triples

def load_files():
    graph_relation_indices = json.load(open("data/biorel/relation_index.json"))
    graph_relation_indices = {idx: idx for idx in range(len(graph_relation_indices))}

    graph = gt.load_graph("data/biorel/full_graph_3_hop.gt")
    return graph, graph_relation_indices

def load_dataset(path: str):
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

        elem["entity_index"] = entity_index
        elem["all_identifiers"] = all_identifiers
        examples.append(elem)
    return examples


def main(path: str):

    dataset = load_dataset(path)

    dataset = prepare_full_examples(dataset)
    pickle.dump(dataset, open(path.replace(".json", "_cache.pkl"), "wb"))

if __name__ == "__main__":
    path = "data/biorel/train_preprocessed.json"
    main(path)
    path = "data/biorel/dev_preprocessed.json"
    main(path)
