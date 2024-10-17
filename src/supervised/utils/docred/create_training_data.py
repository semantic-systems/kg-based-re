import copy
import json
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Set, List

import jsonlines
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import graph_tool.all as gt

sparqlwrapper = SPARQLWrapper("http://sems-ai-1:1234/api/endpoint/sparql")

def get_examples_per_relation(relation_dict: dict, num_triples: int = 2000):
    qid_pairs = defaultdict(set)
    for relation in tqdm(relation_dict, total=len(relation_dict)):
        query = f"""
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        SELECT ?s ?o WHERE {{
            ?s wdt:{relation} ?o.
        }}
        LIMIT {num_triples}
        """
        sparqlwrapper.setQuery(query)
        sparqlwrapper.setReturnFormat(JSON)
        results = sparqlwrapper.query().convert()
        for result in results["results"]["bindings"]:
            qid_pairs[relation].add((result["s"]["value"], result["o"]["value"]))

    qid_pairs = {k: [list(x) for x in v] for k, v in qid_pairs.items()}
    json.dump(qid_pairs, open("data/docred/qid_pairs.json", "w"))


def get_one_hop_neighborhood(qid: str, num_neighbors: int = 50):
    triples = set()
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?p ?o WHERE {{
        wd:{qid} ?p ?o.
        FILTER (STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        FILTER (STRSTARTS(STR(?o), "http://www.wikidata.org/entity/"))
    }}
    LIMIT {num_neighbors}
    """
    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    identified_qids = set()
    for result in results["results"]["bindings"]:
        pid = result["p"]["value"]
        pid = pid.split("/")[-1]
        o_id = result["o"]["value"]
        o_id = o_id.split("/")[-1]
        triples.add((qid, pid, o_id))
        identified_qids.add(o_id)
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?p ?s WHERE {{
        ?s ?p wd:{qid}.
        FILTER (STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        FILTER (STRSTARTS(STR(?s), "http://www.wikidata.org/entity/"))
    }}
    LIMIT {num_neighbors}
    """
    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    for result in results["results"]["bindings"]:
        pid = result["p"]["value"]
        pid = pid.split("/")[-1]
        s_id = result["s"]["value"]
        s_id = s_id.split("/")[-1]
        triples.add((s_id, pid, qid))
        identified_qids.add(s_id)

    return triples, identified_qids

def get_one_hop_neighborhood_outgoing(qid: str, num_neighbors: int = 30):
    triples = set()
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?p ?o WHERE {{
        wd:{qid} ?p ?o.
        FILTER (STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        FILTER (STRSTARTS(STR(?o), "http://www.wikidata.org/entity/"))
    }}
    LIMIT {num_neighbors}
    """
    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    identified_qids = set()
    for result in results["results"]["bindings"]:
        pid = result["p"]["value"]
        pid = pid.split("/")[-1]
        o_id = result["o"]["value"]
        o_id = o_id.split("/")[-1]
        triples.add((qid, pid, o_id))
        identified_qids.add(o_id)

    return triples, identified_qids


def get_two_hop_neighborhood(qid: str, neighborhood_cache, pool, num_neighbors: int = 50):
    if qid in neighborhood_cache:
        triples, identified_qids = neighborhood_cache[qid]
        triples = copy.deepcopy(triples)
    else:
        triples, identified_qids = get_one_hop_neighborhood(qid, num_neighbors)
        neighborhood_cache[qid] = (triples, identified_qids)
    considered_qids = set()
    for neighbor in identified_qids:
        if neighbor in neighborhood_cache:
            triples_, identified_qids = neighborhood_cache[neighbor]
            triples.update(copy.deepcopy(triples_))
        else:
            considered_qids.add(neighbor)
    considered_qids = list(considered_qids)
    # for (triples_, identified_qids), neighbor in zip(pool.imap(get_one_hop_neighborhood, considered_qids), considered_qids):
    #     neighborhood_cache[neighbor] = (triples_, identified_qids)
    #     triples.update(triples_)

    for neighbor in considered_qids:
        (triples_, identified_qids) = get_one_hop_neighborhood(neighbor, num_neighbors)
        neighborhood_cache[neighbor] = (triples_, identified_qids)
        triples.update(triples_)

    return triples

def get_triples(dataset):
    all_qids = set()
    all_triples  = set()
    for item in dataset:
        idx_to_qid = defaultdict(set)
        qids = set()
        for idx, vertex in enumerate(item["vertexSet"]):
            for mention in vertex:
                if mention.get("qid", None) is not None:
                    if isinstance(mention["qid"], str):
                        qid = mention["qid"]
                        if qid.startswith("Q"):
                            idx_to_qid[idx].add(qid)
                            qids.add(qid)
                    else:
                        for qid, score in mention["qid"][:3]:
                            if qid.startswith("Q"):
                                idx_to_qid[idx].add(qid)
                                qids.add(qid)
        triples = set()
        for label in item["labels"]:
            relation = label["r"]
            for head in idx_to_qid[label["h"]]:
                for tail in idx_to_qid[label["t"]]:
                    triples.add((head, relation, tail))
        all_triples.update(triples)
    hard_negatives = []
    for item in dataset:
        idx_to_qid = defaultdict(set)
        qids = set()
        for idx, vertex in enumerate(item["vertexSet"]):
            for mention in vertex:
                if mention.get("qid", None) is not None:
                    if isinstance(mention["qid"], str):
                        qid = mention["qid"]
                        if qid.startswith("Q"):
                            idx_to_qid[idx].add(qid)
                            qids.add(qid)
                    else:
                        for qid, score in mention["qid"][:3]:
                            if qid.startswith("Q"):
                                idx_to_qid[idx].add(qid)
                                qids.add(qid)
        triples = set()
        for label in item["labels"]:
            relation = label["r"]
            for head in idx_to_qid[label["h"]]:
                for tail in idx_to_qid[label["t"]]:
                    triples.add((head, relation, tail))

        for triple in triples:
            head, relation, tail = triple
            current_qids = qids.difference({head, tail})
            hard_negative_qids = set()
            for qid in current_qids:
                if (head, relation, qid) not in all_triples:
                    hard_negative_qids.add(qid)
            hard_negatives.append((triple, hard_negative_qids))
        all_qids.update(qids)

    return all_triples, all_qids, hard_negatives


def get_examples_from_dataset(datasets: List[str]):
    all_qids = set()

    for dataset in datasets:
        _, qids = get_triples(json.load(open(dataset)))
        all_qids.update(qids)

    return all_qids


def get_n_hop_paths(qid, cache, n_hops: int = 4, limit: int = 20):
    paths = set()
    current_qids = [([qid], {qid})]
    for i in range(n_hops):
        new_current_qids = []
        for current_path, contained_qids in current_qids:
            if current_path[-1] in cache:
                triples, identified_qids = cache[current_path[-1]]
            else:
                triples, identified_qids = get_one_hop_neighborhood_outgoing(current_path[-1], limit)
                cache[current_path[-1]] = (triples, identified_qids)

            for triple in triples:
                if triple[2] in contained_qids:
                    continue
                new_contained_qids = copy.deepcopy(contained_qids)
                new_contained_qids.add(triple[2])
                new_path = copy.deepcopy(current_path) + list(triple)[1:]
                paths.add(tuple(new_path))
                new_current_qids.append((new_path, new_contained_qids))
        random.shuffle(new_current_qids)
        current_qids = new_current_qids[:limit]
    return paths

def create_negatives(triples: set, num_negatives=20):
    qids = set()
    for head, relation, tail in triples:
        qids.add(head)

    cache = {}
    paths_per_qid = {}
    if not os.path.exists("data/docred/paths_per_qid_filtered.json"):
        if not os.path.exists("data/docred/paths_per_qid.json"):
            for qid in tqdm(qids):
                paths = get_n_hop_paths(qid, cache)
                paths_per_qid[qid] = [list(x) for x in paths]
            json.dump(paths_per_qid, open("data/docred/paths_per_qid.json", "w"))
        paths_per_qid = json.load(open("data/docred/paths_per_qid.json", "r"))

        new_paths = {}
        for qid, paths in tqdm(paths_per_qid.items(), total=len(paths_per_qid)):
            if not paths:
                continue
            valid_paths_per_qid_ending = defaultdict(list)
            for path in paths:
                valid_paths_per_qid_ending[path[-1]].append(path)
            valid_paths = []
            for paths_ in valid_paths_per_qid_ending.values():
                sorted_paths = sorted(paths_, key=lambda x: len(x))
                valid_paths.append(sorted_paths[0])
            num_paths_per_length = defaultdict(list)
            for path in valid_paths:
                num_paths_per_length[len(path)].append(path)
            # Sample according to the probabilities
            probabilities = [1 / (len(num_paths_per_length) * len(num_paths_per_length[len(path)])) for path in valid_paths]
            sampled_paths = np.random.choice(range(len(valid_paths)), min(num_negatives + 1, len(valid_paths)), p=probabilities, replace=False)
            sampled_paths = [valid_paths[x] for x in sampled_paths]
            new_paths[qid] = sampled_paths
        json.dump(new_paths, open("data/docred/paths_per_qid_filtered.json", "w"))
    else:
        new_paths = json.load(open("data/docred/paths_per_qid_filtered.json", "r"))

    examples = []
    all_qids = set()
    for triple in triples:
        head, relation, tail = triple
        if head in new_paths:
            negative_qids = set()
            for path in new_paths[head]:
                negative_qids.add(path[-1])
            if  tail in negative_qids:
                negative_qids.remove(tail)
            examples.append(((head, relation, tail), negative_qids))
            all_qids.update(negative_qids)
            all_qids.add(head)
            all_qids.add(tail)

    return examples, all_qids


def construct_training_graphs(train_examples, dev_examples, other_examples, neighborhoods: dict):
    blah=  3
    pass

def to_int(x):
    return int(x[1:])


def to_graph(filename: str, suffix: str):
    all_relations = set()
    number_of_lines = sum(1 for line in open(filename))

    full_graph = gt.Graph(directed=True)
    e_prop = full_graph.new_edge_property("int")

    all_edges = set()
    for item in tqdm(jsonlines.open(filename), total=number_of_lines):
        if not item["qid"].startswith("Q") or not item["qid"][1:].isdigit():
            continue

        qid = to_int(item["qid"])
        triples = [(to_int(triple[0]), to_int(triple[1]), to_int(triple[2])) for triple in item["triples"] if
                   triple[0].startswith("Q") and triple[2].startswith("Q")]
        all_edges.update(triples)

        all_relations.update([x[1] for x in triples])
        # neighborhoods[qid] = triples
    for s, p, o in tqdm(all_edges):
        e = full_graph.add_edge(s, o)
        e_prop[e] = p
    full_graph.edge_properties["pid"] = e_prop
    # Dump graph
    full_graph.save(f"data/docred/full_graph{suffix}.gt")
def sample_neighborhoods(datasets: List[str], suffix: str=""):
    filename = f"data/docred/neighborhoods{suffix}.jsonl"
    pool = Pool(10)
    all_qids = get_examples_from_dataset(datasets)

    all_qids = [x for x in all_qids if x.startswith("Q")]
    neighborhood_cache = {}
    neighborhoods = {}
    output_file = jsonlines.open(filename, "w")
    for qid in tqdm(all_qids):
        triples = get_two_hop_neighborhood(qid, neighborhood_cache, pool)
        output_file.write({"qid": qid, "triples": [list(x) for x in triples]})
    pool.close()
    to_graph(filename, suffix)












def load_neighborhood():
    incoming = defaultdict(set)
    outgoing = defaultdict(set)
    for item in tqdm(jsonlines.open("../parsed_current_wikidata_dump.jsonl"), total=96046766):
        head_qid = item["id"]
        head_qid = int(head_qid[1:])
        for pid, tail_qid in item["claims"]:
            tail_qid = int(tail_qid[1:])
            pid = int(pid[1:])
            incoming[tail_qid].add((head_qid, pid))
            outgoing[head_qid].add((pid, tail_qid))

    return incoming, outgoing


def get_two_hop(qids: Set[int], full_graph, limit = 30):
    triples = set()
    other_qids = set()
    pid = 0

    for qid in qids:
        out_edges = full_graph.get_out_edges(qid, [full_graph.ep["pid"]])
        in_edges = full_graph.get_in_edges(qid,  [full_graph.ep["pid"]])
        for edge in out_edges[:limit]:
            head, tail, pid = edge
            triples.add((qid, pid, tail))
            other_qids.add(tail)
        for edge in in_edges[:limit]:
            head, tail, pid = edge
            triples.add((head, pid, qid))
            other_qids.add(head)
    other_qids = other_qids.difference(qids)
    for other_qid in other_qids:
        out_edges = full_graph.get_out_edges(other_qid,   [full_graph.ep["pid"]])
        in_edges = full_graph.get_in_edges(other_qid,  [full_graph.ep["pid"]])
        for edge in out_edges[:limit]:
            head, tail, pid = edge
            triples.add((other_qid, pid, tail))
        for edge in in_edges[:limit]:
            head, tail, pid = edge
            triples.add((head, pid, other_qid))
    return triples


def dump_examples(qid_pairs):
    train_examples, dev_examples, all_qids, all_triples = get_examples_from_dataset()
    additional_triples = set()
    new_all_qids = set()
    for relation, pairs in qid_pairs.items():
        for pair in pairs:
            head = pair[0].split("/")[-1]
            tail = pair[1].split("/")[-1]
            if (head, relation, tail) not in all_triples:
                additional_triples.add((head, relation, tail))
                new_all_qids.add(head)
                new_all_qids.add(tail)
    other_examples, all_qids_ = create_negatives(additional_triples)

    pickle.dump(train_examples, open("data/docred/train_examples.pkl", "wb"))
    pickle.dump(dev_examples, open("data/docred/dev_examples.pkl", "wb"))
    pickle.dump(other_examples, open("data/docred/kg_examples.pkl", "wb"))



def test():
    full_graph = gt.load_graph("data/docred/full_graph.gt")
    train_examples, dev_examples, all_qids, all_triples = get_examples_from_dataset()
    num_nodes = full_graph.num_vertices()
    random.shuffle(train_examples)
    for example in tqdm(train_examples):
        head_qid, relation, tail_qid = example[0]
        all_qids = {head_qid, tail_qid}
        all_qids.update(example[1])
        valid_qids = set()
        for qid in all_qids:
            if qid.startswith("Q") and qid[1:].isdigit() :
                qid = int(qid[1:])
                if qid < num_nodes:
                    valid_qids.add(qid)

        triples = get_two_hop(valid_qids, full_graph)
        blah = 3

    blah = 3
def main():
    sample_neighborhoods(["data/docred/dev_revised_wiki_qid.json", "data/docred/train_revised_wiki_qid.json",
                          "data/docred/test_revised_wiki_qid.json",
                          ], "_all")




if __name__ == "__main__":
    main()