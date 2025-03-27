import copy
import json
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
from typing import Set

import jsonlines
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import graph_tool.all as gt

sparqlwrapper = SPARQLWrapper("anonymized")

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


def get_one_hop_neighborhood(qid: str, num_neighbors: int = 30):
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
    triples = set()
    for item in dataset:
        idx_to_qid = defaultdict(set)
        qids = set()
        for idx, vertex in enumerate(item["vertexSet"]):
            for mention in vertex:
                for qid, score in mention["qid"]:
                    if score > -1.0:
                        all_qids.add(qid)
                        if qid.startswith("Q"):
                            idx_to_qid[idx].add(qid)
                            qids.add(qid)
        for label in item["labels"]:
            relation = label["r"]
            for head in idx_to_qid[label["h"]]:
                for tail in idx_to_qid[label["t"]]:
                    triples.add((head, relation, tail))
    return triples, all_qids


def get_examples_from_dataset():
    all_qids = set()
    triples_dev = set()
    dev_examples = []
    dev_dataset = json.load(open("data/docred/dev_revised_wiki_qid.json"))
    train_dataset = json.load(open("data/docred/train_revised_wiki_qid.json"))


    dev_triples, dev_qids = get_triples(dev_dataset)
    train_triples, train_qids = get_triples(train_dataset)

    all_qids.update(dev_qids)
    all_qids.update(train_qids)

    for item in dev_dataset:
        idx_to_qid = defaultdict(set)
        qids = set()
        for idx, vertex in enumerate(item["vertexSet"]):
            for mention in vertex:
                for qid, score in mention["qid"]:
                    if score > -1.0:
                        all_qids.add(qid)
                        if qid.startswith("Q"):
                            idx_to_qid[idx].add(qid)
                            qids.add(qid)
        for label in item["labels"]:
            relation = label["r"]
            for head in idx_to_qid[label["h"]]:
                for tail in idx_to_qid[label["t"]]:
                    triples_dev.add((head, relation, tail))
                    other_qids = set()
                    for qid in qids:
                        if (head, relation, qid) not in dev_triples:
                            other_qids.add(qid)
                    dev_examples.append(((head, relation, tail), other_qids))

    train_triples = train_triples.difference(dev_triples)

    train_examples = []
    for item in train_dataset:
        idx_to_qid = defaultdict(set)
        qids = set()
        for idx, vertex in enumerate(item["vertexSet"]):
            for mention in vertex:
                for qid, score in mention["qid"]:
                    if score > -1.0:
                        all_qids.add(qid)
                        if qid.startswith("Q"):
                            idx_to_qid[idx].add(qid)
                            qids.add(qid)

        for label in item["labels"]:
            relation = label["r"]
            for head in idx_to_qid[label["h"]]:
                for tail in idx_to_qid[label["t"]]:
                    triple = (head, relation, tail)
                    if triple in train_triples:
                        other_qids = set()
                        for qid in qids:
                            if (head, relation, qid) not in train_triples:
                                other_qids.add(qid)
                        train_examples.append(((head, relation, tail), other_qids))
    all_triples = train_triples.union(dev_triples)
    return train_examples, dev_examples, all_qids, all_triples


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


def sample_neighborhoods(qids: set):
    pool = Pool(10)
    neighborhood_cache = {}
    neighborhoods = {}
    if not os.path.exists("data/docred/neighborhoods_newt.jsonl"):
        output_file = jsonlines.open("data/dwie/neighborhoods_newt.jsonl", "w")
        for qid in tqdm(qids):
            triples = get_two_hop_neighborhood(qid, neighborhood_cache, pool)
            output_file.write({"qid": qid, "triples": [list(x) for x in triples]})
    else:
        output_file = jsonlines.open("data/docred/neighborhoods_newt.jsonl", "r")
        new_output_file = jsonlines.open("data/dwie/neighborhoods_newt.jsonl", "w")
        number_of_lines = sum(1 for line in open("data/docred/neighborhoods_newt.jsonl"))
        for item in tqdm(output_file, total=number_of_lines):
            qid = item["qid"]
            if qid in qids:
                qids.remove(qid)
                new_output_file.write(item)
        for qid in tqdm(qids):
            triples = get_two_hop_neighborhood(qid, neighborhood_cache, pool)
            new_output_file.write({"qid": qid, "triples": [list(x) for x in triples]})
    all_relations = set()
    number_of_lines = sum(1 for line in open("data/dwie/neighborhoods_newt.jsonl"))
    pool.close()
    full_graph = gt.Graph(directed=True)
    e_prop = full_graph.new_edge_property("int")

    all_edges = set()
    for item in tqdm(jsonlines.open("data/dwie/neighborhoods_newt.jsonl"), total=number_of_lines):
        if not item["qid"].startswith("Q") or not item["qid"][1:].isdigit():
            continue

        qid = to_int(item["qid"])
        triples = [(to_int(triple[0]), to_int(triple[1]), to_int(triple[2])) for triple in item["triples"] if triple[0].startswith("Q") and triple[2].startswith("Q")]
        all_edges.update(triples)

        all_relations.update([x[1] for x in triples])
        # neighborhoods[qid] = triples
    for s, p, o in tqdm(all_edges):
        e = full_graph.add_edge(s, o)
        e_prop[e] = p
    full_graph.edge_properties["pid"] = e_prop
    # Dump graph
    full_graph.save("data/dwie/full_graph.gt")

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

def main():
    qids = set()
    for item in json.load(open("data/dwie/train_qid.json")):
        for mention in item["mentions"]:
            if "candidates" in mention:
                for candidate in mention["candidates"]:
                    qids.add(candidate)
    for item in json.load(open("data/dwie/dev_qid.json")):
        for mention in item["mentions"]:
            if "candidates" in mention:
                for candidate in mention["candidates"]:
                    qids.add(candidate)
    for item in json.load(open("data/dwie/test_qid.json")):
        for mention in item["mentions"]:
            if "candidates" in mention:
                for candidate in mention["candidates"]:
                    qids.add(candidate)

    counter = 0
    final_qids = set()
    for qid in qids:
        if not qid.startswith("Q") and not qid[1:].isdigit():
            counter += 1
        else:
            final_qids.add(qid)
    sample_neighborhoods(final_qids)
    # dump_examples(qid_pairs)




if __name__ == "__main__":
    main()