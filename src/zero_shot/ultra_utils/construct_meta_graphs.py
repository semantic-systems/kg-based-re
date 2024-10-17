import copy
import json
import random
from collections import defaultdict

from SPARQLWrapper import SPARQLWrapper
import jsonlines
from SPARQLWrapper import JSON
from tqdm import tqdm
import graph_tool.all as gt

sparqlwrapper = SPARQLWrapper("http://sems-ai-1:1234/api/endpoint/sparql")



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



def get_triples_for_predicate(predicate):
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?s ?o WHERE {{
        ?s wdt:{predicate} ?o.
        FILTER (STRSTARTS(STR(?s), "http://www.wikidata.org/entity/Q"))
    }}
    LIMIT 100000
    """

    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    bindings = results["results"]["bindings"]
    random.shuffle(bindings)
    triples = []
    for result in bindings[:1000]:
        if result["o"]["type"] != "uri":
            continue
        s_id = result["s"]["value"]
        s_id = s_id.split("/")[-1]
        o_id = result["o"]["value"]
        o_id = o_id.split("/")[-1]
        triples.append((s_id, predicate, o_id))
    return triples


def process_fewrel(neighborhood_cache):
    main_path = "fewrel/unseen_5_seed_0"

    relations = set()
    for item in jsonlines.open(f"{main_path}/train_mapped.jsonl"):
        for triple in item["triplets"]:
            relations.add(triple["label_id"])
    for item in jsonlines.open(f"{main_path}/dev_mapped.jsonl"):
        for triple in item["triplets"]:
            relations.add(triple["label_id"])
    for item in jsonlines.open(f"{main_path}/test_mapped.jsonl"):
        for triple in item["triplets"]:
            relations.add(triple["label_id"])

    triples_per_relation = {}
    for relation in relations:
        triples = get_triples_for_predicate(relation)
        triples_per_relation[relation] = [list(x) for x in triples]

    all_qids = set()
    for triples in triples_per_relation.values():
        for triple in triples:
            all_qids.add(triple[0])
            all_qids.add(triple[2])

    neighborhood_cache = {}
    two_hop_neighborhoods = {}
    pool = None
    for qid in tqdm(all_qids):
        triples = get_two_hop_neighborhood(qid, neighborhood_cache, pool)
        two_hop_neighborhoods[qid] = [list(x) for x in triples]

    json.dump(two_hop_neighborhoods, open(f"fewrel/two_hop_neighborhoods.json", "w"))
    json.dump(triples_per_relation, open(f"fewrel/triples_per_relation.json", "w"))


def process_wiki(neighborhood_cache):
    main_path = "wiki/unseen_5_seed_0"

    relations = set()
    for item in jsonlines.open(f"{main_path}/train_mapped.jsonl"):
        for triple in item["triplets"]:
            relations.add(triple["label_id"])
    for item in jsonlines.open(f"{main_path}/dev_mapped.jsonl"):
        for triple in item["triplets"]:
            relations.add(triple["label_id"])
    for item in jsonlines.open(f"{main_path}/test_mapped.jsonl"):
        for triple in item["triplets"]:
            relations.add(triple["label_id"])

    triples_per_relation = {}
    for relation in tqdm(relations):
        triples = get_triples_for_predicate(relation)
        triples_per_relation[relation] = [list(x) for x in triples]

    all_qids = set()
    for triples in triples_per_relation.values():
        for triple in triples:
            all_qids.add(triple[0])
            all_qids.add(triple[2])

    two_hop_neighborhoods = {}
    pool = None
    for qid in tqdm(all_qids):
        triples = get_two_hop_neighborhood(qid, neighborhood_cache, pool)
        two_hop_neighborhoods[qid] = [list(x) for x in triples]

    json.dump(two_hop_neighborhoods, open(f"wiki/two_hop_neighborhoods.json", "w"))
    json.dump(triples_per_relation, open(f"wiki/triples_per_relation.json", "w"))


def create_meta_graphs(two_hop_neighborhoods_file, triples_per_relation_file, storage_path: str ):
    two_hop_neighborhoods = json.load(open(two_hop_neighborhoods_file))
    triples_per_relation = json.load(open(triples_per_relation_file))

    all_types = set()
    for relation, triples in triples_per_relation.items():
        print(relation)
        all_triples = {tuple(triple) for triple in triples}
        subject_qids = set()
        object_qids = set()
        for triple in triples:
            subject_qids.add(triple[0])
            object_qids.add(triple[2])


        subject_types_relation = defaultdict(int)
        object_types_relation = defaultdict(int)
        all_qids = subject_qids.union(object_qids)
        for qid in all_qids:
            if qid not in two_hop_neighborhoods:
                continue
            two_hop_neighbors = two_hop_neighborhoods[qid]
            for triple in two_hop_neighbors:
                all_triples.add(tuple(triple))
                if triple[1] == "P31" and (triple[0] == qid):
                    if qid in subject_qids:
                        subject_types_relation[triple[2]] += 1
                    if qid in object_qids:
                        object_types_relation[triple[2]] += 1
                    all_types.add(triple[2])


        filtered_subject_types = [
            type_ for type_, count in subject_types_relation.items() if count / len(subject_qids) > 0.05
        ]

        filtered_object_types = [
            type_ for type_, count in object_types_relation.items() if count / len(object_qids) > 0.05
        ]


        new_triples = []
        for s, p, o in all_triples:
            if not s.startswith("Q") or not o.startswith("Q") or not p.startswith("P"):
                continue
            s = int(s.replace("Q", ""))
            o = int(o.replace("Q", ""))
            p = int(p.replace("P", ""))
            new_triples.append((s, p, o))

        all_triples = new_triples


        outgoing = defaultdict(set)
        incoming = defaultdict(set)
        for s, p, o in all_triples:
            outgoing[s].add((p, o))
            incoming[o].add((p, s))

        head_head = defaultdict(int)
        head_tail = defaultdict(int)
        tail_head = defaultdict(int)
        tail_tail = defaultdict(int)

        property_counter =defaultdict(int)
        for s, p, o in tqdm(all_triples):
            props = set()
            for p_, o_ in outgoing[s]:
                if not p_ == p and not o_ == o:
                    props.add(p_)
            for p_ in props:
                head_head[(p, p_)] += 1
            props = set()
            for p_, s_ in incoming[o]:
                if not p_ == p and not s_ == s:
                    props.add(p_)
            for p_ in props:
                tail_tail[(p, p_)] += 1
            props = set()
            for p_, o_ in outgoing[o]:
                props.add(p_)
            for p_ in props:
                tail_head[(p, p_)] += 1
            props = set()
            for p_, s_ in incoming[s]:
                props.add(p_)
            for p_ in props:
                head_tail[(p, p_)] += 1
            property_counter[p] += 1

        gt_graph = gt.Graph(directed=True)
        graph_prop = gt_graph.new_edge_property("int")
        for (p, p_), count in head_head.items():
            ratio = max(count / property_counter[p], count / property_counter[p_])
            if ratio < 0.05:
                continue
            e = gt_graph.add_edge(p, p_)
            graph_prop[e] = 0
            e = gt_graph.add_edge(p_, p)
            graph_prop[e] = 0

        for (p, p_), count in head_tail.items():
            ratio = max(count / property_counter[p], count / property_counter[p_])
            if ratio < 0.05:
                continue
            e = gt_graph.add_edge(p, p_)
            graph_prop[e] = 1

        for (p, p_), count in tail_head.items():
            ratio = max(count / property_counter[p], count / property_counter[p_])
            if ratio < 0.05:
                continue
            e = gt_graph.add_edge(p, p_)
            graph_prop[e] = 2

        for (p, p_), count in tail_tail.items():
            ratio = max(count / property_counter[p], count / property_counter[p_])
            if ratio < 0.05:
                continue
            e = gt_graph.add_edge(p, p_)
            graph_prop[e] = 3
            e = gt_graph.add_edge(p_, p)
            graph_prop[e] = 3

        gt_graph.edge_properties["edge_type"] = graph_prop
        gt_graph.save(f"{storage_path}/{relation}_new.gt")
        json.dump(filtered_subject_types, open(f"{storage_path}/{relation}_filtered_subject_types.json", "w"))
        json.dump(filtered_object_types, open(f"{storage_path}/{relation}_filtered_object_types.json", "w"))
    json.dump(list(all_types), open(f"{storage_path}/all_types.json", "w"))


def create_types(two_hop_neighborhoods_file, triples_per_relation_file, storage_path: str ):
    two_hop_neighborhoods = json.load(open(two_hop_neighborhoods_file))
    triples_per_relation = json.load(open(triples_per_relation_file))

    all_types = set()
    for relation, triples in triples_per_relation.items():
        print(relation)
        all_triples = {tuple(triple) for triple in triples}
        subject_qids = set()
        object_qids = set()
        for triple in triples:
            subject_qids.add(triple[0])
            object_qids.add(triple[2])


        subject_types_relation = defaultdict(int)
        object_types_relation = defaultdict(int)
        all_qids = subject_qids.union(object_qids)
        for qid in all_qids:
            if qid not in two_hop_neighborhoods:
                continue
            two_hop_neighbors = two_hop_neighborhoods[qid]
            for triple in two_hop_neighbors:
                all_triples.add(tuple(triple))
                if triple[1] == "P31" and (triple[0] == qid):
                    if qid in subject_qids:
                        subject_types_relation[triple[2]] += 1
                    if qid in object_qids:
                        object_types_relation[triple[2]] += 1
                    all_types.add(triple[2])


        filtered_subject_types = [
            type_ for type_, count in subject_types_relation.items() if count / len(subject_qids) > 0.05
        ]

        filtered_object_types = [
            type_ for type_, count in object_types_relation.items() if count / len(object_qids) > 0.05
        ]
        json.dump(filtered_subject_types, open(f"{storage_path}/{relation}_filtered_subject_types.json", "w"))
        json.dump(filtered_object_types, open(f"{storage_path}/{relation}_filtered_object_types.json", "w"))
    json.dump(list(all_types), open(f"{storage_path}/all_types.json", "w"))


def get_types(qid):
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?type WHERE {{
        wd:{qid} wdt:P31 ?type.
    }}
    """
    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    types = set()
    for result in results["results"]["bindings"]:
        type_ = result["type"]["value"]
        type_ = type_.split("/")[-1]
        types.add(type_)
    return types

def get_stuff(triples_per_relation, two_hop_neighborhoods):
    all_triples = set()
    all_qids = set()

    for relation, triples in list(triples_per_relation.items())[:10]:
        print(relation)
        all_triples.update({tuple(triple) for triple in triples})
        qids = set()
        for triple in triples:
            qids.add(triple[0])
            qids.add(triple[2])

        all_qids.update(qids)

        for qid in qids:
            if qid not in two_hop_neighborhoods:
                continue
            two_hop_neighbors = two_hop_neighborhoods[qid]
            for triple in two_hop_neighbors:
                all_triples.add(tuple(triple))
                #all_qids.add(triple[0])
                #all_qids.add(triple[2])
    return all_triples, all_qids
def identify_type_pairs(two_hop_neighborhoods_file, triples_per_relation_file ):
    two_hop_neighborhoods = json.load(open(two_hop_neighborhoods_file))
    triples_per_relation = json.load(open(triples_per_relation_file))

    all_triples, all_qids = get_stuff(triples_per_relation, two_hop_neighborhoods)

    type_dict = {}
    all_types = defaultdict(int)
    for qid in tqdm(all_qids):
        types = get_types(qid)
        type_dict[qid] = types
        for type_ in types:
            all_types[type_] += 1

    type_pairs = defaultdict(lambda : defaultdict(int))
    for s, p, o in tqdm(all_triples):
        if s not in type_dict or o not in type_dict:
            continue
        for type_s in type_dict[s]:
            for type_o in type_dict[o]:
                type_pairs[p][(type_s, type_o)] += 1

    blah = 3





if __name__ == "__main__":
    neighborhood_cache = {}

    # process_fewrel(neighborhood_cache)
    # process_wiki(neighborhood_cache)

    #identify_type_pairs("fewrel/two_hop_neighborhoods.json", "fewrel/triples_per_relation.json")

    #create_types("fewrel/two_hop_neighborhoods.json", "fewrel/triples_per_relation.json", "fewrel/meta_graphs")
    create_types("wiki/two_hop_neighborhoods.json", "wiki/triples_per_relation.json", "wiki/meta_graphs")