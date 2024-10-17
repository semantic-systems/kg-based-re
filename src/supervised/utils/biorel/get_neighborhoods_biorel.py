import copy
import json
import multiprocessing
import os
from collections import defaultdict
from time import sleep
from typing import List

from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import jsonlines
import graph_tool.all as gt
sparqlwrapper = SPARQLWrapper("anonymized")

def create_single_hop_query(qids_str: str):
    single_hop_query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX bd: <http://www.bigdata.com/rdf#>
            SELECT ?s ?p ?o
            WHERE {{
              {{
                VALUES ?s {{ {qids_str} }}
                ?s ?p ?o .
                FILTER (!isLiteral(?o))
                FILTER (STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
                FILTER (STRSTARTS(STR(?o), "http://www.wikidata.org/entity/"))
              }}
              UNION
                {{ select ?s ?p ?o where{{
                    VALUES ?o {{ {qids_str} }}
                    ?s ?p ?o .
                    FILTER (!isLiteral(?o))
                    FILTER (STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
                    FILTER (STRSTARTS(STR(?o), "http://www.wikidata.org/entity/"))
                }}
                LIMIT 1000 }}
            }}
            """
    return single_hop_query

def create_two_hop_query(qids_str: str):
    two_hop_query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX bd: <http://www.bigdata.com/rdf#>
            SELECT ?s ?p ?o ?s2 ?p2 ?o2
            WHERE {{
              {{
                VALUES ?s {{ {qids_str} }}
                ?s ?p ?o .
                FILTER (!isLiteral(?o))
                FILTER (STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
                FILTER (STRSTARTS(STR(?o), "http://www.wikidata.org/entity/"))
                OPTIONAL {{
                ?o ?p2 ?o2 .
                FILTER (!isLiteral(?o2))
                FILTER (STRSTARTS(STR(?p2), "http://www.wikidata.org/prop/direct/"))
                FILTER (STRSTARTS(STR(?o2), "http://www.wikidata.org/entity/"))
                }}
              }}
            }}
            """
    return two_hop_query


def create_two_hop_query_full_outgoing(qids_str: str):
    two_hop_query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX bd: <http://www.bigdata.com/rdf#>
            SELECT ?s ?p ?o ?s2 ?p2 ?o2
            WHERE {{
                VALUES ?s {{ {qids_str} }}
                ?s ?p ?o .
                FILTER (!isLiteral(?o))
                FILTER (STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
                FILTER (STRSTARTS(STR(?o), "http://www.wikidata.org/entity/"))
                OPTIONAL {{
                ?o ?p2 ?o2 .
                FILTER (!isLiteral(?o2))
                FILTER (STRSTARTS(STR(?p2), "http://www.wikidata.org/prop/direct/"))
                FILTER (STRSTARTS(STR(?o2), "http://www.wikidata.org/entity/"))
                }}
            }}
            """
    return two_hop_query


def get_outgoing_edges(qids, global_outgoing_edge_cache):
    sub_graphs = defaultdict(set)
    new_qids = []
    for qid in qids:
        if qid in global_outgoing_edge_cache:
            sub_graphs[qid] = global_outgoing_edge_cache[qid]
            continue
        new_qids.append(qid)
    if not new_qids:
        return sub_graphs
    qids_str = " ".join([f"wd:{qid}" for qid in qids])

    two_hop_query = f"""
                PREFIX wd: <http://www.wikidata.org/entity/>
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                PREFIX wikibase: <http://wikiba.se/ontology#>
                PREFIX bd: <http://www.bigdata.com/rdf#>
                SELECT ?s ?p ?o
                WHERE {{
                    VALUES ?s {{ {qids_str} }}
                    ?s ?p ?o .
                    FILTER (STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
                    FILTER (!isLiteral(?o))
                    FILTER (STRSTARTS(STR(?o), "http://www.wikidata.org/entity/"))
                }}
                """
    #                     FILTER (STRSTARTS(STR(?o), "http://www.wikidata.org/entity/"))
    #                    FILTER (!isLiteral(?o))

    sparqlwrapper.setQuery(two_hop_query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    results = results["results"]["bindings"]
    for element in results:
        main_qid = element["s"]["value"].split("/")[-1]
        one_hop_p = element["p"]["value"].split("/")[-1]
        one_hop_o = element["o"]["value"].split("/")[-1]
        if one_hop_o.startswith("Q"):
            sub_graphs[main_qid].add((main_qid, one_hop_p, one_hop_o))
        else:
            sub_graphs[main_qid].add((main_qid, one_hop_p, "Literal"))

    return sub_graphs


def run_query(query, qid):
    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    results = results["results"]["bindings"]
    sub_graph = set()
    main_qid = qid[3:]
    for element in results:
        one_hop_s = element["s"]["value"].split("/")[-1]
        one_hop_p = element["p"]["value"].split("/")[-1]
        if one_hop_s.startswith("Q"):
            sub_graph.add((one_hop_s, one_hop_p, main_qid))
    return {main_qid: sub_graph}
def run_queries(queries: List[str], qids, pool=None):
    sub_graphs = defaultdict(set)
    if pool is None:
        for query, qid in zip(queries, qids):
            sub_graph = run_query(query, qid)
            sub_graphs[qid] = sub_graph
    else:
        sub_graphs_ = pool.starmap(run_query, zip(queries, qids))
        for sub_graph in sub_graphs_:
            sub_graphs.update(sub_graph)
    return sub_graphs

def get_incoming_edges(qids, global_incoming_edge_cache, pool=None):
    sub_graphs = defaultdict(set)

    queries_to_run = []
    qids_of_queries = []
    for qid in qids:
        if qid in global_incoming_edge_cache:
            sub_graphs[qid] = global_incoming_edge_cache[qid]
            continue
        qid = f"wd:{qid}"
        two_hop_query = f"""
                    PREFIX wd: <http://www.wikidata.org/entity/>
                    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                    PREFIX wikibase: <http://wikiba.se/ontology#>
                    PREFIX bd: <http://www.bigdata.com/rdf#>
                    SELECT ?s ?p
                    WHERE {{
                        ?s ?p {qid} .
                        FILTER (STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
                        FILTER (STRSTARTS(STR(?s), "http://www.wikidata.org/entity/"))
                    }}
                    LIMIT 100
                    """
        queries_to_run.append(two_hop_query)
        qids_of_queries.append(qid)

    sub_graphs.update(run_queries(queries_to_run, qids_of_queries, pool=pool))

    return sub_graphs


def run_outgoing_query(query):
    prefixes = {"http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#",
                "http://purl.bioontology.org/ontology/MEDLINEPLUS/"}
    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    results = results["results"]["bindings"]
    sub_graph = defaultdict(set)
    for element in results:
        one_hop_s = (element["s"]["type"], element["s"]["value"])
        one_hop_p = (element["p"]["type"], element["p"]["value"])
        one_hop_o = (element["o"]["type"], element["o"]["value"])
        sub_graph[one_hop_s].add((one_hop_s, one_hop_p, one_hop_o))
    return sub_graph


def run_outgoing_queries(queries: List[str], qid_batches, pool=None):
    sub_graphs = defaultdict(set)
    if pool is None:
        for query, qids in zip(queries, qid_batches):
            sub_graph = run_outgoing_query(query)
            sub_graphs.update(sub_graph)
    else:
        sub_graphs_ = pool.map(run_outgoing_query, queries)
        for sub_graph in sub_graphs_:
            sub_graphs.update(sub_graph)
    return sub_graphs

def get_batch_outgoing_edges(qids, global_outgoing_edge_cache, pool, batch_size=100):
    sub_graphs = defaultdict(set)

    queries_to_run = []
    qids_of_queries = []

    filtered_qids = []
    for qid in qids:
        if qid in global_outgoing_edge_cache:
            sub_graphs[qid] = global_outgoing_edge_cache[qid]
            continue
        filtered_qids.append(qid)

    batched_qids = [filtered_qids[i:i + batch_size] for i in range(0, len(filtered_qids), batch_size)]

    for batch in batched_qids:
        qids_str = " ".join([f"<{qid[1]}>" for qid in batch if qid[0] == 'uri'])
        blank_nodes_str = " ".join([f"_:{qid[1]}" for qid in batch if qid[0] != 'uri']) # if qid[0] == 'bnode'
        if qids_str and blank_nodes_str:
            two_hop_query = f"""
                            SELECT ?s ?p ?o
                            WHERE {{
                                {{
                                VALUES ?s {{ {qids_str} }}
                                ?s ?p ?o .
                                FILTER (!isLiteral(?o))
                                }}
                                UNION
                                {{
                                VALUES ?s {{ {blank_nodes_str} }}
                                ?s ?p ?o .
                                FILTER (!isLiteral(?o))
                                }}
                            }}
                            """
        elif qids_str:
            two_hop_query = f"""
                            SELECT ?s ?p ?o
                            WHERE {{
                                VALUES ?s {{ {qids_str} }}
                                ?s ?p ?o .
                                FILTER (!isLiteral(?o))
                            }}
                            """
        elif blank_nodes_str:
            two_hop_query = f"""
                            SELECT ?s ?p ?o
                            WHERE {{
                                VALUES ?s {{ {blank_nodes_str} }}
                                ?s ?p ?o .
                                FILTER (!isLiteral(?o))
                            }}
                            """
        else:
            continue
        queries_to_run.append(two_hop_query)
        qids_of_queries.append(batch)

    sub_graphs.update(run_outgoing_queries(queries_to_run, qids_of_queries, pool=pool))


    return sub_graphs


def get_full_two_hop_neighborhood(main_qids, global_outgoing_edge_cache, global_incoming_edge_cache, batch_size=100, pool=None):
    all_outgoing_edges = {}
    all_incoming_edges = {}
    outgoing_edges = get_outgoing_edges(main_qids, global_outgoing_edge_cache)
    global_outgoing_edge_cache.update(outgoing_edges)
    incoming_edges = get_incoming_edges(main_qids, global_incoming_edge_cache, pool=pool)
    global_incoming_edge_cache.update(incoming_edges)
    found_qids = {x[2] for v in outgoing_edges.values() for x in v if x[2].startswith("Q")}
    found_qids.update({x[0] for v in incoming_edges.values() for x in v})
    all_outgoing_edges.update(outgoing_edges)
    all_incoming_edges.update(incoming_edges)
    for idx in range(0, len(found_qids), batch_size):
        qids = list(found_qids)[idx:idx+batch_size]
        outgoing_edges = get_outgoing_edges(qids, global_outgoing_edge_cache)
        global_outgoing_edge_cache.update(outgoing_edges)
        incoming_edges = get_incoming_edges(qids, global_incoming_edge_cache, pool=pool)
        global_incoming_edge_cache.update(incoming_edges)
        all_outgoing_edges.update(outgoing_edges)
        all_incoming_edges.update(incoming_edges)

    sub_graphs = defaultdict(set)
    for qid in main_qids:
        sub_graphs[qid] = all_outgoing_edges.get(qid, set()).union(all_incoming_edges.get(qid, set()))
        interacted_qids = {x[2] for x in sub_graphs[qid] if x[2].startswith("Q")}
        interacted_qids.update({x[0] for x in sub_graphs[qid]})
        for interacted_qid in interacted_qids:
            sub_graphs[qid].update(all_outgoing_edges.get(interacted_qid, set()))
            sub_graphs[qid].update(all_incoming_edges.get(interacted_qid, set()))

    blah = 3

    return sub_graphs

def get_outgoing_three_hop_neighborhood(main_qids, global_outgoing_edge_cache, global_incoming_edge_cache, n_hops, batch_size=100, pool=None):
    all_outgoing_edges = {}
    all_incoming_edges = {}
    outgoing_edges = get_batch_outgoing_edges(main_qids, global_outgoing_edge_cache, pool)
    global_outgoing_edge_cache.update(outgoing_edges)
    # incoming_edges = get_incoming_edges(main_qids, global_incoming_edge_cache, pool=pool)
    # global_incoming_edge_cache.update(incoming_edges)
    found_qids = {x[2] for v in outgoing_edges.values() for x in v}
    # found_qids.update({x[0] for v in incoming_edges.values() for x in v})
    all_outgoing_edges.update(outgoing_edges)
    # all_incoming_edges.update(incoming_edges)
    seen_qids = set(main_qids)
    for i in range(n_hops - 1):
        new_found_qids = set()
        found_qids -= seen_qids
        outgoing_edges = get_batch_outgoing_edges(found_qids, global_outgoing_edge_cache, pool)
        global_outgoing_edge_cache.update(outgoing_edges)
        all_outgoing_edges.update(outgoing_edges)
        new_found_qids.update({x[2] for v in outgoing_edges.values() for x in v})
        seen_qids.update(found_qids)
        found_qids = new_found_qids

    sub_graphs = defaultdict(set)
    for qid in main_qids:
        sub_graphs[qid] = copy.deepcopy(all_outgoing_edges.get(qid, set()))
        interacted_qids = {x[2] for x in sub_graphs[qid]}
        interacted_qids.update({x[0] for x in sub_graphs[qid]})

        for i in range(n_hops - 1):
            new_interacted_qids = set()
            for interacted_qid in interacted_qids:
                outgoing_edges = all_outgoing_edges.get(interacted_qid, set())
                sub_graphs[qid].update(outgoing_edges)
                new_interacted_qids.update({x[2] for x in outgoing_edges })
            interacted_qids = new_interacted_qids

    return sub_graphs

def get_two_hop_neighborhood(qids_str):
    query = create_two_hop_query(qids_str)
    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    results = results["results"]["bindings"]
    sub_graphs = defaultdict(set)
    for element in results:
        main_qid = element["s"]["value"].split("/")[-1]
        one_hop_p = element["p"]["value"].split("/")[-1]
        one_hop_o = element["o"]["value"].split("/")[-1]
        sub_graphs[main_qid].add((main_qid, one_hop_p, one_hop_o))
        if "p2" not in element:
            continue
        two_hop_p = element["p2"]["value"].split("/")[-1]
        two_hop_o = element["o2"]["value"].split("/")[-1]
        sub_graphs[main_qid].add((one_hop_o, two_hop_p, two_hop_o))

    return sub_graphs


def get_one_hop_neighborhood(qid: str, num_neighbors: int = 50):
    triples = set()
    query = f"""
    SELECT * WHERE {{
        {{<{qid}> ?p ?o.
        FILTER(isIRI(?o))}}
        UNION
        {{<{qid}> ?p ?o.
        ?o ?p2 ?o2.
        FILTER(isIRI(?o2))
        FILTER(isBlank(?o))}}
    }}
    LIMIT {num_neighbors}
    """
    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    identified_qids = set()
    for result in results["results"]["bindings"]:
        if "p2" in result:
            pid = result["p"]["value"]
            blank_id = qid + pid
            o_id = result["o2"]["value"]
            pid2 = result["p2"]["value"]
            triples.add((qid, pid, blank_id))
            triples.add((blank_id, pid2, o_id))
            identified_qids.add(o_id)
        else:
            pid = result["p"]["value"]
            o_id = result["o"]["value"]
            triples.add((qid, pid, o_id))
            identified_qids.add(o_id)
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT * WHERE {{
        {{
        ?s ?p <{qid}>.
        FILTER(isIRI(?s))
        }}
        UNION
        {{
        ?s ?p <{qid}>.
        ?s2 ?p2 ?s.
        FILTER(isIRI(?s2))
        FILTER(isBlank(?s))
        }}
    }}
    LIMIT {num_neighbors}
    """
    sparqlwrapper.setQuery(query)
    sparqlwrapper.setReturnFormat(JSON)
    results = sparqlwrapper.query().convert()
    for result in results["results"]["bindings"]:
        if "p2" in result:
            pid = result["p"]["value"]
            blank_id = qid + pid
            s_id = result["s2"]["value"]
            pid2 = result["p2"]["value"]
            triples.add((blank_id, pid, qid))
            triples.add((s_id, pid2, blank_id))
            identified_qids.add(s_id)
        else:
            pid = result["p"]["value"]
            s_id = result["s"]["value"]
            triples.add((s_id, pid, qid))
            identified_qids.add(s_id)

    return triples, identified_qids


def get_neighborhoods(qids, n_hops=3):

    graph = set()

    global_neighbor_cache = {}
    for qid in tqdm(qids):
        current_qids = [qid]
        all_triples = set()
        encountered_qids = {qid}
        for k in range(n_hops):
            new_current_qids = set()
            for qid_ in current_qids:
                if qid_ in global_neighbor_cache:
                    triples, identified_qids = global_neighbor_cache[qid_]
                else:
                    triples, identified_qids = get_one_hop_neighborhood(qid_)
                    global_neighbor_cache[qid_] = (triples, identified_qids)
                all_triples.update(triples)
                new_current_qids.update(identified_qids)
            current_qids = new_current_qids - encountered_qids
            encountered_qids.update(new_current_qids)
        graph.update(all_triples)

    full_graph = gt.Graph(directed=True)
    e_prop = full_graph.new_edge_property("int")
    node_index = {}
    relation_index = {}

    for s, p, o in graph:
        if s not in node_index:
            node_index[s] = len(node_index)
        if o not in node_index:
            node_index[o] = len(node_index)
        e = full_graph.add_edge(node_index[s], node_index[o])
        if p not in relation_index:
            relation_index[p] = len(relation_index)
        e_prop[e] = relation_index[p]
    full_graph.ep["pid"] = e_prop
    full_graph.save("data/biorel/full_graph_3_hop.gt")
    json.dump(relation_index, open("data/biorel/relation_index.json", "w"))
    json.dump(node_index, open("data/biorel/node_index.json", "w"))

def get_one_hop_query(h_id, t_id):
    one_hop_query = f"""
                SELECT * WHERE {{
                    VALUES ?s1 {{ <{h_id}> }}
                    VALUES ?o1 {{ <{t_id}> }}
                    ?s1 ?p1 ?o1 .
                }}
                """
    return one_hop_query

def get_two_hop_query(h_id, t_id):
    two_hop_query = f"""
                SELECT * WHERE {{
                    VALUES ?s1 {{ <{h_id}> }}
                    VALUES ?o2 {{ <{t_id}> }}
                    ?s1 ?p1 ?o1 .
                    ?o1 ?p2 ?o2.
                }}
                """
    return two_hop_query

def get_three_hop_query(h_id, t_id):
    three_hop_query = f"""
                SELECT * WHERE {{
                VALUES ?s1 {{ <{h_id}> }}
                    VALUES ?o3 {{ <{t_id}> }}
                    ?s1 ?p1 ?o1 .
                    ?o1 ?p2 ?o2 .
                    ?o2 ?p3 ?o3 .
                }}"""
    return three_hop_query


def get_four_hop_query(h_id, t_id):
    four_hop_query = f"""
                SELECT * WHERE {{
                    VALUES ?s1 {{ <{h_id}> }}
                    VALUES ?o4 {{ <{t_id}> }}
                    ?s1 ?p1 ?o1 .
                    ?o1 ?p2 ?o2 .
                    ?o2 ?p3 ?o3 .
                    ?o3 ?p4 <{t_id}> .
                }}"""
    return four_hop_query

def get_paths(pair):
    pair, h_ids, t_ids = pair
    paths = []
    for h_id in h_ids:
        for t_id in t_ids:
            one_hop_query = get_one_hop_query(h_id, t_id)
            two_hop_query = get_two_hop_query(h_id, t_id)
            three_hop_query = get_three_hop_query(h_id, t_id)
            four_hop_query = get_four_hop_query(h_id, t_id)

            queries = [one_hop_query, two_hop_query, three_hop_query, four_hop_query]

            one_hop_query_inv = get_one_hop_query(t_id, h_id)
            two_hop_query_inv = get_two_hop_query(t_id, h_id)
            three_hop_query_inv = get_three_hop_query(t_id, h_id)
            four_hop_query_inv = get_four_hop_query(t_id, h_id)

            queries_inv = [one_hop_query_inv, two_hop_query_inv, three_hop_query_inv, four_hop_query_inv]

            for query in queries:
                sparqlwrapper.setQuery(query)
                sparqlwrapper.setReturnFormat(JSON)
                results = sparqlwrapper.query().convert()
                results = results["results"]["bindings"]
                for element in results:
                    path = []
                    for idx in range(1, 5):
                        if f"p{idx}" in element:
                            if idx == 1:
                                head_entity = f"s{idx}"
                            else:
                                head_entity = f"o{idx-1}"
                            path.append((element[head_entity]["value"], element[f"p{idx}"]["value"], element[f"o{idx}"]["value"]))
                    paths.append(path)

            for query in queries_inv:
                sparqlwrapper.setQuery(query)
                sparqlwrapper.setReturnFormat(JSON)
                results = sparqlwrapper.query().convert()
                results = results["results"]["bindings"]
                for element in results:
                    path = []
                    for idx in range(1, 5):
                        if f"p{idx}" in element:
                            if idx == 1:
                                head_entity = f"s{idx}"
                            else:
                                head_entity = f"o{idx - 1}"
                            path.append((element[head_entity]["value"], element[f"p{idx}"]["value"],
                                         element[f"o{idx}"]["value"]))
                    paths.append(path)
    return pair, paths

    
    
def get_neighborhoods_new(qid_pairs, special_identifiers, batch_size=10, n_hops=3):

    existing_qids = set()
    subgraphs_file = f"data/biorel/sub_graphs_{n_hops}_hop.jsonl"


    output_file = jsonlines.open(subgraphs_file, "a")

    global_outgoing_edge_cache = {}
    global_incoming_edge_cache = {}
    sub_graphs = defaultdict(set)
    pool = multiprocessing.Pool(6)
    all_valid_pairs = []
    for pair in tqdm(qid_pairs):
        h_ids = special_identifiers[pair[0]]
        t_ids = special_identifiers[pair[1]]
        if h_ids and t_ids:
            all_valid_pairs.append((pair, h_ids, t_ids))

    pool = multiprocessing.Pool(6)

    all_paths = tqdm(pool.imap(get_paths, all_valid_pairs), total=len(all_valid_pairs))
    for pair, paths in all_paths:
        for path in paths:
            for triple in path:
                sub_graphs[pair[0]].add(triple)
                sub_graphs[pair[1]].add(triple)

    for qid, subgraph in sub_graphs.items():
        subgraph = [list(x) for x in subgraph]
        output_file.write({"qid": qid, "subgraph": subgraph})
        

def get_qids(name: str):
    filename = f"data/biorel/{name}.json"
    qid_pairs = set()

    for idx, item in enumerate(json.load(open(filename))):

        qid_pairs.add((item["head"]["CUI"], item["tail"]["CUI"]))

    return qid_pairs

def check_coverage(qids):
    #http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#
    list_qids = list(qids)
    found = set()
    special_identifiers = defaultdict(set)
    all_identifiers = set()
    for qid in tqdm(qids):
        query = f"""
        PREFIX th: <http://purl.bioontology.org/ontology/MEDLINEPLUS/>
        SELECT * WHERE {{
        ?s ?p "{qid}" .
        }}
        """
        sparqlwrapper.setQuery(query)
        sparqlwrapper.setReturnFormat(JSON)
        results = sparqlwrapper.query().convert()
        results = results["results"]["bindings"]
        for elem in results:
            found.add(qid)
            special_identifiers[qid].add(elem["s"]["value"])
            all_identifiers.add(elem["s"]["value"])

    missing = qids - found
    print(f"Ratio of missing qids: {len(missing) / len(qids)}")
    return all_identifiers, special_identifiers

if __name__ == "__main__":

    qid_pairs =  get_qids("train")
    qid_pairs.update(get_qids("test"))
    qid_pairs.update(get_qids("dev"))
    qids = set()
    for pair in qid_pairs:
        qids.add(pair[0])
        qids.add(pair[1])
    all_identifiers, special_identifiers = check_coverage(qids)
    special_identifiers = {k: list(v) for k, v in special_identifiers.items()}
    json.dump(special_identifiers, open("data/biorel/identifier_mapping.json", "w"))
    get_neighborhoods(all_identifiers)

