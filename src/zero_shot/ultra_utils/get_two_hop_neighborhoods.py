import json

import jsonlines
from tqdm import tqdm

from src.zero_shot.ultra_utils.construct_meta_graphs import get_two_hop_neighborhood


def retrieve_wiki():
    two_hop_neighborhoods = {}
    neighborhood_cache = {}
    pool = None
    all_qids = set()
    for item in jsonlines.open(f"wiki/unseen_5_seed_0/train_mapped.jsonl"):
        for triple in item["triplets"]:
            all_qids.add(triple["head_id"])
            all_qids.add(triple["tail_id"])
    for item in jsonlines.open(f"wiki/unseen_5_seed_0/dev_mapped.jsonl"):
        for triple in item["triplets"]:
            all_qids.add(triple["head_id"])
            all_qids.add(triple["tail_id"])
    for item in jsonlines.open(f"wiki/unseen_5_seed_0/test_mapped.jsonl"):
        for triple in item["triplets"]:
            all_qids.add(triple["head_id"])
            all_qids.add(triple["tail_id"])

    for qid in tqdm(all_qids):
        if not qid.startswith("Q") or not qid[1:].isdigit():
            continue
        triples = get_two_hop_neighborhood(qid, neighborhood_cache, pool)
        two_hop_neighborhoods[qid] = [list(x) for x in triples]

    json.dump(two_hop_neighborhoods, open(f"wiki/two_hop_neighborhoods_data.json", "w"))

def retrieve_fewrel():
    two_hop_neighborhoods = {}
    neighborhood_cache = {}
    pool = None
    all_qids = set()
    for item in jsonlines.open(f"fewrel/unseen_5_seed_0/train_mapped.jsonl"):
        for triple in item["triplets"]:
            all_qids.add(triple["head_id"])
            all_qids.add(triple["tail_id"])
    for item in jsonlines.open(f"fewrel/unseen_5_seed_0/dev_mapped.jsonl"):
        for triple in item["triplets"]:
            all_qids.add(triple["head_id"])
            all_qids.add(triple["tail_id"])
    for item in jsonlines.open(f"fewrel/unseen_5_seed_0/test_mapped.jsonl"):
        for triple in item["triplets"]:
            all_qids.add(triple["head_id"])
            all_qids.add(triple["tail_id"])

    for qid in tqdm(all_qids):
        if not qid.startswith("Q"):
            continue
        triples = get_two_hop_neighborhood(qid, neighborhood_cache, pool)
        two_hop_neighborhoods[qid] = [list(x) for x in triples]

    json.dump(two_hop_neighborhoods, open(f"fewrel/two_hop_neighborhoods_data.json", "w"))


if __name__ == "__main__":
    retrieve_wiki()
    retrieve_fewrel()