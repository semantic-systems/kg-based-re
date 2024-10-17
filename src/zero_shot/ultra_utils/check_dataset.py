import json
import pickle
from pathlib import Path

import jsonlines


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
                if not (s.startswith("Q") and o.startswith("Q") and p.startswith("P") and s[1:].isdigit() and o[
                                                                                                              1:].isdigit() and p[
                                                                                                                                1:].isdigit()):
                    continue
                s = int(s[1:])
                o = int(o[1:])
                p = int(p[1:])
                triples.append((s, p, o))
            new_data[key] = triples
        pickle.dump(new_data, open(pickle_path, "wb"))
        return data

def main(path: str, graph):
    content = list(jsonlines.open(path))
    direct_connect_counter = 0
    counter = 0
    for item in content:
        for triplet in item["triplets"]:
            if not triplet["head_id"].startswith("Q") or not triplet["tail_id"].startswith("Q"):
                continue
            head_id = int(triplet["head_id"][1:])
            tail_id = int(triplet["tail_id"][1:])
            label_id = int(triplet["label_id"][1:])
            if head_id not in graph or tail_id not in graph:
                continue
            if (head_id, label_id, tail_id) in graph[head_id]:
                direct_connect_counter += 1
            counter += 1
    ratio = direct_connect_counter / counter
    print(ratio)
    blah = 3

    blah = 3



if __name__ == "__main__":
    path = "fewrel/unseen_5_seed_0/train_mapped.jsonl"
    two_hop_neighborhoods = load_neighborhoods(f"fewrel/two_hop_neighborhoods_data.json")
    main(path, two_hop_neighborhoods)