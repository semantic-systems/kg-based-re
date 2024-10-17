import json
import graph_tool.all as gt

if __name__ == "__main__":
    full_graph = gt.load_graph("data/docred/full_graph_all.gt")

    all_props = set(full_graph.ep["pid"])

    sorted_all_props = sorted(list(all_props))
    prop_idx = {prop: idx for idx, prop in enumerate(sorted_all_props)}
    json.dump(prop_idx, open("data/docred/prop_dict_all.json", "w"))