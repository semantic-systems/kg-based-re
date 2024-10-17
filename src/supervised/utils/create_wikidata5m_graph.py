import graph_tool.all as gt
import jsonlines
from tqdm import tqdm


def main():
    full_graph = gt.Graph(directed=True)
    e_prop = full_graph.new_edge_property("int")



    for line in tqdm(open("data/wikidata5m_transductive/wikidata5m_transductive_train.txt")):
        sub, pred, obj = line.strip().split("\t")
        head_qid = int(sub[1:])
        p_qid = int(pred[1:])
        tail_qid = int(obj[1:])
        e = full_graph.add_edge(head_qid, tail_qid)
        e_prop[e] = p_qid

    full_graph.edge_properties["pid"] = e_prop
    # Dump graph
    full_graph.save(f"data/wikidata5m_graph.gt")


if __name__ == "__main__":
    main()

# import jsonlines
# from multiprocessing import Pool, cpu_count
# from tqdm import tqdm
#
#
# # Function to process a chunk of data and return a graph
# def process_chunk(chunk):
#     local_graph = gt.Graph(directed=True)
#     e_prop = local_graph.new_edge_property("int")
#
#     for item in chunk:
#         head_qid = int(item["id"][1:])
#         for p_qid, tail_qid in item["claims"]:
#             p_qid = int(p_qid[1:])
#             tail_qid = int(tail_qid[1:])
#             e = local_graph.add_edge(head_qid, tail_qid)
#             e_prop[e] = p_qid
#
#     local_graph.edge_properties["pid"] = e_prop
#     return local_graph
#
#
# # Read input file and divide into chunks
# def read_chunks(filename):
#     all_chunks = []
#     with jsonlines.open(filename) as reader:
#         for item in tqdm(reader, total=96046766):
#             all_chunks.append(item)
#     return all_chunks
#
#
# # Function to merge two graphs
# def merge_graphs(graph1, graph2):
#     graph1.add_edge_list(graph2.edges(), eprops=[graph2.edge_properties["pid"]])
#     return graph1
#
#
# if __name__ == "__main__":
#     input_file = "/storage/moeller/parsed_current_wikidata_dump.jsonl"
#     chunk_size = 100000  # Adjust chunk size based on available memory
#     num_workers = cpu_count()
#
#     # Process the file in parallel and collect the graphs
#     with Pool(processes=num_workers) as pool:
#         chunks = read_chunks(input_file)
#         split_into = len(chunks) // num_workers
#         chunks = [chunks[i:i + split_into] for i in range(0, len(chunks), split_into)]
#         results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))
#
#     # Merge all graphs into a single graph
#     full_graph = results[0]
#     for graph in results[1:]:
#         full_graph = merge_graphs(full_graph, graph)
#
#     # Dump the merged graph
#     full_graph.save(f"data/wikidata_graph.gt")
