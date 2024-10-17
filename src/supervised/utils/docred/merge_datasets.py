import json


def main(dataset_1, dataset_2):

    dataset_content = json.load(open(dataset_1))
    dataset_content_2 = json.load(open(dataset_2))

    assert len(dataset_content) == len(dataset_content_2)

    for item_1, item_2 in zip(dataset_content, dataset_content_2):
        positions_to_qid = {}
        for entity in item_2["entities"]:
            qid = entity["entity_linking"]["wikidata_resource"]
            for mention in entity["mentions"]:
                positions_to_qid[tuple(mention["pos"])] = qid
        for vertex in item_1["vertexSet"]:
            for elem in vertex:
                pos = tuple(elem["pos"])
                assert pos in positions_to_qid
                qid = positions_to_qid[pos]
                if qid is not None:
                    elem["qid"] = [[qid, 1.0]]
                else:
                    elem["qid"] = []
    json.dump(dataset_content, open(dataset_1, "w"))


if __name__ == "__main__":
    main("data/docred/train_annotated.json", "data/docred/train_annotated_linked.json")
    main("data/docred/dev.json", "data/docred/dev_linked.json")
    main("data/docred/test.json", "data/docred/test_linked.json")