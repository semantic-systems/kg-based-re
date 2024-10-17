import json

from tqdm import tqdm
from wikimapper import WikiMapper



def main(filename: str):
    mapper = WikiMapper("wikipedia_data/index_enwiki-latest.db")
    content = json.load(open(filename))
    counter_entities=  0
    counter_qids = 0
    for item in tqdm(content):
        for entity in item["vertexSet"]:
            counter_entities += 1
            wikipedia_titles = set()
            for mention in entity:
                wikipedia_identifiers = mention["wikipedia_identifiers"]
                if wikipedia_identifiers is not None:
                    wikipedia_titles.update([x[0] for x in mention["wikipedia_identifiers"]])
            title_mapping = {}
            for title in wikipedia_titles:
                qid = mapper.title_to_id(title.replace(" ", "_"))
                if qid is None:
                    qid = title
                title_mapping[title] = qid
            for mention in entity:
                if mention["wikipedia_identifiers"] is not None:
                    mention["qid"] = [[title_mapping[x[0]], x[1]] for x in mention["wikipedia_identifiers"] if x[0] in title_mapping]
                    if mention["qid"]:
                        counter_qids += 1
                else:
                    mention["qid"] = []
    json.dump(content, open(filename.replace(".json", "_qid.json"), "w"), indent=4)


if __name__ == "__main__":
    main("data/docred/train_distant_wiki.json")
    # main("data/docred/test_revised_wiki.json")
    # main("data/docred/train_revised_wiki.json")

