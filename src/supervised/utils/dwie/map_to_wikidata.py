import json

from tqdm import tqdm
from wikimapper import WikiMapper



def main(filename: str):
    mapper = WikiMapper("wikipedia_data/index_enwiki-latest.db")
    content = json.load(open(filename))
    counter_entities=  0
    counter_qids = 0
    for item in tqdm(content):
        for mention in item["mentions"]:
            counter_entities += 1
            wikipedia_titles = set()
            if "candidates" not in mention:
                continue
            for candidate in mention["candidates"]:
                wikipedia_titles.add(candidate)
            title_mapping = {}
            for title in wikipedia_titles:
                qid = mapper.title_to_id(title.replace(" ", "_"))
                if qid is None:
                    qid = title
                title_mapping[title] = qid
            mention["candidates"] = [title_mapping[x] for x in mention["candidates"] if x in title_mapping]
    # json.dump(content, open(filename.replace(".json", "_qid.json"), "w"), indent=4)


if __name__ == "__main__":
    main("data/dwie/train.json")
    main("data/dwie/dev.json")
    main("data/dwie/test.json")

