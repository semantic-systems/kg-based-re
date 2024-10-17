import json
import os
import random


def main():
    relation_dict = {}
    train = []
    dev = []
    test = []
    for file in os.listdir("data/dwie/data/annos_with_content"):
        if file.endswith(".json"):
            document = json.load(open(f"data/dwie/data/annos_with_content/{file}"))
            tags = set(document["tags"])
            text = document["content"]
            for relation in document["relations"]:
                if relation["p"] not in relation_dict:
                    relation_dict[relation["p"]] = len(relation_dict)
            example = {
                "text": text,
                "mentions": document["mentions"],
                "relations": document["relations"],
                "concepts": document["concepts"]
            }
            if "train" in tags:
                train.append(example)
            elif "test" in tags:
                test.append(example)
            else:
                raise ValueError(f"Unknown tag in {file}: {tags}")
    dev_sampled = random.sample(list(range(len(train))) , 100)
    for i in sorted(dev_sampled, reverse=True):
        dev.append(train[i])
        del train[i]
    json.dump(relation_dict, open("data/dwie/relation_dict.json", "w"))
    json.dump(train, open("data/dwie/train.json", "w"))
    json.dump(dev, open("data/dwie/dev.json", "w"))
    json.dump(test, open("data/dwie/test.json", "w"))

if __name__ == "__main__":
    main()