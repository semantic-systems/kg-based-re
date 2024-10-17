import json
import logging
from collections import defaultdict

from tqdm import tqdm
import spacy
import blink.main_dense as main_dense
import argparse


def prepare_entity_representations(full_text: str, entities: list, offset, tokenizer):
    reps = []
    for idx, entity in enumerate(entities):
        context_left = full_text[:entity[1]].lower().strip()
        context_right = full_text[entity[1] + entity[2]:].lower().strip()
        mention = full_text[entity[1]:entity[1] + entity[2]].lower()

        tokenized_context_left = tokenizer.tokenize(context_left)
        tokenized_mention = tokenizer.tokenize(mention)
        tokenized_context_right = tokenizer.tokenize(context_right)
        l_mention = len(tokenized_mention)
        l_context_left = len(tokenized_context_left)
        l_context_right = len(tokenized_context_right)
        maximum_size = 250
        left_over_size = maximum_size - l_mention
        equal_split = left_over_size // 2
        left_context_size = min(equal_split, l_context_left)
        right_context_size = min(left_over_size - left_context_size, l_context_right)

        context_left = tokenizer.convert_tokens_to_string(tokenized_context_left[-left_context_size:])
        context_right = tokenizer.convert_tokens_to_string(tokenized_context_right[:right_context_size])

        rep =  {
            "id": idx + offset,
            "label": "unknown",
            "label_id": -1,
            "context_left": context_left,
            "mention": mention,
            "context_right": context_right,
        }
        reps.append(rep)
    return reps
def add_qids_to_redocred(name: str):

    redocred_file = f"data/docred/{name}.json"

    source = "en_core_web_trf"

    spacy_model = spacy.load(source)

    redocred = json.load(open(redocred_file))

    models_path = "BLINKRep/models/"  # the path where you stored the BLINK models

    config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": 10,
        "biencoder_model": models_path + "biencoder_wiki_large.bin",
        "biencoder_config": models_path + "biencoder_wiki_large.json",
        "entity_catalogue": models_path + "entity.jsonl",
        "entity_encoding": models_path + "all_entities_large.t7",
        "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
        "crossencoder_config": models_path + "crossencoder_wiki_large.json",
        "fast": False,  # set this to be true if speed is a concern
        "output_path": "logs/",  # logging directory
        "index_path":  models_path + "faiss_hnsw_index.pkl",
        "faiss_index": "hnsw"
    }

    args = argparse.Namespace(**config)

    models = main_dense.load_models(args, logger=None)

    models[1]["eval_batch_size"] = 64
    models[1]["encode_batch_size"] = 64
    models[3]["eval_batch_size"] = 16

    # refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
    #                                   entity_set="wikipedia")



    all_data_to_link = []

    all_spans = []
    entity_spans = []
    entity_offset = 0
    for idx, item in enumerate(tqdm(redocred)):
        entities = {}

        offsets = []
        offset = 0
        for sent in item["sents"]:
            offsets.append(offset)
            offset += len(sent)

        full_text = []
        for sent in item["sents"]:
            full_text.extend(sent)
        spans = []
        for idx, elem in enumerate(item["vertexSet"]):
            for mention in elem:
                sent_idx = mention["sent_id"]
                mention_pos = mention["pos"]
                mention_pos = (mention_pos[0] + offsets[sent_idx], mention_pos[1] + offsets[sent_idx])
                entities[mention_pos[0]] = (idx, mention_pos)
                before = " ".join(full_text[:mention_pos[0]])
                text_start = len(before) + int(len(before) > 0)
                mention_text = " ".join(full_text[mention_pos[0]:mention_pos[1]])
                spans.append(((mention["name"], text_start, len(mention_text)), idx))


        concatenated_text = " ".join(full_text)

        processed = None
        # qids = refined.process_text(concatenated_text,  [span[0] for span in spans])
        data_to_link = prepare_entity_representations(concatenated_text, [span[0] for span in spans], entity_offset, models[0].tokenizer)
        entity_offset += len(data_to_link)
        all_spans.append((spans, processed, item))
        entity_spans.append((len(all_data_to_link), len(all_data_to_link) + len(data_to_link)))
        all_data_to_link.extend(data_to_link)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    _, _, _, _, _, all_predictions, all_scores, = main_dense.run(args, root, *models, test_data=all_data_to_link)
    for (spans, processed, item), (start, end) in zip(all_spans, entity_spans):
        predictions = all_predictions[start:end]
        scores = all_scores[start:end]
        assert len(predictions) == len(spans)
        special_types = []
        for span in spans:
            span_start = span[0][1]
            span_end = span[0][1] + span[0][2]
            included_tokens = set(range(span_start, span_end))
            if processed is not None:
                for entity in processed.ents:
                    included_tokens_entity = set(range(entity.start_char, entity.end_char))
                    if included_tokens.intersection(included_tokens_entity):
                        if entity.label_ in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"}:
                            special_types.append(entity.label_)
                        else:
                            special_types.append(None)
                        break
                else:
                    special_types.append(None)


        entity_qids = defaultdict(list)
        for (_, idx), prediction, special_type in zip(spans, zip(predictions, scores), special_types):
            candidate_entities = zip(*prediction)
            if special_type is not None:
                candidate_entities = [(special_type, 1.0)]
            entity_qids[idx].extend(candidate_entities)

        for idx, elem in enumerate(item["vertexSet"]):
            candidates = entity_qids[idx]
            if candidates:
                count_per_qid = defaultdict(float)
                for candidate in candidates:
                    count_per_qid[candidate[0]] += candidate[1]
                sorted_qids = sorted(list(count_per_qid.items()), key=lambda x: x[1], reverse=True)
                wikipedia_identifiers = [list(x) for x in sorted_qids]
            else:
                wikipedia_identifiers = None
            for mention in elem:
                mention["wikipedia_identifiers"] = wikipedia_identifiers


    json.dump(redocred, open(f"data/docred/{name}_wiki.json", "w"), indent=4)


def add_qids_to_redocred_alt(name: str):
    redocred_file = f"data/docred/{name}_revised.json"
    qid_file = f"data/docred/{name}_joint_qcode.json"

    redocred = json.load(open(redocred_file))
    qid_docred = json.load(open(qid_file))

    mention_to_qid = {}
    qid_docs = {}
    for item in qid_docred:
        qid_docs[item["title"]] = item
        for entity in item["vertexSet"]:
            qids = defaultdict(int)
            for mention in entity:
                qids[mention["wikidata_qcode"]] += 1
            if len(qids) > 1:
                qid = max(qids, key=lambda x: qids[x])
            else:
                qid=  qids.popitem()[0]
            for mention in entity:
                mention_to_qid[mention["name"].lower()] = qid
    for item in redocred:
        title = item["title"]
        item_vertexSet = item["vertexSet"]
        # assert len(item_vertexSet) == len(qid_vertexSet)

        new_vertexSet = []
        for entity_1 in zip(item_vertexSet):
            mentions_1 = set()
            for mention in entity_1:
                mentions_1.add(mention["name"])
            mentions_2 = set()
            qids = set()

            for mention in entity_1:
                mention["wikidata_qcode"] = list(qids)

            new_vertexSet.append(entity_1)
            item["vertexSet"] = new_vertexSet

    json.dump(redocred, open(f"data/docred/{name}_revised_qid.json", "w"), indent=4)


if __name__ == "__main__":
    add_qids_to_redocred("train_distant")
    # add_qids_to_redocred("dev_revised")
    # add_qids_to_redocred("test_revised")
