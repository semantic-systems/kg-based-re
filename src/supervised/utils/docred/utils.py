import json
from collections import defaultdict

from refined.data_types.base_types import Span
from refined.inference.processor import Refined
from tqdm import tqdm
import spacy

def add_qids_to_redocred(name: str):

    redocred_file = f"data/docred/{name}.json"

    source = "en_core_web_trf"

    spacy_model = spacy.load(source)

    redocred = json.load(open(redocred_file))

    refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                      entity_set="wikipedia")

    for item in tqdm(redocred):
        entities = {}

        offsets = []
        offset = 0
        for sent in item["sents"]:
            offsets.append(offset)
            offset += len(sent)
        for idx, elem in enumerate(item["vertexSet"]):
            for mention in elem:
                sent_idx = mention["sent_id"]
                mention_pos = mention["pos"]
                mention_pos = (mention_pos[0] + offsets[sent_idx], mention_pos[1] + offsets[sent_idx])
                entities[mention_pos[0]] = (idx, mention_pos)
        full_text = []
        for sent in item["sents"]:
            full_text.extend(sent)

        current_idx = 0
        concatenated_text = ""
        spans = []
        while current_idx < len(full_text):
            if current_idx in entities:
                idx, mention_pos = entities[current_idx]
                end_position = mention_pos[1]
                text_start = len(concatenated_text)
                mention_str = " ".join(full_text[current_idx:end_position])
                concatenated_text += " " + mention_str
                text_end = len(concatenated_text)
                spans.append((Span(mention_str, text_start + 1, text_end-text_start), idx))
                current_idx = end_position
            else:
                concatenated_text += " " + full_text[current_idx]
                current_idx += 1

        processed = spacy_model(concatenated_text)
        qids = refined.process_text(concatenated_text,  [span[0] for span in spans])
        special_types = []
        for span in spans:
            span_start = span[0].start
            span_end = span[0].start + span[0].ln
            included_tokens = set(range(span_start, span_end))
            for entity in processed.ents:
                included_tokens_entity = set(range(entity.start_char, entity.end_char))
                if included_tokens.intersection(included_tokens_entity):
                    if entity.label_ in {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL"}:
                        special_types.append(entity.label_)
                    else:
                        special_types.append(None)
                    break


        entity_qids = defaultdict(list)
        for (_, idx), prediction, special_type in zip(spans, qids, special_types):
            candidate_entities = prediction.candidate_entities
            if special_type is not None:
                candidate_entities = [(special_type, 1.0) for x in candidate_entities]
            entity_qids[idx].extend(candidate_entities)

        for idx, elem in enumerate(item["vertexSet"]):
            candidates = entity_qids[idx]
            if candidates:
                count_per_qid = defaultdict(float)
                for candidate in candidates:
                    count_per_qid[candidate[0]] += candidate[1]
                sorted_qids = sorted(list(count_per_qid.items()), key=lambda x: x[1], reverse=True)
                best_qid = sorted_qids[0][0]
            else:
                best_qid = None
            for mention in elem:
                mention["qid"] = best_qid


    json.dump(redocred, open(f"data/docred/{name}_qid.json", "w"), indent=4)


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
    add_qids_to_redocred("train_revised")
    add_qids_to_redocred("dev_revised")
    add_qids_to_redocred("test_revised")
