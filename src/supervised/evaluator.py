import json
import os
from collections import defaultdict

import numpy as np


class Evaluator:
    def __init__(self, id2rel, path, train_file="train_annotated.json", dev_file="dev.json"):
        id2rel = {k + 1: v for k, v in id2rel.items()}
        id2rel[0] = 'Nolabel'
        self.id2rel = id2rel
        truth_dir = os.path.join(path, 'ref')

        if not os.path.exists(truth_dir):
            os.makedirs(truth_dir)

        self.fact_in_train_annotated = self.gen_train_facts(os.path.join(path, train_file), truth_dir)
        self.fact_in_train_distant = self.gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

        self.truth = json.load(open(os.path.join(path, dev_file)))


    def extract_relative_score(self, scores: list, topks: list) -> list:
        '''
        Get relative score from topk predictions.
        Input:
            :scores: a list containing scores of topk predictions.
            :topks: a list containing relation labels of topk predictions.
        Output:
            :scores: a list containing relative scores of topk predictions.
        '''

        na_score = scores[-1].item() - 1
        if 0 in topks:
            na_score = scores[np.where(topks == 0)].item()

        scores -= na_score

        return scores

    def to_official(self, preds: list, features: list, scores: list = [], topks: list = []):
        '''
        Convert the predictions to official format for evaluating.
        Input:
            :preds: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
            :features: list of features within each document. Identical to the lists obtained from pre-processing.
            :scores: list of scores of topk relation labels for each entity pair.
            :topks: list of topk relation labels for each entity pair.
        Output:
            :official_res: official results used for evaluation.
            :res: topk results to be dumped into file, which can be further used during fushion.
        '''

        h_idx, t_idx, title, sents = [], [], [], []

        for f in features:
            if "entity_map" in f:
                hts = [[f["entity_map"][ht[0]], f["entity_map"][ht[1]]] for ht in f["hts"]]
            else:
                hts = f["hts"]

            h_idx += [ht[0] for ht in hts]
            t_idx += [ht[1] for ht in hts]
            title += [f["title"] for ht in hts]

        official_res = []
        res = []

        for i in range(preds.shape[0]):  # for each entity pair
            if scores.size != 0:
                score = self.extract_relative_score(scores[i], topks[i])
                pred = topks[i]
            else:
                pred = preds[i]
                pred = np.nonzero(pred)[0].tolist()

            for p in pred:  # for each predicted relation label (topk)
                curr_result = {
                    'title': title[i],
                    'h_idx': h_idx[i],
                    't_idx': t_idx[i],
                    'r': self.id2rel[p],
                }
                if scores.size != 0:
                    curr_result["score"] = score[np.where(topks[i] == p)].item()
                if p != 0 and p in np.nonzero(preds[i])[0].tolist():
                    official_res.append(curr_result)
                res.append(curr_result)

        return official_res, res

    def gen_train_facts(self, data_file_name, truth_dir):

        fact_file_name = data_file_name[data_file_name.find("train_"):]
        fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

        if os.path.exists(fact_file_name):
            fact_in_train = set([])
            triples = json.load(open(fact_file_name))
            for x in triples:
                fact_in_train.add(tuple(x))
            return fact_in_train

        fact_in_train = set([])
        ori_data = json.load(open(data_file_name))
        for data in ori_data:
            vertexSet = data['vertexSet']
            for label in data['labels']:
                rel = label['r']
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

        json.dump(list(fact_in_train), open(fact_file_name, "w"))

        return fact_in_train



    def official_evaluate(self, tmp):
        '''
            Adapted from the official evaluation code
        '''
        std = {}
        tot_evidences = 0
        titleset = set([])

        title2vectexSet = {}

        time_relations = defaultdict(int)

        for x in self.truth:
            title = x['title']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            time_indices =set()
            for idx, item in enumerate(vertexSet):
                if item[0]["type"] == "TIME":
                    time_indices.add(idx)


            if 'labels' not in x:  # official test set from DocRED
                continue

            for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                if h_idx in time_indices or t_idx in time_indices:
                    time_relations[r] += 1

                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])

        tot_relations = len(std)
        tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        submission_answer = [tmp[0]]

        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i - 1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                submission_answer.append(tmp[i])

        correct_re = 0
        correct_evidence = 0
        pred_evi = 0

        macro_tp = defaultdict(int)
        macro_fp = defaultdict(int)
        macro_fn = defaultdict(int)

        correct_in_train_annotated = 0
        correct_in_train_distant = 0
        titleset2 = set([])
        submission_answers= set()
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if 'evidence' in x:  # and (title, h_idx, t_idx) in std:
                evi = set(x['evidence'])
            else:
                evi = set([])
            pred_evi += len(evi)
            submission_answers.add((title, r, h_idx, t_idx))
            if (title, r, h_idx, t_idx) in std:
                correct_re += 1
                macro_tp[r] += 1
                stdevi = std[(title, r, h_idx, t_idx)]
                correct_evidence += len(stdevi & evi)
                in_train_annotated = in_train_distant = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in self.fact_in_train_annotated:
                            in_train_annotated = True
                        if (n1['name'], n2['name'], r) in self.fact_in_train_distant:
                            in_train_distant = True

                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1
            else:
                macro_fp[r] += 1



        for key in std:
            if key not in submission_answers:
                macro_fn[key[1]] += 1

        macro_p = {
            k: macro_tp[k] / (macro_tp[k] + macro_fp[k] + 1e-5) for k in macro_tp
        }
        macro_r = {
            k: macro_tp[k] / (macro_tp[k] + macro_fn[k] + 1e-5) for k in macro_tp
        }
        macro_f1 = {
            k: 2 * macro_p[k] * macro_r[k] / (macro_p[k] + macro_r[k] + 1e-5) for k in macro_p
        }

        macro_p = np.mean(list(macro_p.values()))
        macro_r = np.mean(list(macro_r.values()))
        macro_f1 = np.mean(list(macro_f1.values()))

        re_p = 1.0 * correct_re / len(submission_answer)
        re_r = 1.0 * correct_re / tot_relations if tot_relations != 0 else 0
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
        evi_r = 1.0 * correct_evidence / tot_evidences if tot_evidences > 0 else 0

        re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                    len(submission_answer) - correct_in_train_annotated + 1e-5)
        re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
                    len(submission_answer) - correct_in_train_distant + 1e-5)

        if re_p_ignore_train_annotated + re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (
                        re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train + re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

        return [re_p, re_r, re_f1], \
            [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated], \
            [re_p_ignore_train, re_r, re_f1_ignore_train], \
            [macro_p, macro_r, macro_f1]
