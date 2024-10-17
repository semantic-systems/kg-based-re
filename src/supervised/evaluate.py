import argparse
import datetime
import json
import os

import torch
from pytorch_lightning import Callback

from src.supervised.custom_datasets import DocREDDataset, BioRELDataset, DWIEDataset, ReDocREDDataset
from src.supervised.evaluator import Evaluator
from src.supervised.models.re_model import REModel
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from src.supervised.utils.docred.entity_types import get_valid_types_refined

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("model_checkpoint", type=str)
    argparser.add_argument("--batch_size", type=int, default=4)
    argparser.add_argument("--num_rules", type=int, default=1)
    argparser.add_argument("--num_hops", type=int, default=6)
    argparser.add_argument("--use_at_loss", action="store_true")
    argparser.add_argument("--use_distantly", action="store_true")
    argparser.add_argument("--max_length", type=int, default=512)
    argparser.add_argument("--num_epochs", type=int, default=30)
    argparser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    argparser.add_argument("--graph_only", action="store_true")
    argparser.add_argument("--use_hinge_abl", action="store_true")
    argparser.add_argument("--re_weight", type=float, default=0.5)
    argparser.add_argument("--lr_encoder", type=float, default=3e-5)
    argparser.add_argument("--lr_classifier", type=float, default=1e-4)
    argparser.add_argument("--random_dropout", type=float, default=0.2)
    argparser.add_argument("--deactivate_graph", action="store_true")
    argparser.add_argument("--short_cut", action="store_true")
    argparser.add_argument("--use_biorel", action="store_true")
    argparser.add_argument("--use_dwie", action="store_true")
    argparser.add_argument("--remove_direct_links", action="store_true")
    argparser.add_argument("--separated", action="store_true")
    argparser.add_argument("--graph_dim", type=int, default=128)
    argparser.add_argument("--dynamic_freeze", type=int, default=0)
    argparser.add_argument("--post_prediction", action="store_true")


    args = argparser.parse_args()

    name_from_current_time = f"{datetime.datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')}"
    # Change precision to high
    pl.trainer.seed_everything(42)

    valid_qids, _ = get_valid_types_refined()
    torch.set_float32_matmul_precision("high")

    if args.use_biorel:
        relation_dict = BioRELDataset.get_relation_dict()
        prop_idx = json.load(open("data/biorel/relation_index.json"))
        prop_idx = {idx: idx for idx in range(len(prop_idx))}
        evaluator = None
        labels_to_predict = 1
        model_name: str = "dmis-lab/biobert-v1.1"
    elif args.use_dwie:
        relation_dict = DWIEDataset.get_relation_dict()
        prop_idx = {int(x): y for x, y in json.load(open("data/dwie/prop_dict.json")).items()}
        labels_to_predict = 4
        evaluator = None
        model_name: str = "bert-base-uncased"
        # model_name: str = "roberta-large"
    else:
        relation_info = json.load(open("data/docred/rel_info.json"))
        relation_info = sorted(list(relation_info.items()), key=lambda x: x[0])
        relation_dict = {}
        for idx, relation in enumerate(relation_info):
            relation_dict[relation[0]] = idx

        evaluator = Evaluator({v: k for k, v in relation_dict.items()}, "data/docred", "train_revised.json",
                              "test_revised.json")
        prop_idx = {int(x): v for x, v in json.load(open("data/docred/prop_dict_all.json")).items()}
        labels_to_predict = 4
        model_name: str = "roberta-large"
    num_relations = len(relation_dict)
    if args.use_at_loss or args.use_hinge_abl:
        num_relations += 1

    model = REModel.load_from_checkpoint(args.model_checkpoint, num_output_relation=num_relations,
                                         num_classes=len(valid_qids), graph_only=args.graph_only,
                                         at_loss=args.use_at_loss, num_rules=args.num_rules,
                                         num_hops=args.num_hops, re_weight=args.re_weight,
                                         lr_classifier=args.lr_classifier,
                                         use_hinge_abl=args.use_hinge_abl,
                                         cross_relation_weight=args.cross_relation_weight,
                                         evaluator=evaluator, alt_mode=args.alt_mode,
                                         deactivate_graph=args.deactivate_graph,
                                         use_only_types=args.use_only_types, random_dropout=args.random_dropout,
                                         short_cut=args.short_cut, num_relations=len(prop_idx),
                                         labels_to_predict=labels_to_predict,
                                         model_name=model_name,
                                         graph_dim=args.graph_dim, separated=args.separated,
                                         full_document=args.max_length > 1024,
                                         dynamic_freeze=args.dynamic_freeze,
                                            post_prediction=args.post_prediction)

    if args.use_biorel:
        dataset = BioRELDataset(model.tokenizer, batch_size=args.batch_size,
                                add_no_relation=args.use_at_loss or args.use_hinge_abl, max_length=args.max_length,
                                num_workers=8, remove_direct_links=args.remove_direct_links)
    elif args.use_dwie:
        dataset = DWIEDataset(model.tokenizer, relation_dict=relation_dict, batch_size=args.batch_size,
                              add_no_relation=args.use_at_loss or args.use_hinge_abl, max_length=args.max_length,
                              num_workers=8, remove_direct_links=args.remove_direct_links)
    else:
        dataset = ReDocREDDataset(model.tokenizer, relation_dict, batch_size=args.batch_size,
                                add_no_relation=args.use_at_loss or args.use_hinge_abl,
                                use_gold=not args.use_distantly, max_length=args.max_length,
                                remove_direct_links=args.remove_direct_links,
                                num_workers=4)



    directory = f'checkpoints/{name_from_current_time}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Dump config args to file for later reference
    with open(f'{directory}/config.json', 'w') as f:
        json.dump(vars(args), f)

    wandb_logger = pl_loggers.WandbLogger(project='GraphRE')
    dict_args = vars(args)
    dict_args["name"] = name_from_current_time
    wandb_logger.log_hyperparams(dict_args)

    num_devices = torch.cuda.device_count()

    trainer = pl.Trainer(max_epochs=args.num_epochs, devices=num_devices if num_devices > 0 else "auto",
                         logger=wandb_logger,
                         strategy="ddp_find_unused_parameters_true" if num_devices > 1 else "auto",
                         accumulate_grad_batches=args.gradient_accumulation_steps,
                         gradient_clip_val=1.0)

    dataset.setup()
    trainer.validate(model, dataset.test_dataloader())
