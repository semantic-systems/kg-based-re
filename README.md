Install the ULTRA model from the following repository: [https://github.com/DeepGraphLearning/ULTRA](https://github.com/DeepGraphLearning/ULTRA).
It is the prerequisite for all further experiments.

#  Zero-shot setting

## Steps to reproduce

1. Unpack the following files:
    * [wiki.tar.gz](wiki.tar.gz)
    * [fewrel.tar.gz](fewrel.tar.gz)
    * [rel_id_title_description_cleaned.jsonl.gz](rel_id_title_description_cleaned.jsonl.gz)
    * [graph_wiki.gt.gz](graph.gt.gz)
    * [graph_fewrel.gt.gz](graph_fewrel.gt.gz)
2. Run the following command to train the model for each dataset:
    ```
    python3 src/train.py ...
    ```
   
3. Evaluate on each seed of each dataset by using the following command:
    ```
    python3 src/evaluate.py ...
    ```
   

## Functions
Arguments for `train.py`:

| Argument                  | Type      | Default Value     | Description                                                                                                        |
|---------------------------|-----------|-------------------|--------------------------------------------------------------------------------------------------------------------|
| --dataset_name            | str       | "fewrel/unseen_5" | Specifies the name of the dataset. This executes the training for all seeds as specified by the --seeds parameter. |
| --model_type              | str       | "bert-base-cased" | Specifies the type of model to be used.                                                                            |
| --batch_size              | int       | 24                | Sets the batch size for training.                                                                                  |
| --num_workers             | int       | 2                 | Number of worker processes for data loading.                                                                       |
| --accumulate_grad_batches | int       | 2                 | Accumulates gradients over a specified number of batches.                                                          |
| --lr                      | float     | 5e-5              | Learning rate for optimization.                                                                                    |
| --seeds                   | int, List | [0, 1, 2, 3, 4]   | List of seeds of the dataset to train on.                                                                          |
| --include_descriptions    | bool      | False             | Includes descriptions in the textual representation if this flag is present.                                       |
| --deactivate_graph        | bool      | False             | Deactivate the graph component.                                                                                    |
| --only_graph              | bool      | False             | Only use the graph component.                                                                                      |
| --use_filtered_meta_graph        | bool      | False             | Use filtered graph.                                                                                                |

Arguments for `evaluate.py`:

| Argument                              | Type       | Default Value                  | Required               | Description                                                                  |
|---------------------------------------|------------|--------------------------------|------------------------|------------------------------------------------------------------------------|
| --model_checkpoint                    | str        | -                              | Yes                    | Specifies the path to the model checkpoint.                                  |
| --dataset_name                        | str        | "fewrel/unseen_5_seed_0"         | No                     | Specifies the name of the dataset with the corresponding seed.               |
| --model_type                          | str        | "bert-base-cased"      | No                     | Specifies the type of model to be used.                                      |
| --batch_size                          | int        | 24                             | No                     | Sets the batch size for training.                                            |
| --num_workers                         | int        | 2                              | No                     | Number of worker processes for data loading.                                 |
| --accumulate_grad_batches             | int        | 1                              | No                     | Accumulates gradients over a specified number of batches.                    |
| --other_properties                    | int        | 5                              | No                     | Specifies the value for some other properties.                               |
| --include_descriptions                | bool | False                          | No                     | Includes descriptions in the textual representation if this flag is present. |
| --deactivate_graph        | bool       | False             | No                     | Deactivate the graph component.                                              |
| --only_graph              | bool       | False             | No                     | Only use the graph component.                                                |
| --use_filtered_meta_graph        | bool       | False             | No|  Use filtered graph.                                                         |





#  Supervised setting

DISCLAIMER: Again, we were not able to upload all the auxiliary files in an anonymized way due to their size (especially the used graphs). 
We will upload them for the published paper.

## Steps to reproduce

1. Unpack the following files:
    * [dwie.tar.gz](dwie.tar.gz)
    * [redocred.tar.gz](redocred.tar.gz)
    * [biorel.tar.gz](biorel.tar.gz)
    * [graph_dwie.gt.gz](graph_dwie.gt.gz)
    * [graph_docred.gt.gz](graph_docred.gt.gz)
    * [graph_biorel.gt.gz](graph_biorel.gt.gz)
2. Run the following command to train the model for each dataset:
    ```
    python3 src/train.py ...
    ```
   
3. Evaluate checkpoint for each dataset by using the following command:
    ```
    python3 src/evaluate.py ...
    ```
   

## Functions
Arguments for `train.py`:


Here is a table similar to the one you provided, which describes the parameters for your method:

| Argument                  | Type          | Default Value | Description                                                       |
|---------------------------|---------------|---------------|-------------------------------------------------------------------|
| --model_checkpoint        | str           | None          | Specifies the path to a pre-trained model checkpoint.             |
| --batch_size              | int           | 4             | Sets the batch size for training and evaluation.                  |
| --num_rules               | int           | 1             | Number of rules to be used in the model.                          |
| --num_hops                | int           | 4             | Number of hops for the reasoning process.                         |
| --max_length              | int           | 1024          | Maximum sequence length for input examples.                       |
| --num_epochs              | int           | 30            | Number of training epochs.                                        |
| --gradient_accumulation_steps | int       | 1             | Number of steps to accumulate gradients before updating.          |
| --graph_only              | bool    | False         | Use only the graph component if this flag is present.             |
| --use_hinge_abl           | bool    | False         | Uses hinge loss for abductive reasoning if this flag is present.  |
| --use_at_loss             | bool    | False         | Activates the use of ATLoss if this flag is present.              |
| --lr_encoder              | float         | 3e-5          | Learning rate for the encoder component.                          |
| --lr_classifier           | float         | 1e-4          | Learning rate for the classifier component.                       |
| --random_dropout          | float         | 0.2           | Dropout rate applied randomly to the model layers during training. |
| --deactivate_graph        | bool    | False         | Deactivates the graph component if this flag is present.          |
| --short_cut               | bool    | False         | Enables shortcut connectivity if this flag is present.            |
| --use_biorel              | bool    | False         | Utilizes BioRel dataset if this flag is present.                  |
| --use_dwie                | bool    | False         | Utilizes DWIE dataset if this is specified.                       |
| --use_docred              | bool    | False         | Utilizes DocRED dataset if this flag is present.                  |
| --remove_direct_links     | bool    | False         | Removes direct links in the graph or data if this flag is present. |
| --graph_dim               | int           | 64            | Dimensionality of graph embeddings.                               |
| --seed                    | int           | 42            | Seed for random number generation to ensure reproducibility.      |
| --post_prediction         | bool    | False         | Activates post-prediction processing if this flag is present.     |
Arguments for `evaluate.py`:

| Argument                     | Type          | Default Value | Description                                                                                      |
|------------------------------|---------------|---------------|--------------------------------------------------------------------------------------------------|
| model_checkpoint             | str           | None          | Specifies the path to a pre-trained model checkpoint. This is a required positional argument.    |
| --batch_size                 | int           | 4             | Sets the batch size for training and evaluation.                                                 |
| --num_rules                  | int           | 1             | Number of rules to be used in the model.                                                         |
| --num_hops                   | int           | 4             | Number of hops for the reasoning process.                                                        |
| --max_length                 | int           | 1024          | Maximum sequence length for input examples.                                                      |
| --gradient_accumulation_steps| int           | 1             | Number of steps to accumulate gradients before updating.                                         |
| --graph_only                 | bool    | False         | Use only the graph component if this flag is present.                                            |
| --deactivate_graph           | bool    | False         | Deactivates the graph component if this flag is present.                                         |
| --short_cut                  | bool    | False         | Enables shortcut connectivity if this flag is present.                                           |
| --use_biorel                 | bool    | False         | Utilizes BioRel dataset if this flag is present.                                                 |
| --use_dwie                   | bool    | False         | Utilizes DWIE dataset if specified.                                                              |
| --remove_direct_links        | bool    | False         | Removes direct links in the graph or data if this flag is present.                               |
| --separated                  | bool    | False         | Processes inputs as separated components if this flag is present.                                |
| --graph_dim                  | int           | 64            | Dimensionality of graph embeddings.                                                              |
| --post_prediction            | bool    | False         | Activates post-prediction processing if this flag is present.                                    |


