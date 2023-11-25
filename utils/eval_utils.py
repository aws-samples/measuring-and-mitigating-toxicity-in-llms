#!/usr/bin/env python3

import evaluate
import torch


def _evaluate_toxicity(text=[], aggregation_method=None):
    """
    Evaluate toxicity of a list of text strings using a pre-trained model.

    Args:
        text (list): List of text strings to evaluate for toxicity.
        aggregation_method (str): Method for aggregating toxicity scores.

    Returns:
        float: Toxicity score based on the specified aggregation method.
    """
    # specify model name
    toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"

    toxicity_evaluator = evaluate.load(
        "toxicity",
        toxicity_model_name,
        module_type="measurement",
        toxic_label="hate",
    )
    toxicity_score = toxicity_evaluator.compute(
        predictions=text, aggregation=aggregation_method
    )

    if aggregation_method == None:
        toxicity_measure = "toxicity"
    elif aggregation_method == "maximum":
        toxicity_measure = "max_toxicity"
    elif aggregation_method == "ratio":
        toxicity_measure = "toxicity_ratio"
    else:
        toxicity_measure = "toxicity"

    return toxicity_score[toxicity_measure]


def _add_toxicty_column(data, column_to_evaluate="dialogue"):
    """
    Add a toxicity score column to a dataset based on the specified column.

    Args:
        data (Dataset): The input dataset.
        column_to_evaluate (str): The name of the column in the dataset to evaluate for toxicity.

    Returns:
        data (Dataset): The input dataset with an additional "toxicity_score" column.
    """

    def _add_toxicity_score(sample):
        """
        Add toxicity score to a single sample in the dataset.

        Args:
            sample (dict): A dictionary representing a single sample in the dataset.

        Returns:
            dict: The sample dictionary with an added "tox_score" key containing the toxicity score.
        """
        # calculate toxicity score
        sample["tox_score"] = _evaluate_toxicity(sample[column_to_evaluate])
        return sample

    def _group_batch(batch):
        """
        Group datapoints into batches for faster toxicity evaluation.

        Args:
            batch (dict): A dictionary containing batched data.

        Returns:
            dict: The batched data.
        """
        return {k: [v] for k, v in batch.items()}

    # set batch size
    BATCH_SIZE = 6

    # batch data
    batched_data = data.map(
        _group_batch, batched=True, batch_size=BATCH_SIZE, drop_last_batch=False
    )

    # calculate toxicity scores in batches
    batched_data = batched_data.map(_add_toxicity_score)

    # create empty list to flatted
    toxicities = []
    for b in batched_data["tox_score"]:
        toxicities.append(b)

    # flatten batches
    tox_scores = [item for sublist in toxicities for item in sublist]

    # add new column
    data = data.add_column("toxicity_score", tox_scores)
    return data
