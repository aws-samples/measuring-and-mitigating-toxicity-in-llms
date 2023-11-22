import evaluate
import torch


def _evaluate_toxicity(text=[], aggregation_method=None):
    """ """
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
    """ """

    def _add_toxicity_score(sample):
        """
        Function to create summaries of the movie dialogue dataset.
        """
        # calculate toxicity score
        sample["tox_score"] = _evaluate_toxicity(sample[column_to_evaluate])
        return sample

    def _group_batch(batch):
        """
        Function to batch datapoints for faster evaluation.
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
