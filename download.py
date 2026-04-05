import pandas as pd


def download_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Description:
        Download the required dataset from Hugging Face.

        The addresses are hard-coded for this particular assignment.

        For future assignments, we have prepared a more elaborate implementation
          which also does local caching.

    Args:
        None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: the training split (120k samples)
            and the test split (7k samples).
    """
    train_120k_samples: pd.DataFrame = pd.read_json("hf://datasets/sh0416/ag_news/train.jsonl",
                                                    lines = True)

    test_7k_samples: pd.DataFrame = pd.read_json("hf://datasets/sh0416/ag_news/test.jsonl",
                                                 lines = True)

    return (train_120k_samples, test_7k_samples)
