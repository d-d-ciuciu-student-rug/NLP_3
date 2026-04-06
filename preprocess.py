from collections import Counter
from typing import Callable
import re

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset



def concatenate_title_and_description_into_merged(data: pd.DataFrame,
                                                  title_alias: str = "title",
                                                  description_alias: str = "description",
                                                  merged_alias: str = "input"
                                                  ) -> None:
    """
    Description:

        Merge the "title" and "description" vectors into a single "input" vector.
         Missing values (, if any,) are replaced with empty strings.

    Args:
        data (pd.DataFrame): Pandas DataFrame containing columns "title" and "description".

    Returns:
        None -> side-effect mutates ::data.
    """

    data[merged_alias] = (data[title_alias].fillna("") +
                          " " +
                          data[description_alias].fillna("")
                         ).str.strip()


def align_labels_to_zero_index_and_rename(data: pd.DataFrame,
                                          label_alias: str = "label",
                                          label_new_alias: str = "output",
                                          sentinel_value_for_missing: int = -1
                                          ) -> dict[int, int]:
    """
    Description:

        This function is idempotent. It will align the labels to be zero-indexed.
         Even if called multiple times, it will not invalidate the labels.
    """

    # First, make sure there are no missing values.
    data[label_alias] = data[label_alias].fillna(sentinel_value_for_missing)

    # Then extract the unique label values, and sort them.
    unique_labels: list[int] = sorted(data[label_alias].unique().tolist())

    # Then generate the mapping (original index -> zero-based final index).
    #  We simply make use of enumerate() starting its index count at 0.
    mapping: dict[int, int] = {original: zero_based
                               for (zero_based, original) in enumerate(unique_labels)}

    # And finally, apply the ::mapping dictionary onto the `pd.DataFrame`.
    data[label_alias] = data[label_alias].map(mapping)

    data[label_new_alias] = data[label_alias]

    data.drop(columns = [label_alias],
              inplace = True)

    return mapping


def split_training_data(data: pd.DataFrame,
                        stratify_column: str,
                        seed: int
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Description:

        Download the dataset from Hugging Face.

    Args:
        data (pd.DataFrame): Pandas DataFrame with the training dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: the split into training,
            development/validation and testing.
    """

    # Note: besides the seed, we also need equal representation of all labels.
    #  For that purpose, we need to use the "stratify" parameter, based on the
    #  column of labels. 
    train_df, temp_df = train_test_split(data, test_size = 0.2,
                                               random_state = seed,
                                               shuffle = True,
                                               stratify = data[stratify_column])

    dev_df, test_df = train_test_split(temp_df, test_size = 0.5,
                                                random_state = seed,
                                                shuffle = True,
                                                stratify = temp_df[stratify_column])

    return (train_df, dev_df, test_df)


def tokenize(text: str,
             rules: tuple[bool, list[tuple[str, str]]],
             padding_token: tuple[str, int],
             unknown_token: tuple[str, int]
             ) -> list[str]:
    """
    Description:

        This function fits well with our ideal of easily automating
         various combinations of tokenization, training and model-tuning.

        1. Tokenization first decides whether to lower-case the text or not.
        2. The second step is to apply a list of regex-based substitutions.
        3. The third step is to sanitize the ::text such as not to contain
            the ::padding_token or ::unknown_token, avoiding collision.
        4. The final step is to split the string by space, into "words"/tokens.
    """

    # Whether to .lower() ::text or not.
    to_lower: bool = rules[0]
    if to_lower:
        text = text.lower()

    # Apply (possibly multiple) regex substitution rules.
    regexes: list[tuple[str, str]] = rules[1]
    for regex in regexes:
        pattern: str = regex[0]
        substitution: str = regex[1]

        text = re.sub(pattern, substitution, text)

    # Sanitization.
    padding_token_string: str = padding_token[0]
    text = text.replace(padding_token_string, "")

    unknown_token_string: str = unknown_token[0]
    text = text.replace(unknown_token_string, "")

    # Final split.
    return text.split()


def build_vocabulary(text_vector: pd.Series,
                     rules: tuple[bool, list[tuple[str, str]]],
                     padding_token: tuple[str, int],
                     unknown_token: tuple[str, int],
                     token_filter_condition: Callable[[str, int], bool],
                     maximum_vocabulary_size: int
                     ) -> dict[str, int]:
    """
    Description:

        This is meant to be used only on the training set's "input" vector.

        A vector of strings is used for building a "vocabulary of tokens".
         All tokens outside of this vocabulary will be ignored by the model.

        It's important to note that the vocabulary must have a fixed length,
         therefore we make use of a "padding token" to fill in shorter lists.

        Also, we need to make sure that space is allocated for the "unknown_token"
         as well, although this doesn't get used during training itself (but rather
         during evaluation/inferences, when the model might see new words/tokens).
    """

    # Frequency counter, traverse through the strings vector.

    counter: Counter = Counter()
    for text in text_vector:
        tokens: list[str] = tokenize(text, rules, padding_token, unknown_token)
        counter.update(tokens)

    # Vocabulary for mapping tokens to integer values.
    #  Initialize with the "padding token" and the "unknown token".

    vocabulary: dict[str, int] = {
        padding_token[0]: padding_token[1],
        unknown_token[0]: unknown_token[1],
    }

    assigned_integers: set[int] = {padding_token[1], unknown_token[1]}

    # Traverse all of the encountered tokens, and assign them an integer value.
    #
    #  - make use of the ::token_filter_condition() function pointer, which easily
    #     allows us to automate various filtering conditions;
    #
    #  - ensure that the string:integer assignment does not overwrite "padding token"
    #     or the "unknown token".

    for token, frequency in counter.items():

        # Let there be an upper limit for the vocabulary size.
        if maximum_vocabulary_size <= len(vocabulary):
            break

        # But if it hasn't been reached yet, apply the filter lambda-function.
        if token_filter_condition(token, frequency):

            # Ensure a unique integer is assigned to each token.
            #
            #  - len(vocabulary) acts as an initial guess;
            #
            #  - also, the domain cannot be larger than len(vocabulary), so it's safe
            #     to use it as an upper-bound for the for-loop in case of collision.

            assigned_integer: int = len(vocabulary)

            if assigned_integer in assigned_integers:
                for integer in range(0, len(vocabulary) + 1):
                    if integer not in assigned_integers:
                        assigned_integer = integer
                        break

            vocabulary[token] = assigned_integer
            assigned_integers.add(assigned_integer)

    # It is therefore possible for the vocabulary to be smaller than the target/maximum size.
    return vocabulary


def encode(text_vector: pd.Series,
           rules: tuple[bool, list[tuple[str, str]]],
           padding_token: tuple[str, int],
           unknown_token: tuple[str, int],
           vocabulary: dict[str, int],
           target_input_list_length: int
           ) -> tuple[list[list[int]], list[int]]:
    """
    Description:
        Transform input strings into lists of string tokens (tokenization), and then map the
         tokens onto integer values (encoding), according to a given vocabulary.

        Unrecognized tokens are replaced by ::unknown_token.

        Lists shorter than ::target_input_list_length are padded with ::padding_token,
         and longer lists are simply trimmed to match ::target_input_list_length.
    """

    # For each ::text in ::text_vector, generate a `list[int]` that map its component tokens onto
    #  integer values, according to ::vocabulary.
    encodings: list[list[int]] = []

    # The LSTM shouldn't be looking at the padding tokens, since it doesn't make use of a sliding
    #  window (being a recurrent network).
    #
    # The padding tokens are mostly for the CNN, as it uses a sliding window (actually this depends
    #  on the kernels being used, so it likely uses multiple sliding windows), together with a dot
    #  product. Given that the "padding token"'s index is 0, it actually won't arithmetically affect
    #  the (convolution's) dot-product.
    non_padded_input_lengths: list[int] = []

    padding_token_index: int = padding_token[1]
    unknown_token_index: int = unknown_token[1]

    for text_scalar in text_vector:

        # A cell's value in pd.Series can have multiple types actually, including {`None`, `str`, ..?}.
        text: str = str(text_scalar)

        tokens: list[str] = tokenize(text, rules, padding_token, unknown_token)
        encoding: list[int] = [vocabulary.get(token, unknown_token_index) for token in tokens]

        # Upper-bound on encoding list length.
        if len(encoding) > target_input_list_length:
            encoding = encoding[ : target_input_list_length]
 
        # We truncated :encoding first, because the non-padded length should still be clamped by
        #  ::target_input_list_length, but not be affected by extension with [::padding_token_index].
        non_padded_length: int = len(encoding)

        # And the lower-bound on encoding list length.
        repeats: int = target_input_list_length - len(encoding)
        encoding = encoding + [padding_token_index] * repeats

        encodings.append(encoding)
        non_padded_input_lengths.append(non_padded_length)

    return (encodings, non_padded_input_lengths)
