from __future__ import annotations

import gc
import json
import pickle
from typing import Any, Union


import torch


import generic


def parse_intents(data: dict) -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """
    Parses passed json data.

    Parameters
    -----------
        data (dict): Json data.

    Returns
    --------
    tuple[
        words (list[str]): List of words in patterns from `intent.json`.
        tags (list[str]): List of tags from `intent.json`.
        docs_X (list[list[str]]): List of list of all the different tokenized words in a pattern.
        docs_y (list[str]): List of tags where each element corresponds to a pattern in `docs_X`.
    ]
    """
    import nltk

    words: list[str] = []  # List of words in patterns from `intent.json`.
    tags: list[str] = []  # List of tags from `intent.json`.
    docs_X: list[
        list[str]
    ] = []  # List of list of all the different tokenized words in a pattern.
    docs_y: list[
        str
    ] = []  # List of tags where each element corresponds to a pattern in `docs_X`.

    for intent in data["intents"]:
        if intent["tag"] not in tags:
            tags.append(intent["tag"])

        for pattern in intent["patterns"]:
            docs_y.append(intent["tag"])
            # Gets all the words in a sentence, in this case `pattern`.
            tokenized_words: list[str] = nltk.word_tokenize(pattern)
            words.extend(tokenized_words)
            docs_X.append(tokenized_words)
    tags.sort()

    return words, tags, docs_X, docs_y


def get_training_data(
    words: list[str], tags: list[str], docs_X: list[list[str]], docs_y: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates training and output list.

    Parameters
    -----------
        words (list[str]): List of all words from all the patterns.
        tags (list[str]): List of tags.
        docs_X (list[list[str]]): List of list of tokenized words from each pattern.
        docs_y (list[list[str]]): List of tags where each element corresponds to a pattern in `docs_X`.

    Returns
    --------
    tuple[
        training (torch.Tensor): Rank 2 Tensor of zeroes and ones where each `1` corresponds to a word in `words` that is also in a set of stemmed tokenized words from a pattern.
        output (torch.Tensor) Rank 1 Tensor of one hot vectors where each `1` corresponds to the pattern's tag.
    ]
    """
    dummy_output_row: list[int] = [0] * len(tags)
    stemmed_patterns: list[list[str]] = [generic.stem_words(doc_X) for doc_X in docs_X]

    training: list[list[int]] = []
    output: list[list[int]] = []

    for index, stemmed_pattern in enumerate(stemmed_patterns):

        bag = [1 if word in stemmed_pattern else 0 for word in words]

        output_row = dummy_output_row.copy()
        output_row[tags.index(docs_y[index])] = 1

        training.append(bag)
        output.append(output_row)

    return (
        torch.tensor(training, dtype=torch.float),
        torch.tensor(output, dtype=torch.float),
    )


def train_and_save_or_load_model(
    training: torch.Tensor,
    output: torch.Tensor,
    *,
    filepath: str,
    train: bool = False,
    n_epoch: int = 8,
    batch_size: int = 32,
    shuffle: bool = False,
) -> Union[None, torch.nn.Module]:
    """
    Depending on the arguments, either trains or loads a model.
    Args:
        training (torch.Tensor): Rank 2 Tensor of zeroes and ones where each `1` corresponds to a word in `words` that is also in a set of stemmed tokenized words from a pattern.
        output (torch.Tensor): Rank 1 Tensor of one hot vectors where each `1` corresponds to the pattern's tag.
        filepath (str): Path to save or load model to or from.
        train (bool, optional): To train or not to train. Defaults to False.
        n_epoch (int, optional): Maximum number of epochs. Defaults to 8.
        batch_size (int, optional): Size of batch. Defaults to 32.
        shuffle (bool, optional): To shuffle the dataset or not to shuffle to dataset.

    Returns:
        net (torch.nn.Module)
    """
    import torch

    class Net(torch.nn.Module):
        def __init__(self: Net):
            super().__init__()
            self.fc1 = torch.nn.Linear(training.shape[1], 8)
            self.fc2 = torch.nn.Linear(8, 8)
            self.fc3 = torch.nn.Linear(8, 8)
            self.fc4 = torch.nn.Linear(8, output.shape[1])

        def forward(self: Net, X):
            X = torch.nn.functional.relu(self.fc1(X))
            X = torch.nn.functional.relu(self.fc2(X))
            X = torch.nn.functional.relu(self.fc3(X))
            X = self.fc4(X)

            return torch.nn.functional.softmax(X, dim=1)

        def predict(self: Net, *data: str):
            self.eval()
            for X in data:
                output = self(X)
                return output

    net = Net()

    if train:
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        dataset: list[tuple[torch.Tensor, torch.Tensor]] = list(zip(training, output))
        trainset: torch.utils.data.dataloader.DataLoader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        loss_function = torch.nn.MSELoss()

        for epoch in range(n_epoch):
            for data in trainset:
                X, y = data
                net.zero_grad()
                output = net(X)
                loss = loss_function(output, y)
                loss.backward()
                optimizer.step()
                print(loss.item(), end="\r")
            print(loss)

        torch.save(net.state_dict(), filepath)

        return None

    net.load_state_dict(torch.load(filepath))

    return net


def create_and_train(
    *,
    pickle_filepath: str,
    intents_model_filepath: str,
    intents_filepath: str,
    n_epoch: int = 8,
    batch_size: int = 32,
    shuffle: bool = True,
) -> None:
    """
    Convenient function to create data and train model.
    Args:
        pickle_filepath (str): Path to data pickle file.
        model_filepath (str): Path to saved model file.
        intents_filepath (str, optional): Path to intents.json file.
        n_epoch (int, optional): Maximum number of epochs. Defaults to 8.
        batch_size (int, optional): Size of batch. Defaults to 32.
        shuffle (bool, optional): To shuffle or not to shuffle. Defaults to True.
    """
    data = generic.read_intents_file(intents_filepath)
    words, tags, docs_X, docs_y = parse_intents(data)
    words = generic.stem_words(words)
    words = sorted(
        list(set(word for word in words if word not in "?.!,"))
    )  # Removing duplicates and some punctuation marks, and sorting.

    training, output = get_training_data(words, tags, docs_X, docs_y)

    with open(pickle_filepath, "wb") as f:
        pickle.dump((words, tags, training, output), f)
        
    del data, words, tags, docs_X, docs_y
    gc.collect()

    train_and_save_or_load_model(
        training,
        output,
        filepath=intents_model_filepath,
        train=True,
        n_epoch=n_epoch,
        batch_size=batch_size,
        shuffle=shuffle,
    )
