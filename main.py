# Callable file reading a discourse provided in french via input file
# outputs a file with the same discourse supertagged with CCG
# Tags are generated by a trained model which is either trained or to-be-trained

import pickle
import sys
import os
import argparse
import time
import torch
from transformers import logging
from torch.nn.utils.rnn import pad_sequence
from model import CamemBERTTagger
from dataHandle import fetch_data_camembert
from utils import START_TAG, END_TAG
from typing import List, Tuple, Dict
#from category import Node

device = "cuda" if torch.cuda.is_available() else "cpu"

# we don't use CamemBERT head layer. Following line will mute this warning
logging.set_verbosity_error()


def decode_outputs(
    model: CamemBERTTagger,
    words: List[List[int]],
    word_ids: List[List[int]],
    outputs: List[List[int]],
    id2tags: Dict[int, str],
) -> Tuple[List[str], List[List[str]]]:
    sentences = model.decode(words)
    tags = []
    prev = None
    for s in range(len(word_ids)):
        sentence_tags = []
        pos = 1
        for id in word_ids[s][1:]:
            if id is not None and id != prev:
                prev = id
                sentence_tags.append(id2tags[outputs[s][pos]])
            pos += 1
        tags.append(sentence_tags)
    return (sentences, tags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="French CCG supertagger [main].")
    parser.add_argument(
        "-m",
        "--model",
        default="",
        type=str,
        help="Trained model weights dictionary file",
    )
    parser.add_argument(
        "-d",
        "--data",
        default="",
        type=str,
        help="Trained model data file (.pickle file)",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="",
        type=str,
        help="Discourse formulated in French natural language.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tagged_discourse.txt",
        type=str,
        help="tagged discourse file (default: tagged_discourse.txt",
    )
    parser.add_argument(
        "--max-sequence-length",
        default=120,
        type=int,
        help="Max sequence length (default: 120)",
    )
    parameters = parser.parse_args()
    model_file: str = parameters.model
    data_file: str = parameters.data
    input_file: str = parameters.input
    output_file: str = parameters.output
    max_sequence_length: int = parameters.max_sequence_length

    if not os.path.exists(model_file) or not os.path.isfile(model_file):
        print(
            "Model weights dictionary file invalid. It can be generated by calling the file train.py."
        )
        print("Exiting program.")
        sys.exit(1)
    if not os.path.exists(data_file) or not os.path.isfile(data_file):
        print(
            "Model data file invalid. It can be generated by calling the file train.py."
        )
        print("Exiting program.")
        sys.exit(1)
    if not os.path.exists(input_file) or not os.path.isfile(input_file):
        print("Input file invalid.")
        print("Exiting program.")
        sys.exit(1)

    start = time.time()
    print("**** Model loading... ****")
    with open(data_file, "rb") as f:
        model_data = pickle.load(f)
    model = CamemBERTTagger(
        dict2id=model_data["tags2id"],
        pos2id=model_data["pos2id"],
        hidden_dim=model_data["hidden_size"],
        batch_size=model_data["batch_size"],
        latent_space=model_data["latent_dim"],
    )
    model.load_state_dict(torch.load(model_file))
    model.to(device)
    model.train(False)
    print("**** Done. ****")

    with torch.no_grad():
        print("**** Starting data preprocessing... ****")
        data = fetch_data_camembert(input_file)
        tokenized_data = model.tokenize_inputs(
            data, max_sequence_length=model_data["max_sequence_length"]
        )
        inputs_len = len(tokenized_data)
        print("**** Done. ****")
        # fill to reach batch size
        while len(tokenized_data) < model_data["batch_size"]:
            tokenized_data.append(
                (
                    {
                        "input_ids": [5, 6],
                        "attention_mask": [1, 1],
                        "words_id": [None, None],
                    }
                )
            )

        masks_batch = [w["attention_mask"] for w in tokenized_data]
        words_batch = [w["input_ids"] for w in tokenized_data]
        words_id = [w["words_id"] for w in tokenized_data]

        words_batch = pad_sequence(
            [torch.tensor(w, device=device) for w in words_batch], batch_first=True
        )
        masks_batch = pad_sequence(
            [torch.tensor(m, device=device) for m in masks_batch], batch_first=True
        )

        predictions = model(words_batch, masks_batch, False, k=1)
        end = time.time()
        print(f"Predicted {len(words_batch)} sentences in {end-start}s.")
        id2tags = model_data["id2tags"]
        #predictions = predictions
        sentences, tags = decode_outputs(
            model, words_batch, words_id, predictions, id2tags
        )
        print("----------------")
        for s, t in zip(sentences, tags):
            for word, tag in zip(s.split(), t):
                print(f"{word:<25}\t\t{tag:<50}")
            print("----------------")
