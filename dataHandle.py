# file used to parse and format different possible input data.


import os
from typing import Any
from torch import Tensor

def load_tlgbank_dataset_camembert(dataset_file: str) -> Any:
    print(f"Loading dataset from {dataset_file} for CamemBERTTager architecture...")
    # only words and tags
    data = []
    with open(dataset_file, "r") as f:
        lines = f.readlines()
        for l in lines:
            elements = l.split(" ")
            words_text = []
            words_pos = []
            words_category = []
            for words in elements:
                w = words.split("|")
                words_text.append(w[0].strip())
                words_pos.append(w[1].split("-")[0].split("+")[0].strip())
                words_category.append(w[2].strip())
            data.append((words_text, words_pos, words_category))
    print("Done!")
    return data



def fetch_data_camembert(input_file: str) -> Tensor:
    print("Utilizing CamemBERTTagger. Applying whitespace tokenization.")
    sentences = []
    with open(input_file, "r") as f:
        lines = f.readlines()
        for x in lines:
            sentences.append(x.split())
    return sentences

