from copy import deepcopy
import torch
from torch import Tensor
from typing import Any, Dict, List, Tuple
from collections import Counter
import random
import numpy as np

## useful tags added to data
START_TAG = "<START>"
END_TAG = "<END>"
PAD_TAG = "<pad>"
UNK_TAG = "<unk>"

# task label
TASK_TAG = "tag"
TASK_POS = "pos"

# useful type alias
Dataset = List[List[Tuple[str, str, str, str, str]]]

# sorts by frequency a dictionary
def sort_dict(dict: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    # lambda key: sorts by frequency then by alphanumerical order
    sorted_items = sorted(dict.items(), key=lambda x: (-x[1], x[0]))
    item2id = {item[0]: id for id, item in enumerate(sorted_items)}
    id2item = {id: item[0] for id, item in enumerate(sorted_items)}
    id2freq = {id: item[1] for id, item in enumerate(sorted_items)}
    return item2id, id2item, id2freq


# creates item2id and id2item, sorted by frequency
def map_dictionary(dict: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    # add pad and unk so that they are the 2 first values
    dict[PAD_TAG] = 100001
    dict[UNK_TAG] = 100000
    return sort_dict(dict)


# maps tags in the same fashion as map_dictionary, but adds start and end tags for computation
def map_tags(dict: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    dict[START_TAG] = -1
    dict[END_TAG] = -2
    dict[PAD_TAG] = -3
    return sort_dict(dict)


def build_vocab_camembert(
    dataset: List[Tuple[List[str], List[str]]]
) -> List[Tuple[Dict[Any, Any]]]:
    print("Building vocabulary...")
    postags, tags = [], []
    for _, p, t in dataset:
        for pos, tag in zip(p, t):
            tags.append(tag)
            postags.append(pos)
    tags = dict(Counter(tags))
    postags = dict(Counter(postags))
    tags2id, id2tags, id2freqtags = map_tags(tags)
    pos2id, id2pos, id2freqpos = map_tags(postags)
    return [(tags2id, id2tags, id2freqtags), (pos2id, id2pos, id2freqpos)]


def extract_tag_stat(
    id2frq: Dict[int, int], test: Any, end_id: int
) -> Tuple[Dict[int, int]]:
    for _, _, sentence_tag in test:
        for tag in sentence_tag[1:]:  # start tag at pos 0
            if tag == end_id:
                break
            elif tag < end_id:  # pad_tag > end_id
                id2frq[tag] -= 1
    common = {t: f for t, f in id2frq.items() if f >= 100}
    uncommon = {t: f for t, f in id2frq.items() if f >= 10 and f < 100}
    rare = {t: f for t, f in id2frq.items() if f >= 1 and f < 10}
    unseen = {t: f for t, f in id2frq.items() if f == 0}
    return (common, uncommon, rare, unseen)


# build vocabulary for each feature (word, lemma, postag, deprel) and tlg outputs from dataset
def build_vocab(
    dataset: List[
        List[
            Tuple[
                str,
                str,
                str,
                str,
                str,
            ]
        ]
    ]
) -> List[Tuple[Dict[str, int], Dict[int, str]]]:
    print("Building vocabulary...")
    words, lemmas, postags, deprels, ccgs = [], [], [], [], []
    for sentence in dataset:
        for word, lemma, postag, deprel, ccg in sentence:
            words.append(word.lower())
            lemmas.append(lemma)
            postags.append(postag)
            deprels.append(deprel)
            ccgs.append(ccg)
    # dict(Counter(x)) will create a dictionary of each feature with the number of times it appears in the dataset
    words = dict(Counter(words))
    lemmas = dict(Counter(lemmas))
    postags = dict(Counter(postags))
    deprels = dict(Counter(deprels))
    ccgs = dict(Counter(ccgs))
    words2id, id2words = map_dictionary(words)
    lemmas2id, id2lemmas = map_dictionary(lemmas)
    postags2id, id2postags = map_dictionary(postags)
    deprels2id, id2deprels = map_dictionary(deprels)
    # we need START and END tags for ccg tags
    ccgs2id, id2ccgs = map_tags(ccgs)
    print("Done!")
    return [
        (words2id, id2words),
        (lemmas2id, id2lemmas),
        (postags2id, id2postags),
        (deprels2id, id2deprels),
        (ccgs2id, id2ccgs),
    ]


# adds words from pre_words_embedding to vocabulary
def enhance_vocabulary(
    words2id: Dict[str, int],
    id2words: Dict[int, str],
    pre_words_embedding: Dict[str, Tensor],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    print("Enhancing vocabulary using pre_trained embedding...")
    for w in pre_words_embedding.keys():
        word = w.lower()
        if word not in words2id.keys():
            id = len(words2id)
            words2id[word] = id
            id2words[id] = word
    print("Done!")
    return (words2id, id2words)


# splits randomly into 3 subsets (train, test and validation) a given dataset
def shuffle_and_split(
    dataset: Dataset, vratio: float, tratio: float, seed: int
) -> Dataset:
    random.seed(seed)
    random.shuffle(dataset)
    test_index = int(len(dataset) * tratio)
    test_dataset = dataset[:test_index]
    train_dataset = dataset[test_index:]
    random.shuffle(train_dataset)
    validation_index = int(len(train_dataset) * vratio)
    validation_dataset = train_dataset[:validation_index]
    train_dataset = train_dataset[validation_index:]
    return (train_dataset, validation_dataset, test_dataset)


# creates proper data from a given dataset, using the vocabulary.
# transforms each feature into its proper id in order to feed it later in the neural model
# returns a list (dataset) of dictionaries (sentences) of lists (words)
def get_data_from_dataset(
    dataset: Dataset,
    words2id: Dict[str, int],
    lemmas2id: Dict[str, int],
    postags2id: Dict[str, int],
    deprels2id: Dict[str, int],
    ccgs2id: Dict[str, int],
) -> List[Dict[str, List[str]]]:
    print("Creating data from dataset...")
    data = []
    tagset = []
    for sentence in dataset:
        sentence_text = []
        words_id = []
        lemmas_id = []
        postags_id = []
        deprels_id = []
        ccgs_id = []
        for word, lemma, postag, deprel, ccg in sentence:
            sentence_text.append(word)
            wordl = word.lower()
            # we can expect typo or error in words and lemma
            words_id.append(words2id[wordl] if wordl in words2id else UNK_TAG)
            lemmas_id.append(lemmas2id[lemma] if lemma in lemmas2id else UNK_TAG)
            postags_id.append(postags2id[postag])
            deprels_id.append(deprels2id[deprel])
            ccgs_id.append(ccgs2id[ccg])
            tagset.append(ccg)
        data.append(
            {
                "sentence_text": sentence_text,
                "words_id": words_id,
                "lemmas_id": lemmas_id,
                "postags_id": postags_id,
                "deprels_id": deprels_id,
                "ccgs_id": ccgs_id,
            }
        )
    print("Done!")
    tagset = dict(Counter(tagset))
    return data, tagset


# pads a given sequence in parameter until it reaches sequence_length.
def pad_sequence(
    sequence: List[int], sequence_length: int, pad_value: int
) -> List[int]:
    padded_sequence = [pad_value for _ in range(sequence_length)]
    padded_sequence[: len(sequence)] = sequence[:]
    return padded_sequence
