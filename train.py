# Callable file to train the neural model stored in model.py.
# Will create log files at the end of the training and a weight files, result of the training in order to be used later on.
# Training results will be logged using classical prints.

import pickle
import atexit
import argparse
import sys
import os
import time
import random
import numpy as np
import torch
import pltpublish as pub
import tqdm
import matplotlib.pyplot as plt
from transformers import logging
from typing import List, Dict
from model import CamemBERTTagger
from torch_poly_lr_decay import PolynomialLRDecay
from torch.nn.utils.rnn import pad_sequence
from utils import (
    build_vocab_camembert,
    extract_tag_stat,
    shuffle_and_split,
    PAD_TAG,
    START_TAG,
    END_TAG,
    TASK_POS,
    TASK_TAG,
)
from dataHandle import (
    load_tlgbank_dataset_camembert,
)


def evaluate(
    model: CamemBERTTagger,
    data: List[Dict[str, List[str]]],
    best_acc: float,
    log_result: bool = False,
    mute_vae: bool = True,
    k: int = 1,
):
    common, uncommon, rare, unseen = tag_stat
    correct_total = [(0, 0), (0, 0), (0, 0), (0, 0)]
    correct_total_labels = ["Common", "Uncommon", "Rare", "Unseen"]
    correct_pos = 0
    total_pos = 0
    batch_index = 0
    number_batch_per_epoch = int(np.ceil(len(data) / batch_size))
    errors: Dict[int, int] = {}
    with torch.no_grad():
        for _ in range(number_batch_per_epoch):
            dataset_index = batch_index * batch_size
            batch = data[dataset_index : dataset_index + batch_size]
            batch_index += 1

            while len(batch) < batch_size:
                batch.append(
                    (
                        {"input_ids": [5, 6], "attention_mask": [1, 1]},
                        [pos2id[START_TAG], pos2id[END_TAG]],
                        [tags2id[START_TAG], tags2id[END_TAG]],
                    )
                )

            words_batch, pos_batch, tags_batch = list(zip(*batch))
            masks_batch = [w["attention_mask"] for w in words_batch]
            words_batch = [w["input_ids"] for w in words_batch]

            words_batch = pad_sequence(
                [torch.tensor(w, device=device) for w in words_batch], batch_first=True
            )
            pos_batch = pad_sequence(
                [torch.tensor(p, device=device) for p in pos_batch],
                batch_first=True,
                padding_value=pos2id[PAD_TAG],
            )
            tags_batch = pad_sequence(
                [torch.tensor(t, device=device) for t in tags_batch],
                batch_first=True,
                padding_value=tags2id[PAD_TAG],
            )

            masks_batch = pad_sequence(
                [torch.tensor(m, device=device) for m in masks_batch], batch_first=True
            )

            predictions = model(words_batch, masks_batch, mute_vae, k=k)

            predictions_pos = model(
                words_batch, masks_batch, mute_vae, k=k, task=TASK_POS
            )
            for i in range(len(masks_batch)):
                for (m, predicted_pos, true_pos, predicted_tag, true_tag) in zip(
                    masks_batch[i],
                    predictions_pos[i],
                    pos_batch[i],
                    predictions[i],
                    tags_batch[i],
                ):
                    m = m.item()
                    true_tag = true_tag.item()
                    true_pos = true_pos.item()
                    if (
                        not (m)
                        or true_tag == tags2id[PAD_TAG]
                        or true_tag == tags2id[END_TAG]
                        or true_tag == tags2id[START_TAG]
                    ):
                        continue
                    tag_type = 3
                    if true_tag in common:
                        tag_type = 0
                    elif true_tag in uncommon:
                        tag_type = 1
                    elif true_tag in rare:
                        tag_type = 2
                    correct, total = correct_total[tag_type]
                    if true_tag == predicted_tag:
                        correct += 1
                    elif log_result:
                        errors[true_tag] = (
                            errors[true_tag] + 1 if true_tag in errors else 1
                        )
                    if true_pos == predicted_pos:
                        correct_pos += 1
                    total += 1
                    total_pos += 1
                    correct_total[tag_type] = (correct, total)
        errorsf = eval_temp + "/unrecognized_tags.txt"
        if log_result:
            with open(errorsf, "w") as f:
                f.write("\n".join([f"{id2tags[x]} \t\t {errors[x]}" for x in errors]))
            print(f"Errors logged in {errorsf}")

    print("Correct / wrong guesses:")
    global_correct = 0
    global_total = 0
    for type in range(4):
        correct, total = correct_total[type]
        if total == 0:
            print(f"No '{correct_total_labels[type]}' tags were evaluated.")
            continue
        global_correct += correct
        global_total += total
        print(
            f"\t- {correct_total_labels[type]}:\n\t\t- Correct/wrong: {correct}/{total-correct}\n\t\t- Acc : {correct/total if total > 0 else 'N/A'}"
        )
    new_acc = global_correct / global_total
    print(
        f"\t- Global:\n\t\t- Correct/wrong: {global_correct}/{global_total-global_correct}\n\t\t- Pos: {correct_pos}/{total_pos-correct_pos}\n\t\t- Acc : {new_acc}\n\t\t- Acc Pos : {correct_pos/total_pos}"
    )
    if new_acc > best_acc:
        best_acc = new_acc

    return new_acc, best_acc


def train_model(learning_rate: float, mute_vae: bool):
    def save_model() -> None:
        torch.save(model.state_dict(), model_file)
        print(f"Saved model dictionary state file in {model_file}")

    def do_batch(batch_index: int, dataset: List) -> int:
        dataset_index = batch_index * batch_size
        batch = dataset[dataset_index : dataset_index + batch_size]

        # complete with dummy sentences uncompleted batch (last one)
        while len(batch) < batch_size:
            batch.append(
                (
                    {"input_ids": [5, 6], "attention_mask": [1, 1]},
                    [pos2id[START_TAG], pos2id[END_TAG]],
                    [tags2id[START_TAG], tags2id[END_TAG]],
                )
            )

        words_batch, pos_batch, tags_batch = list(zip(*batch))

        masks_batch = [w["attention_mask"] for w in words_batch]
        words_batch = [w["input_ids"] for w in words_batch]

        words_batch = pad_sequence(
            [torch.tensor(w, device=device) for w in words_batch], batch_first=True
        )
        pos_batch = pad_sequence(
            [torch.tensor(p, device=device) for p in pos_batch],
            batch_first=True,
            padding_value=pos2id[PAD_TAG],
        )
        tags_batch = pad_sequence(
            [torch.tensor(t, device=device) for t in tags_batch],
            batch_first=True,
            padding_value=tags2id[PAD_TAG],
        )
        masks_batch = pad_sequence(
            [torch.tensor(m, device=device) for m in masks_batch], batch_first=True
        )

        model.zero_grad()
        nll = model.nll(words_batch, tags_batch, masks_batch, mute_vae=mute_vae)
        nll += model.nll(
            words_batch, pos_batch, masks_batch, mute_vae=mute_vae, task=TASK_POS
        )
        nll.backward()

        optimizer.step()
        return nll.item()

    lr = learning_rate

    camembert_param = [] 
    camembert_param.extend(model.get_camembert().parameters())
    vae_param = []
    vae_param.extend(model.get_vae().parameters())
    if mute_vae:
        params = [p for p in model.parameters() if id(p) not in [id(c) for c in camembert_param] and id(p) not in [id(v) for v in vae_param]]
    else:
        params = [p for p in model.parameters() if id(p) not in [id(c) for c in camembert_param]]
    optimizer = torch.optim.Adam([
        {'params': camembert_param, 'lr': lr/10},
        {'params': params, 'lr': lr}
        ], lr=lr)
    lr_reduce = PolynomialLRDecay(
        optimizer,
        max_decay_steps=epochs,
        end_learning_rate=0
        )

    loss = 0.0
    history = []
    best_accuracy_valid = -1.0
    print("Starting training...")
    model.train(True)
    number_batch_per_epoch = int(np.ceil(len(dataset_train) / batch_size))
    total_counter = 0
    atexit.register(save_model)
    early_stopping_ctr: int = 5
    early_stopping_threshold: float = 0.5
    early_stopping_previous: float = 0.0
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        random.shuffle(dataset_train)
        batch_index = 0
        for _ in tqdm.trange(number_batch_per_epoch, desc="batchs"):
            loss += do_batch(batch_index, dataset_train)
            total_counter += batch_size
            batch_index += 1

        model.train(False)
        previous_best = best_accuracy_valid
        accuracy_valid, best_accuracy_valid = evaluate(
            model, dataset_valid, best_accuracy_valid, mute_vae=mute_vae
        )
        if previous_best < best_accuracy_valid:
            save_model()
        print("Overall validation accuracy:")
        print(f"\t- valid: {accuracy_valid}")
        model.train(True)
        avg_loss = loss / number_batch_per_epoch
        # simple early stopping
        if abs(early_stopping_previous - round(avg_loss, 1)) < early_stopping_threshold:
            early_stopping_ctr -= 1
        else:
            early_stopping_ctr = 5
            early_stopping_previous = round(avg_loss, 1)
        history.append(avg_loss)
        loss = 0.0
        print(f"Average batch loss during this epoch: \t{avg_loss}")
        lr_reduce.step()
        if early_stopping_ctr <= 0:
            print(
                "No loss modification for the last 5 epochs. Early stopping will now end the training."
            )
            break

    atexit.unregister(save_model)
    pub.setup()
    plt.figure()
    plt.plot(history)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    pub.save_fig("loss.png")
    plt.close()
    accuracy_test, best_accuracy_test = evaluate(
        model, dataset_test, -1.0, True, mute_vae=mute_vae
    )
    print(f"Final test accuracy: {accuracy_test}")


def train_vae():
    def save_model() -> None:
        torch.save(model.state_dict(), vae_model_file)
        print(f"Saved model dictionary state file in {vae_model_file}")

    def do_batch(batch_index: int, dataset: List) -> None:
        dataset_index = batch_index * batch_size
        batch = dataset[dataset_index : dataset_index + batch_size]

        # complete with dummy sentences uncompleted batch (last one)
        while len(batch) < batch_size:
            batch.append(
                (
                    {"input_ids": [5, 6], "attention_mask": [1, 1]},
                    [pos2id[START_TAG], pos2id[END_TAG]],
                    [tags2id[START_TAG], tags2id[END_TAG]],
                )
            )

        words_batch, _, _ = list(zip(*batch))
        masks_batch = [w["attention_mask"] for w in words_batch]
        words_batch = [w["input_ids"] for w in words_batch]

        words_batch = pad_sequence(
            [torch.tensor(w, device=device) for w in words_batch], batch_first=True
        )
        masks_batch = pad_sequence(
            [torch.tensor(m, device=device) for m in masks_batch], batch_first=True
        )

        model.zero_grad()
        loss, kl, recon = model.vae_loss(words_batch, masks_batch)
        loss.backward()
        optimizer.step()
        return loss.item(), kl.item(), recon.item()



    epochs_vae = 10
    history_loss = []
    lr = 1e-4
    optimizer = torch.optim.Adam(model.get_vae().parameters(), lr=lr)
    lr_reduce = PolynomialLRDecay(
        optimizer,
        max_decay_steps=epochs,
        end_learning_rate=0
        )
    
    loss = 0.0
    kl = 0.0
    recon = 0.0
    print("Starting training...")
    number_batch_per_epoch = int(np.ceil(len(dataset_train) / batch_size))
    total_counter = 0
    atexit.register(save_model)
    early_stopping_ctr: int = 5
    early_stopping_threshold: float = 0.0001
    early_stopping_previous: float = 0.0
    best_acc: float = 0.0
    for i in range(epochs_vae):
        model.get_vae().train(True)
        print(f"Epoch {i+1}/{epochs_vae}")
        random.shuffle(dataset_train)
        batch_index = 0
        for _ in tqdm.trange(number_batch_per_epoch, desc="batchs"):
            t_loss, t_kl, t_recon = do_batch(batch_index, dataset_train)
            loss += t_loss
            kl += t_kl
            recon += t_recon
            total_counter += batch_size
            batch_index += 1
        print(
            f"Loss: {loss/number_batch_per_epoch}\tkl: {kl/number_batch_per_epoch}\trecon: {recon/number_batch_per_epoch}"
        )
        # simple early stopping
        if (
            abs(early_stopping_previous - round(loss / number_batch_per_epoch, 6))
            < early_stopping_threshold
        ):
            early_stopping_ctr -= 1
        else:
            early_stopping_ctr = 5
            early_stopping_previous = round(loss / number_batch_per_epoch, 6)
        history_loss.append(loss)
        loss = 0.0
        kl = 0.0
        recon = 0.0
        model.get_vae().train(False)
        acc, best_acc = evaluate(model, dataset_valid, best_acc, mute_vae=False)
        print("Overall validation accuracy:")
        print(f"\t- valid: {acc}")
        if acc == best_acc:
            save_model()
        lr_reduce.step()
    atexit.unregister(save_model)
    pub.setup()
    plt.figure()
    plt.plot(history_loss[1:])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    pub.save_fig("loss_vae.png")
    plt.close()
    accuracy_test, best_accuracy_test = evaluate(
        model, dataset_test, 0.0, mute_vae=False
    )
    print(f"Final test accuracy: {accuracy_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="French CCG supertagger [train].")
    parser.add_argument("-m", "--model", default="", type=str, help="model file")
    parser.add_argument(
        "-vaem", "--vae-model", default="", type=str, help="vae model file"
    )
    parser.add_argument(
        "-ep",
        "--epochs",
        default=20,
        type=int,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="tlgbank.txt",
        type=str,
        help="dataset file of FrenchCCGBank in txt format (default: tlgbank.txt)",
    )
    parser.add_argument(
        "-vr",
        "--validation-ratio",
        default=0.1,
        type=float,
        help="validation/training ratio for training (default: 0.1)",
    )
    parser.add_argument(
        "-tr",
        "--test-ratio",
        default=0.1,
        type=float,
        help="test/dataset ratio for training (default: 0.1)",
    )
    parser.add_argument(
        "-s", "--seed", default=4444, type=int, help="random seed (default: 4444)"
    )
    parser.add_argument(
        "-b", "--batch-size", default=5, type=int, help="batch size (default: 5)"
    )
    parser.add_argument(
        "--max-sequence-length",
        default=200,
        type=int,
        help="Max sequence length (default: 200)",
    )
    parser.add_argument(
        "--hidden-size", default=768, type=int, help="Hidden size (default: 768)"
    )
    parser.add_argument(
        "--dropout", default=0.4, type=float, help="Dropout (default: 0.4)"
    )
    parser.add_argument(
        "--num-layers", default=1, type=int, help="Number of LSTM layers (default: 1)"
    )
    parser.add_argument(
        "--latent-dim", default=200, type=int, help="VAE latent dim (default: 200)"
    )

    parameters = parser.parse_args()
    model_file: str = parameters.model
    vae_model_file: str = parameters.vae_model
    epochs: int = parameters.epochs
    dataset_file: str = parameters.dataset
    vratio: float = parameters.validation_ratio
    tratio: float = parameters.test_ratio
    seed: int = parameters.seed
    batch_size: int = parameters.batch_size
    max_sequence_length: int = parameters.max_sequence_length
    hidden_size: int = parameters.hidden_size
    dropout: float = parameters.dropout
    num_layers: int = parameters.num_layers
    latent_dim: int = parameters.latent_dim

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    eval_path = "./evaluation"
    eval_temp = os.path.join(eval_path, "temp")
    start_time = time.time()
    if not model_file == "" and (
        not os.path.exists(model_file) or not os.path.isfile(model_file)
    ):
        print(f"Model file was provided but could not be loaded. Given model file: {model_file}", file=sys.stderr)
        sys.exit(1)
    if not vae_model_file == "" and (
        not os.path.exists(vae_model_file) or not os.path.isfile(vae_model_file)
    ):
        print(
            f"Model file with trained VAE was provided but could not be loaded. Given model file: {vae_model_file}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.exists(dataset_file) or not os.path.isfile(dataset_file):
        print("Dataset must be a valid dataset file!", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)
    # dataset loading
    ## French TLG Bank
    dataset = load_tlgbank_dataset_camembert(dataset_file)

    vocab = build_vocab_camembert(dataset)
    tags2id, id2tags, id2freq = vocab[0]
    pos2id, id2pos, _ = vocab[1]

    # model initialization
    # we don't use CamemBERT head layer. Following line will mute related warning
    logging.set_verbosity_error()
    model = CamemBERTTagger(
        tags2id,
        pos2id,
        hidden_dim=hidden_size,
        batch_size=batch_size,
        dropout=dropout,
        latent_space=latent_dim,
    )
    model.to(device)

    # adapt dataset to RoBERTa architecture
    tokenized_dataset = model.tokenize_dataset(
        dataset,  max_sequence_length=max_sequence_length
    )

    # seeding 
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # shuffling and splitting dataset

    dataset_train, dataset_valid, dataset_test = shuffle_and_split(
        tokenized_dataset, vratio, tratio, seed
    )
    print(
        f"Lengths of training / validation / test datasets in number of sentences: {len(dataset_train)} / {len(dataset_valid)} / {len(dataset_test)}"
    )

    tag_stat = extract_tag_stat(id2freq, dataset_test, tags2id[END_TAG])

    end_time = time.time()
    print(f"Training prep done in {end_time - start_time}s")

    if vae_model_file == "":
        if model_file == "":
            model_file = "model.pt"
            print("Starting model training...")
            start_time = time.time()
            lr = 1e-4
            train_model(lr, mute_vae=True)
            end_time = time.time()
            print(f"Training done in {end_time - start_time}s")
        else:
            print(f"Found model file {model_file}.")
            model.load_state_dict(torch.load(model_file))
            print("Model state dictionary loaded.")
            accuracy_test, best_accuracy_test = evaluate(
                model, dataset_test, -1.0, True, mute_vae=True
            )
        vae_model_file = "vae_model.pt"
        print("Adding and training VAE layer...")
        start_time = time.time()
        train_vae()
        end_time = time.time()
        print(f"VAE training done in {end_time - start_time}s")
    else:
        print(f"Found vae model file {vae_model_file}.")
        model.load_state_dict(torch.load(vae_model_file))
        print("Model with trained vae state dictionary loaded.")
    print(f"Final training phrase...")
    start_time = time.time()
    lr = 1e-4
    # final phase
    epochs = 10
    model_file = "final_model.pt"
    train_model(lr, mute_vae=False)
    end_time = time.time()
    print(f"Final training done in {end_time - start_time}s")

    pickle_file = "model_data.pickle"
    print(f"Saving model data in {pickle_file}...")

    model_data = {
        "tags2id": tags2id,
        "id2tags": id2tags,
        "pos2id": pos2id,
        "id2pos": id2pos,
        "nb_output_class": len(tags2id),
        "max_sequence_length": max_sequence_length,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
    }

    with open(pickle_file, "wb") as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
