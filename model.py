import torch
from torch import nn
from torch import Tensor
from torch.nn import init
from typing import Dict, Tuple, List
from torchcrf import CRF

from transformers import CamembertModel, CamembertTokenizerFast
from utils import PAD_TAG, START_TAG, END_TAG, TASK_TAG, TASK_POS

def init_lstm(lstm: nn.Module) -> None:
    for param in lstm.parameters():
        if len(param.shape) >= 2:
            init.orthogonal_(param.data)
        else:
            init.normal_(param.data)


class BiLSTMVAE(nn.Module):
    def __init__(
        self,
        nb_outputs: int,
        device: str,
        hidden_dim: int = 768,
        latent_dim: int = 400,
        num_layers: int = 2,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        self._nb_outputs = nb_outputs
        self._hidden_dim = hidden_dim
        self._latent_dim = latent_dim
        self._num_layers = num_layers
        self._batch_size = batch_size
        self._device = device
        # encoder
        self._encoder_lstm = nn.LSTM(
            input_size=self._hidden_dim,
            hidden_size=self._hidden_dim // 2,
            num_layers=self._num_layers,
            bidirectional=True,
            batch_first=True,
        )
        init_lstm(self._encoder_lstm)
        self._mean = nn.Linear(
            in_features=(self._hidden_dim // 2) * self._num_layers,
            out_features=self._latent_dim,
        )
        self._log_var = nn.Linear(
            in_features=(self._hidden_dim // 2) * self._num_layers,
            out_features=self._latent_dim,
        )

        # decoder
        self._hidden_decoder = nn.Linear(
            in_features=self._latent_dim,
            out_features=(self._hidden_dim // 2) * self._num_layers,
        )
        self._decoder_lstm = nn.LSTM(
            input_size=self._hidden_dim,
            hidden_size=self._hidden_dim // 2,
            num_layers=self._num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self._mse = nn.MSELoss()

    def encoder(self, sentence: Tensor, hidden_states: Tuple[Tensor]) -> Tuple[Tensor]:
        _, hidden_states = self._encoder_lstm(sentence, hidden_states)
        mean = self._mean(hidden_states[0])
        log_var = self._log_var(hidden_states[0])
        var = torch.exp(0.5 * log_var).to(self._device)

        # gaussian noise
        gauss = torch.randn(self._batch_size, self._num_layers, device=self._device)

        z = gauss * var + mean

        return z, mean, log_var, hidden_states

    def _decoder(self, sentence: Tensor, z: Tensor) -> Tensor:
        hidden_decoder = self._hidden_decoder(z)
        hidden_decoder = (hidden_decoder, hidden_decoder)

        predictions, hidden_decoder = self._decoder_lstm(sentence, hidden_decoder)
        return predictions

    def forward(self, sentence: Tensor, hidden_states: Tuple[Tensor]) -> Tensor:
        z, mean, log_var, hidden_states = self.encoder(sentence, hidden_states)
        predictions = self._decoder(sentence, z)
        return predictions, mean, log_var, hidden_states

    def _kl_loss(self, mean: Tensor, log_var: Tensor) -> Tensor:
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return kl

    def _reconstruction_loss(self, sentence: Tensor, prediction: Tensor) -> Tensor:
        reconstruction = self._mse(sentence, prediction)
        return reconstruction

    def loss(
        self, sentence: Tensor, prediction: Tensor, mean: Tensor, log_var: Tensor
    ) -> Tensor:
        kl = self._kl_loss(mean, log_var)
        reconstruction = self._reconstruction_loss(sentence, prediction)
        elbo = kl + reconstruction
        return elbo, kl, reconstruction


class TaggerHead(nn.Module):
    def __init__(self, dict2id: Dict[str, int], hidden_dim: int, beam: float = 5e-2) -> None:
        super().__init__()
        self._dict2id = dict2id
        self._nb_outputs = len(dict2id)
        self._hidden_dim = hidden_dim
        self._beam = beam
        self._hidden2tags = nn.Linear(self._hidden_dim, self._nb_outputs)
        init.xavier_normal_(self._hidden2tags.weight.data)
        init.normal_(self._hidden2tags.bias.data)
        self._crf = CRF(self._nb_outputs, batch_first=True)#, beam=beam)
        self._activation = nn.LogSoftmax(-1)


    def forward(
        self, emissions: Tensor, mask: Tensor, k: int = 1
    ) -> Tuple[List[float], List[int]]:
        predictions = self._hidden2tags(emissions)
        paths = self._crf.decode(predictions, mask.bool())
        return paths

    def nll(self, emissions: Tensor, real_tags: Tensor, mask: Tensor) -> Tensor:
        predictions = self._hidden2tags(emissions)
        nll = -self._crf(predictions, real_tags, mask.bool())
        return nll


class CamemBERTTagger(nn.Module):
    def __init__(
        self,
        dict2id: Dict[str, int],
        pos2id: Dict[str, int],
        hidden_dim: int = 768,
        num_layers: int = 1,
        batch_size: int = 16,
        latent_space: int = 400,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._dict2id = dict2id
        self._pos2id = pos2id
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._num_layers = num_layers
        self._batch_size = batch_size

        self._camembert = CamembertModel.from_pretrained("camembert-base")
        self._tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")

        self._lstm = nn.LSTM(
            input_size=self._hidden_dim,
            hidden_size=self._hidden_dim // 2,
            num_layers=self._num_layers,
            bidirectional=True,
            batch_first=True,
        )
        init_lstm(self._lstm)

        self._vae_hidden_state = (
            torch.zeros(
                self._num_layers * 2,
                self._batch_size,
                self._hidden_dim // 2,
                device=self._device,
            ),
            torch.zeros(
                self._num_layers * 2,
                self._batch_size,
                self._hidden_dim // 2,
                device=self._device,
            ),
        )
        self._lstm_vae = BiLSTMVAE(
            nb_outputs=self._hidden_dim,
            device=self._device,
            hidden_dim=self._hidden_dim,
            latent_dim=latent_space,
            num_layers=self._num_layers,
            batch_size=batch_size,
        )
        self._head_tag = TaggerHead(self._dict2id, self._hidden_dim)
        self._head_pos = TaggerHead(self._pos2id, self._hidden_dim)
        self._dropout = lambda x, y: nn.functional.dropout(x, dropout, y, False)

    def _get_sentence_prediction_emission(
        self, sentence: Tensor, mask: Tensor, mute_vae: bool, dropout: bool
    ) -> Tensor:
        # get features form camembert
        features = self._camembert(sentence, attention_mask=mask).last_hidden_state
        # dims = [batch_size, sentence_length, 768]
        emissions, _ = self._lstm(features)
        emissions = self._dropout(emissions, dropout)
        if not mute_vae:
            emissions, _, _, self._vae_hidden_state = self._lstm_vae(
                emissions, self._vae_hidden_state
            )
            emissions = self._dropout(emissions, dropout)
        return emissions

    def vae_loss(self, sentence: Tensor, mask: Tensor) -> Tensor:
        features = self._camembert(sentence, attention_mask=mask).last_hidden_state
        emissions, _ = self._lstm(features)
        new_emissions, mean, log_var, self._vae_hidden_state = self._lstm_vae(
            emissions, self._vae_hidden_state
        )
        self._vae_hidden_state = (
            self._vae_hidden_state[0].detach(),
            self._vae_hidden_state[1].detach(),
        )
        loss = self._lstm_vae.loss(emissions, new_emissions, mean, log_var)
        return loss

    def nll(
        self,
        sentence: Tensor,
        real_tags: Tensor,
        mask: Tensor,
        mute_vae: bool = True,
        dropout: bool = True,
        task: str = TASK_TAG,
    ) -> Tensor:
        emissions = self._get_sentence_prediction_emission(
            sentence, mask, mute_vae, dropout
        )
        self._vae_hidden_state = (
            self._vae_hidden_state[0].detach(),
            self._vae_hidden_state[1].detach(),
        )
        if task == TASK_TAG:
            nll = self._head_tag.nll(emissions, real_tags, mask)
        elif task == TASK_POS:
            nll = self._head_pos.nll(emissions, real_tags, mask)
        else:
            raise ValueError(
                "Incorrect task value. Tasks currently supported are 'tag' and 'pos'."
            )
        return nll

    def forward(
        self,
        sentence: Tensor,
        mask: Tensor,
        mute_vae: bool = True,
        dropout: bool = False,
        k: int = 1,
        task: str = TASK_TAG,
    ) -> Tuple[List[float], List[int]]:
        emissions = self._get_sentence_prediction_emission(
            sentence, mask, mute_vae, dropout
        )

        if task == TASK_TAG:
            paths = self._head_tag(emissions, mask)
        elif task == TASK_POS:
            paths = self._head_pos(emissions, mask)
        else:
            raise ValueError(
                "Incorrect task value. Tasks currently supported are 'tag' and 'pos'."
            )
        return paths

    def tokenize_dataset(
        self,
        dataset: List[Tuple[List[str], List[str]]],
        max_sequence_length: int = 200,
    ) -> Tuple[Dict, List[List[int]]]:
        tokenized_dataset = []
        for sentence, postags, tags in dataset:
            s = self._tokenizer(
                sentence, is_split_into_words=True
            )  
            token_ids = s.word_ids()
            if len(token_ids) <= max_sequence_length:
                new_tagset = [self._dict2id[START_TAG]]
                new_postagset = [self._pos2id[START_TAG]]
                last = False
                prev = None
                for t in token_ids[1:]:
                    if t is None:
                        if last is False:
                            new_tagset.append(self._dict2id[END_TAG])
                            new_postagset.append(self._pos2id[END_TAG])
                            last = True
                        else:
                            new_tagset.append(self._dict2id[PAD_TAG])
                            new_postagset.append(self._pos2id[PAD_TAG])
                    elif t == prev:
                        new_tagset.append(self._dict2id[PAD_TAG])
                        new_postagset.append(self._pos2id[PAD_TAG])
                    else:
                        new_tagset.append(self._dict2id[tags[t]])
                        new_postagset.append(self._pos2id[postags[t]])
                    prev = t
                tokenized_dataset.append((s, new_postagset, new_tagset))
        return tokenized_dataset

    def decode(self, sentence: List[List[int]]) -> str:
        import re

        return [
            re.sub(
                "([.,?!()])",
                r" \1 ",
                self._tokenizer.decode(s, skip_special_tokens=True),
            )
            for s in sentence
        ]

    def get_vae(self) -> BiLSTMVAE:
        return self._lstm_vae

    def tokenize_inputs(
        self, inputs: List[List[str]], max_sequence_length: int = 200
    ) -> List[Dict]:
        tokenized = [
            self._tokenizer(sentence, is_split_into_words=True) for sentence in inputs
        ]
        for dict in tokenized:
            dict["words_id"] = dict.word_ids()
        if any([len(t.word_ids()) > max_sequence_length for t in tokenized]):
            print(f"WARNING: Sequence length exceeded {max_sequence_length}.")
        return tokenized

    def get_camembert(self):
        return self._camembert
