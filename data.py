import torch
import torchtext.vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import *
from conllu import parse, TokenList, Token
from torchtext.vocab import build_vocab_from_iterator
import os
import random

UNKNOWN_TOKEN = "*unknown*"
START_TOKEN = "*start*"
END_TOKEN = "*end*"


class POSDataset(Dataset):
    def __init__(self, pos_file_path: Union[str, bytes, os.PathLike], tagset: torchtext.vocab.Vocab = None,
                 vocabulary=None, dataset_augment_percent: float = 0, remove_token_percentage: float = 0,
                 token_removal_probability: float = 0):
        with open(pos_file_path) as f:
            self.sentences = parse(f.read())

        word_iterator = ((str(i) for i in sentence) for sentence in self.sentences)
        if vocabulary is None:
            self.vocabulary = build_vocab_from_iterator(word_iterator, specials=[UNKNOWN_TOKEN, START_TOKEN, END_TOKEN])
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            self.vocabulary = vocabulary
        tag_iterator = ((i["upos"] for i in sentence) for sentence in self.sentences)
        if tagset is None:
            self.tagset = build_vocab_from_iterator(tag_iterator, specials=[START_TOKEN, END_TOKEN])
        else:
            self.tagset = tagset

        augment_sentences = random.choices(self.sentences, k=int(dataset_augment_percent * len(self.sentences) / 100))
        remove_indices = []
        for i, sentence in enumerate(augment_sentences):
            augment_sentences[i] = sentence = sentence.copy()
            tokens = set(str(i) for i in sentence)
            remove_tokens = random.choices(list(tokens), k=int(remove_token_percentage * len(tokens) / 100))
            augmented = False
            for word in sentence:
                if str(word) in remove_tokens and random.random() < token_removal_probability:
                    word["form"] = UNKNOWN_TOKEN
                    augmented = True
            if not augmented:
                remove_indices.append(i)

        for i in remove_indices[::-1]:
            augment_sentences.pop(i)

        self.sentences.extend(augment_sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        sentence = self.sentences[idx]
        encoded_sentence = torch.LongTensor(
            [self.vocabulary[START_TOKEN]] +
            [self.vocabulary[str(i)] for i in sentence] +
            [self.vocabulary[END_TOKEN]]
        )
        encoded_tags = torch.LongTensor(
            [self.tagset[START_TOKEN]] +
            [self.tagset[i["upos"]] for i in sentence] +
            [self.tagset[END_TOKEN]]
        )

        return encoded_sentence, encoded_tags


def create_collate(vocabulary: torchtext.vocab.Vocab, tagset: torchtext.vocab.Vocab) -> Callable:
    def custom_collate(data: Sequence[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_list, y_list = [], []
        for x, y in data:
            x_list.append(x)
            y_list.append(y)

        return (
            pad_sequence(x_list, batch_first=True, padding_value=vocabulary[END_TOKEN]),
            pad_sequence(y_list, batch_first=True, padding_value=tagset[END_TOKEN])
        )

    return custom_collate


if __name__ == "__main__":
    d1 = POSDataset("ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-train.conllu")
    d1.sentences.append(TokenList(Token()))
    print(d1.tagset.get_stoi())
