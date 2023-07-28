from data import START_TOKEN, END_TOKEN
from model import BiLSTMPOSTagger
import torch
from tokenizer import tokenize_english
from data import POSDataset

training_data = POSDataset("ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-train.conllu")
device = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 10
EMBEDDING_DIM = 202
HIDDEN_DIM = 100
BATCH_SIZE = 376
LSTM_STACKS = 1
LEARNING_RATE = 0.022633996121221748
DATASET_AUGMENT_PERCENT = 40.95957493392948
REMOVE_TOKEN_PERCENT = 28.51465299453985
TOKEN_REMOVAL_PROBABILITY = 0.573774390048683
model = BiLSTMPOSTagger(EMBEDDING_DIM, HIDDEN_DIM, len(training_data.vocabulary), len(training_data.tagset),
                        LSTM_STACKS).to(device)

model.load_state_dict(torch.load("tagger.pt"))
model.eval()

sentence = tokenize_english(input("sentence: "))[0]
encoded_sentence = torch.LongTensor(
    [training_data.vocabulary[START_TOKEN]] +
    [training_data.vocabulary[i] for i in sentence] +
    [training_data.vocabulary[END_TOKEN]]
).to(device)

pred = model(encoded_sentence).argmax(1)
tags = training_data.tagset.lookup_tokens(list(pred))
print(tags[1:-1])
