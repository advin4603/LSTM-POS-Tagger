import optuna
import torch
from torch.utils.data import DataLoader
from model import BiLSTMPOSTagger
from data import POSDataset, create_collate
from tqdm import tqdm
from sklearn import metrics
from optuna import Trial

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def main(
        # trial: Trial,
        EMBEDDING_DIM=200,
        HIDDEN_DIM=200,
        BATCH_SIZE=128,
        LSTM_STACKS=1,
        OPTIMIZER_FUNCTION=torch.optim.Adam,
        LEARNING_RATE=0.2,
        LOSS_FUNCTION=torch.nn.CrossEntropyLoss(),
        EPOCHS=20,
        DATASET_AUGMENT_PERCENT: float = 50,
        REMOVE_TOKEN_PERCENT: float = 50,
        TOKEN_REMOVAL_PROBABILITY: float = 0.5,
):
    training_data = POSDataset("ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-train.conllu",
                               dataset_augment_percent=DATASET_AUGMENT_PERCENT,
                               remove_token_percentage=REMOVE_TOKEN_PERCENT,
                               token_removal_probability=TOKEN_REMOVAL_PROBABILITY)
    validation_data = POSDataset("ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-dev.conllu",
                                 tagset=training_data.tagset,
                                 vocabulary=training_data.vocabulary)
    test_data = POSDataset("ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-test.conllu", tagset=training_data.tagset,
                           vocabulary=training_data.vocabulary)

    model = BiLSTMPOSTagger(EMBEDDING_DIM, HIDDEN_DIM, len(training_data.vocabulary), len(training_data.tagset),
                            LSTM_STACKS).to(device)

    collate_function = create_collate(training_data.vocabulary, training_data.tagset)

    training_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, collate_fn=collate_function, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, collate_fn=collate_function)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_function)

    optimizer = OPTIMIZER_FUNCTION(model.parameters(), lr=LEARNING_RATE)
    for t in range(EPOCHS):
        train_loop(training_dataloader, model, LOSS_FUNCTION, optimizer, t)
        # test_loop(trial, validation_dataloader, model, LOSS_FUNCTION, True, t)
        test_loop(validation_dataloader, model, LOSS_FUNCTION, True, t)
    test_loop(test_dataloader, model, LOSS_FUNCTION)

    torch.save(model.state_dict(), "tagger.pt")
    # f1 = test_loop(trial, validation_dataloader, model, LOSS_FUNCTION, True, EPOCHS)
    # torch.cuda.empty_cache()
    # return f1


def train_loop(dataloader: DataLoader, train_model: BiLSTMPOSTagger, loss_fn: torch.nn.modules.loss,
               train_optimizer: torch.optim.Optimizer, epoch_number: int):
    size = len(dataloader.dataset)
    pbar = tqdm(enumerate(dataloader))
    pbar.set_description(f"Training Epoch={epoch_number} :")
    for batch, (X, y) in pbar:
        X = X.to(device)
        y = y.to(device)
        pred = train_model(X).permute(0, 2, 1)
        loss = loss_fn(pred, y)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        loss, current = loss.item(), (batch + 1) * len(X)
        pbar.set_description(f"Training Epoch={epoch_number} : loss = {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: DataLoader, test_model: BiLSTMPOSTagger, loss_fn: torch.nn.modules.loss,
              validate: bool = False, epoch_number: int = 0, calculate_precision=True, calculate_recall=True,
              calculate_f1=True):
    test_loss, correct = 0, 0
    with torch.no_grad():
        pbar = tqdm(dataloader)
        pbar.set_description(f"Validating Epoch: {epoch_number}" if validate else "Testing:")
        y_true, y_pred = [], []
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            pred = test_model(X).permute(0, 2, 1)
            test_loss += loss_fn(pred, y).item()
            y_true.extend(y.flatten().cpu())
            y_pred.extend(pred.argmax(1).flatten().cpu())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() / pred.shape[2]

        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        precision_micro = metrics.precision_score(y_true, y_pred, average="micro") if calculate_precision else None
        precision_macro = metrics.precision_score(y_true, y_pred, average="macro") if calculate_precision else None
        recall_micro = metrics.recall_score(y_true, y_pred, average="micro") if calculate_recall else None
        recall_macro = metrics.recall_score(y_true, y_pred, average="macro") if calculate_recall else None
        f1_micro = metrics.f1_score(y_true, y_pred, average="micro") if calculate_f1 else None
        f1_macro = metrics.f1_score(y_true, y_pred, average="macro") if calculate_f1 else None
        print(f"{'Validation' if validate else 'Test'} Error: \n\tAccuracy: {(100 * correct):>0.1f}%"
              f"\n\tAvg loss: {test_loss:>8f}")

        if calculate_precision:
            print(f"\tPrecision(micro): {precision_micro}\n\tPrecision(macro): {precision_macro}")
        if calculate_recall:
            print(f"\tRecall(micro): {recall_micro}\n\tRecall(macro): {recall_macro}")
        if calculate_f1:
            print(f"\tF1-Score(micro): {f1_micro}\n\tF1-Score(macro): {f1_macro}")

    # if validate:
    #     trial.report(f1_macro, epoch_number)
    #
    #     if trial.should_prune():
    #         raise optuna.exceptions.TrialPruned()

    return f1_macro


def objective(trial: Trial):
    return main(
        # trial,
        EMBEDDING_DIM=trial.suggest_int("EMBEDDING_DIM", 50, 500),
        HIDDEN_DIM=trial.suggest_int("HIDDEN_DIM", 50, 500),
        BATCH_SIZE=trial.suggest_int("BATCH_SIZE", 16, 400),
        LSTM_STACKS=trial.suggest_int("LSTM_STACKS", 1, 3),
        OPTIMIZER_FUNCTION=torch.optim.Adam,
        LEARNING_RATE=trial.suggest_float("LEARNING_RATE", 1e-5, 5e-1),
        LOSS_FUNCTION=torch.nn.CrossEntropyLoss(),
        EPOCHS=10,
        DATASET_AUGMENT_PERCENT=trial.suggest_float("DATASET_AUGMENT_PERCENT", 0, 100),
        REMOVE_TOKEN_PERCENT=trial.suggest_float("REMOVE_TOKEN_PERCENT", 0, 100),
        TOKEN_REMOVAL_PROBABILITY=trial.suggest_float("TOKEN_REMOVAL_PROBABILITY", 0, 1)
    )


if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=50)
    # best_trial = study.best_trial
    #
    # for key, value in best_trial.params.items():
    #     print("{}: {}".format(key, value))
    # fig = optuna.visualization.plot_param_importances(study)
    # fig.show()
    main(
        EPOCHS=10,
        EMBEDDING_DIM=202,
        HIDDEN_DIM=100,
        BATCH_SIZE=376,
        LSTM_STACKS=1,
        LEARNING_RATE=0.022633996121221748,
        DATASET_AUGMENT_PERCENT=40.95957493392948,
        REMOVE_TOKEN_PERCENT=28.51465299453985,
        TOKEN_REMOVAL_PROBABILITY=0.573774390048683,
    )
