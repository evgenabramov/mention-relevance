from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from scipy.special import softmax

BATCH_SIZE = 32
MAX_LENGTH = 128

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def prepare_dataloader(frame: pd.DataFrame) -> DataLoader:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    input_ids = []
    token_type_ids = []
    attention_masks = []

    for i, row in tqdm(frame.iterrows()):
        query = row["query"]
        document = row["title"] + " " + row["snippet"]

        encoded_dict = tokenizer.encode_plus(
            query, document,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict["token_type_ids"])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    dataset = TensorDataset(input_ids, token_type_ids, attention_masks)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def make_predictions(model, dataloader: DataLoader) -> np.ndarray:
    model.eval()

    predictions = []

    for batch in tqdm(dataloader):
        b_input_ids = batch[0].to(device)
        b_token_type_ids = batch[1].to(device)
        b_input_mask = batch[2].to(device)

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()

        predictions.append(logits)

    probs = softmax(np.concatenate(predictions, axis=0), axis=1)[:, 1]
    return probs


def run_model(input_file: str, output_file: str) -> None:
    frame = pd.read_csv(input_file).reset_index(drop=True)

    assert all(column in frame.columns for column in ["id", "Query", "Title", "Snippet"]), \
        "Wrong columns in input dataframe"

    frame = frame.rename(columns={"Query": "query", "Title": "title", "Snippet": "snippet"})

    test_dataloader = prepare_dataloader(frame)

    model = BertForSequenceClassification.from_pretrained(
        "./models/pretrained-bert.pt",
        num_labels=2,  # binary classification
        output_attentions=False,
        output_hidden_states=False,
    )

    model = model.to(device)

    frame["relevance_probability"] = make_predictions(model, test_dataloader)
    frame[["id", "relevance_probability"]].to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input csv location")
    parser.add_argument("-o", "--output", default="predictions.csv", help="Result csv file location")
    args = parser.parse_args()
    run_model(args.input, args.output)
