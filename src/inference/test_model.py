# Local
from data_operations.vocabulary import Vocabulary
from inference.greedy_search import greedy_decode
from model_components.model import EncoderDecoder
from model_training.batch import rebatch

# External
from torch.utils.data import DataLoader


def test_model(test_dataloader: DataLoader, model: EncoderDecoder, kmer_size: int):

    vocab = Vocabulary(kmer_size)
    predicted_sequences = []
    alphas = []
    for i, batch in enumerate(test_dataloader, 1):
        print(f"Predicting {i}")
        batch = rebatch(batch)
        ground_truth = batch.trg_input.tolist()[0]
        max_len = len(ground_truth) - 1
        pred, attention = greedy_decode(model, batch.src_input, max_len=max_len)
        pred_kmer = [vocab.itos[idx] for idx in pred]
        concat = []
        for index, kmer in enumerate(pred_kmer, 1):
            if index != max_len - 1:
                concat.append(kmer[0])
            else:
                concat.append(kmer)
        concat = "".join(concat)
        assert len(concat) == max_len, "Concatenated string has incorrect length"
        predicted_sequences.append(concat)
        alphas.append(attention)
    return predicted_sequences, alphas
