# Local
from data_operations.vocabulary import Vocabulary
from inference.greedy_search import greedy_decode
from model_components.model import EncoderDecoder
from model_components.two_dimensional_weights import compute_2d_weight_vector
from model_training.batch import rebatch

# External
from torch.utils.data import DataLoader


def test_model(test_dataloader: DataLoader, model: EncoderDecoder, kmer_size: int):

    vocab = Vocabulary(kmer_size)
    predicted_sequences = []
    pred_kmer_seqs = []
    alphas = []
    for i, batch in enumerate(test_dataloader, 1):
        print(f"Predicting {i}")
        batch = rebatch(batch)
        ground_truth = batch.trg_input.tolist()[0]
        groundtruth_target_length_plus_bos = len(ground_truth)
        pred, attention = greedy_decode(model, batch.src_input, max_len=groundtruth_target_length_plus_bos,)
        pred_kmer_seqs.append(pred)
        pred_kmer = [vocab.itos[idx] for idx in pred]
        concat = []
        for index in range(len(pred_kmer)):
            if index != len(pred_kmer) - 1:
                concat.append(pred_kmer[index][0])
            else:
                concat.append(pred_kmer[index])
        concat = "".join(concat)
        print(len(concat))
        assert (
            len(concat) == groundtruth_target_length_plus_bos + kmer_size - 1 - 1
        ), f"Misprediction of {groundtruth_target_length_plus_bos + kmer_size - 1 - 1 -len(concat)}"
        predicted_sequences.append(concat)
        alphas.append(attention)
    compute_2d_weight_vector(pred_kmer_seqs, vocab, "predicted")
    return predicted_sequences, alphas
