import pickle
import os
from src.benchmarks import *

## EXPRESS A DESIRE

#desire = 1  # Create primitive NLP dataset for next token prediction
desire = 2  # Create primitive NLP dataset for summing task
#desire = 3  # Create Next Histogram Task dataset

print("\n")

if(desire == 1):

    num_samples = 50000
    sequence_length = 10
    context_window = 10
    vocab_size = round(sequence_length * 7.8125)
    vocab = list(range(vocab_size))
    embedding_dim = 50
    embedding_path = 'Datasets/glove/glove.6B.50d.txt'
    embedding_model = 'glove.6B.50d'
    seed = 42
    distr_param = 1.1

    print("Building Primitive NLP Next Token Prediction Dataset...")
    dataset = PrimitiveNLP_NTP(num_samples, sequence_length, context_window, vocab, embedding_dim, embedding_path, seed, distr_param)

    if(embedding_path == None):
        embedding_model = 'Rand'

    save_path = "Datasets/Data"
    file_name = f"primitive_NLP_NTP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}.pkl"

    print(f"number of samples = {num_samples}, sequence lenght = {sequence_length}, context_window = {context_window},")
    print(f"vocabulary size = {vocab_size}, embedding dimension = {embedding_dim}, embedding type = {embedding_model},")
    print(f"seed = {seed}, distribution's parameter = {distr_param}")
    print("\n")


elif(desire == 2):

    num_samples = 50000
    sequence_length = 10
    context_window = 10
    vocab_size = round(sequence_length * 7.8125)
    vocab = list(range(vocab_size))
    embedding_dim = 50
    embedding_path = 'Datasets/glove/glove.6B.50d.txt'
    embedding_model = 'glove.6B.50d'
    seed = 42
    distr_param = 1.1

    print("Building Primitive NLP Dataset...")
    dataset = PrimitiveNLP(num_samples, sequence_length, context_window, vocab, embedding_dim, embedding_path, seed, distr_param)

    if(embedding_path == None):
        embedding_model = 'Rand'

    save_path = "Datasets/Data"
    file_name = f"primitive_NLP_dataset_n_smpl{num_samples}__seq_len{sequence_length}__cont_win{context_window}__" \
    + f"v_size{vocab_size}__emb_dim{embedding_dim}__emb_type{embedding_model}__seed{seed}__d_par{distr_param}.pkl"

    print(f"number of samples = {num_samples}, sequence lenght = {sequence_length}, context_window = {context_window},")
    print(f"vocabulary size = {vocab_size}, embedding dimension = {embedding_dim}, embedding type = {embedding_model},")
    print(f"seed = {seed}, distribution's parameter = {distr_param}")
    print("\n")


elif(desire == 3):

    num_samples = 50000
    sequence_length = 10
    vocab_size = 15
    seed = 42

    print("Building Next Histogram Dataset...")
    dataset = NextHistogramDataset(sequence_length, vocab_size, num_samples, seed)

    save_path = "Datasets/Data"
    file_name = f"NextHistogramDataset_n_smpl{num_samples}__seq_len{sequence_length}__v_size{vocab_size}__seed{seed}.pkl"

    print(f"number of samples = {num_samples}, sequence lenght = {sequence_length}, vocabulary size = {vocab_size}, seed = {seed}")
    print("\n")

if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path = os.path.join(save_path, file_name)
print(f"save path: {save_path}")
print("\n")

with open(save_path, "wb") as f:
    pickle.dump(dataset, f)
