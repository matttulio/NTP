# ---
# # Training a single Layer of Attention on the Histogram Task may lead to two solutions
# 
# This notebook shows how to train and evaluate single-layer transformers with dot-product attention and trained positional encodings.
# 
# - Part 1: Define data and model architecture
# - Part 2: Training of models with different settings on the attention layer (positional, semantic, or both)
# - Part 3: Introspecting the attention layers for some input sequences
# - Part 4: Checking whether the parameter values with the frozen weights stay close to their original position in the unfrozen weight space

import torch
from torch.utils.data import random_split
import pandas as pd
from src.transformer import *
from src.benchmarks import *
import os
import pickle

data_path = 'Datasets/Data/'

# Define the device to work on
if(torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'

#device = 'cuda'

print(f"Device used: {device}\n")

#task = 1  # Load primitive NLP dataset
task = 2  # Load NextHistogramTask dataset
#task = 3  # Load primitive NLP NTP dataset

if(task == 1):
    print("PRIMITIVE NLP TASK")
    file_name = 'primitive_NLP_dataset_n_smpl50000__seq_len10__cont_win10__'\
        'v_size78__emb_dim50__emb_typeglove.6B.50d__seed42__d_par2'
    file_ext = '.pkl'
    save_dir = os.path.join('Empirics', file_name)
elif(task == 2):
    print("NEXT HISTOGRAM TASK")
    file_name = 'NextHistogramDataset_n_smpl50000__seq_len10__v_size15__seed42'
    file_ext = '.pkl'
    save_dir = os.path.join('Empirics', file_name)
elif(task == 3):
    print("PPRIMITIVE NLP NTP TASK")
    file_name = 'primitive_NLP_NTP_dataset_n_smpl50000__seq_len10__cont_win10__'\
    'v_size78__emb_dim50__emb_typeglove.6B.50d__seed42__d_par1.1'
    file_ext = '.pkl'
    save_dir = os.path.join('Empirics', file_name)

file_name = file_name + file_ext

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

with open(os.path.join(data_path, file_name), "rb") as f: 
        dataset = pickle.load(f)

# Define the length, maximum value, and number of samples
n_classes = dataset.n_classes
vocab_size = dataset.vocab_size
seq_len = dataset.seq_len
num_samples = dataset.num_samples

# Split the dataset into train, test, and validation sets
seed = 42
generator = torch.Generator().manual_seed(seed)
train_ratio, val_ratio = 0.7, 0.3

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = generator)
train_size, val_size

hidden_dimension_fc = 128
model_dim = 64
n_runs = 5
#model_types = ['only_pos', 'only_sem']
model_types = ['only_sem', 'only_pos']
n_epochs = 200


results = []

for model_type in model_types:
    print(f'Running {model_type}...')
    for i in range(n_runs):
        transformer = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, model_type).to(device)
        torch.save(transformer.state_dict(), os.path.join(save_dir, f'run_{i}_initmodel_{transformer.attention_input}_orig.pt'))
        transformer, train_losses, val_losses, val_acc = train(transformer, train_dataset, val_dataset, n_epochs = n_epochs, n_classes = n_classes)
        
        torch.save(transformer.state_dict(), os.path.join(save_dir, f'run_{i}_model_{transformer.attention_input}_orig.pt'))
        results.append({
            'model_type': model_type,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_acc': val_acc,
            'run':i,
        })
    print(f'Done.')
    
pd.DataFrame(results).to_csv(os.path.join(save_dir, 'frozen_transformer_result.csv'), index=False)


n_epochs = n_epochs // 2
reparameterized_transformers = []

for r in range(n_runs):
    for model_type in model_types:
  
        print(r,model_type)
        orig_trans = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, model_type).to(device)
        orig_dict = torch.load(os.path.join(save_dir, f'run_{r}_model_{model_type}_orig.pt'))
        orig_trans.load_state_dict(orig_dict)

        rep_trans = reparameterize(orig_trans, vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len).to(device)
        torch.save(rep_trans.state_dict(), os.path.join(save_dir, f'run_{r}_model_{model_type}_repar.pt'))

        rep_trans, train_losses, val_losses, val_acc = train(rep_trans, train_dataset, val_dataset, 1e-4, n_epochs, n_classes)
        reparameterized_transformers.append({
          'train_losses': train_losses,
          'val_losses': val_losses,
          'val_acc': val_acc,
          'run': r,
          'model_type': model_type,
        })
        torch.save(rep_trans.state_dict(), os.path.join(save_dir, f'run_{r}_model_{model_type}_retrained.pt'))

pd.DataFrame(reparameterized_transformers).to_csv(os.path.join(save_dir, 'reparameterized_transformers.csv'), index=False)


