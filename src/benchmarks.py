import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm


def hist(s):
    
    counts = {}
    result = []

    for num in s:
        if num not in counts:
            counts[num] = 0
        counts[num] += 1
        result.append(counts[num])

    return result

class NextHistogramDataset(Dataset):
    def __init__(self, seq_len, vocab_size, n_samples,seed=42):
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = n_samples
        
        rs = np.random.RandomState(seed)
        
        self.X = rs.randint(0, self.vocab_size, (n_samples, seq_len))
        self.X = np.unique(self.X, axis=0).tolist()
        
        self.y = []
        
        for seq in self.X:
            counts = hist(seq)
            self.y.append(counts)
            
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.n_classes = len(np.unique(self.y)) + 1
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)

    
    
class PrimitiveNLP(Dataset):
    def __init__(self, num_samples, sequence_length, context_window, vocab, embedding_dim, embedding_path = None, seed = 42, distr_param = 1.1):

        """
        Generate a dataset for next token prediction that emulates natural language.

        Args:
        - num_samples (int): Number of samples to generate.
        - sequence_length (int): Length of each sequence.
        - context_window (int): Lenght of the context.
        - vocab (list): Vocabulary.
        - embedding_dim (int): Dimension of the token embeddings.
        - embedding_path (string): Path for retrieving a pretreined embedding.
        - seed (int): Seed for reproducibility.
        - distr_param (float): Parameter of the zipf distribution.

        Returns:
        - X (numpy array): Input sequences with shape (num_samples, sequence_length).
        - y (numpy array): Target labels with shape (num_samples,).
        """

        # Set the seed for reproducibility
        rs = np.random.RandomState(seed)
        
        self.num_samples = num_samples  # number of samples
        self.seq_len = sequence_length  # length of the sequences
        self.vocab = vocab  # list of the vocabulary
        self.vocab_size = len(vocab)  # size of the vocabulary
        self.embedding_dim = embedding_dim  # dimension in the embedding space
        self.n_classes = self.vocab_size + 1  # GPT predicts a token from the vocabulary plus the <EOS>

        # Shuffle the order of the vocabulary
        rs.shuffle(self.vocab)

        if(embedding_path == None):
            # Build a random embedding
            embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)  # embedding layer
            embedding_matrix = embeddings.weight.data  # embedding matrix
        else:
            # Load pre-trained embedding matrix
            embedding_vectors = []
            with open(embedding_path, 'r', encoding = 'utf-8') as f:
                next(f)  # Skip the header or first line if any
                # Use the readlines() method to read all lines into a list
                lines = f.readlines()

                # Count the number of lines in the list
                num_rows = len(lines)

                step = num_rows // self.vocab_size
                for i, line in enumerate(lines):
                    if i >= self.vocab_size * step:  # Break if enough vectors are read
                        break
                    if i % step == 0:  # Only take every step-th vector
                        values = line.split()
                        vector = torch.tensor([float(val) for val in values[1:]])
                        embedding_vectors.append(vector)
                    
            embedding_matrix = embedding_vectors
        

        self.X = []  # List for the design matrix
        n_gen_seqs = 0  # Total Number of generated sequences

        self.y = []  # List for the labels

        # Process for standard positional encoding as Attention is All You Need
        max_pos = sequence_length
        position_enc = torch.tensor([[torch.sin(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) if i % 2 == 0 else torch.cos(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) for i in range(self.embedding_dim)] for pos in range(max_pos)], dtype=torch.float)

        
        stuck_limit = self.seq_len * 5  # Number of iterations that determines if the sequence is cursed, and hence should be dropped
        
        
        # Loop to build the num_samples sequences

        pbar = tqdm(total=self.num_samples, desc='Generating dataset', unit='sample')  # Initialize progress bar

        while(n_gen_seqs < self.num_samples):

            sequence = []  # Initialise the seq to an empty list
            length = 0  # Variable for the length of the sequence
            stuck = 0  # Counter that establish if the algorithm in stuck
            
            while(length < self.seq_len):  # While loop that generates the sequence
                
                
                # If it is the first token, then sample it uniformily from the vocabulary
                if length == 0:
                    
                    while True:
                        # Sample from the distribution
                        number = rs.zipf(distr_param)

                        # Check if the number is within the range
                        if number < self.vocab_size:
                            break  # Exit the loop if the number is within the range
                    
                    sequence.append(self.vocab[number - 1])
                    length += 1
                    

                else:  #if it is another token then choose it from similarity
                    
                    # Combine token and positional embeddings 
                    combined_embedding = 0

                    for i in range(length - 1, max(length - context_window - 1, -1), -1):
                        token_index = self.vocab.index(sequence[i])
                        combined_embedding += embedding_matrix[token_index] + position_enc[i]

                            
                        
                    #combined_embedding = combined_embedding + position_enc[length - 1]

                    # Calculate similarity with previous tokens and select the most similar one
                    similarities = [torch.dot(embedding_matrix[k], combined_embedding) for k in range(self.vocab_size)]
                    similarities = torch.tensor([similarities])
                    _, next_token_i = torch.topk(similarities, self.vocab_size)
                
                
                    # Stochastic step
                    while True:
                        # Sample from the distribution
                        number = rs.zipf(distr_param)

                        # Check if the number is within the range
                        if number < self.vocab_size:
                            break  # Exit the loop if the number is within the range
                    
                                
                    next_token = self.vocab[next_token_i[0][number - 1]]
                        
                    
                    sequence.append(next_token)
                    length += 1
                            
                    
                                  
                stuck += 1
                
                # I assumed that if took more then stuck limit iterations to build the sequence, then the seq is cursed
                if(stuck == stuck_limit):
                    stuck = 0
                    sequence = []
                    length = 0
                    combined_embedding = 0
                    
                    
            # Check if the built sequence is already in the dataset
            if(n_gen_seqs != 0):
                is_in_matrix = sequence in self.X
            else:
                is_in_matrix = False  # If it is the first sequence add it in X
            
            
            # If the generated sequence is not already present, build the padded seqs 
            if(not is_in_matrix):
                
                self.X.append(sequence)
                n_gen_seqs += 1
                pbar.update(1)
      
        pbar.close()
        print("\n")
                    
        # Build the target sequences
        for i in range(self.num_samples):
            
            labels = []
            
            label = self.X[i][0] + self.X[i][1]
            labels.append(label)
            
            for j in range(1, self.seq_len - 1):
                #label = self.X[i][j-1] + self.X[i][j] + self.X[i][j+1]
                label = self.X[i][j] + self.X[i][j+1]
                labels.append(label)

            #label = self.X[i][-2] + self.X[i][-1] + self.X[i][0]
            label = self.X[i][-1] + self.X[i][0]
            labels.append(label)
            
            self.y.append(labels) 
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        bound = 2 * np.mean(self.X)# + 2.5 * np.std(self.X)
        self.y = np.where(self.y >= bound, 1, 0)

        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype=torch.long), torch.tensor(self.y[idx],dtype=torch.long)
    
    
    
    
class PrimitiveNLP_NTP(Dataset):
    def __init__(self, num_samples, sequence_length, context_window, vocab, embedding_dim, embedding_path = None, seed = 42, distr_param = 1.1):

        """
        Generate a dataset for next token prediction that emulates natural language.

        Args:
        - num_samples (int): Number of samples to generate.
        - sequence_length (int): Length of each sequence.
        - context_window (int): Lenght of the context.
        - vocab (list): Vocabulary.
        - embedding_dim (int): Dimension of the token embeddings.
        - embedding_path (string): Path for retrieving a pretreined embedding.
        - seed (int): Seed for reproducibility.
        - distr_param (float): Parameter of the zipf distribution.

        Returns:
        - X (numpy array): Input sequences with shape (num_samples, sequence_length).
        - y (numpy array): Target labels with shape (num_samples,).
        """

        # Set the seed for reproducibility
        rs = np.random.RandomState(seed)
        
        self.num_samples = num_samples  # number of samples
        self.seq_len = sequence_length  # length of the sequences
        self.vocab = vocab  # list of the vocabulary
        self.vocab_size = len(vocab)  # size of the vocabulary
        self.embedding_dim = embedding_dim  # dimension in the embedding space
        self.n_classes = self.vocab_size + 1  # GPT predicts a token from the vocabulary plus the <EOS>

        # Shuffle the order of the vocabulary
        rs.shuffle(self.vocab)

        if(embedding_path == None):
            # Build a random embedding
            embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)  # embedding layer
            embedding_matrix = embeddings.weight.data  # embedding matrix
        else:
            # Load pre-trained embedding matrix
            embedding_vectors = []
            with open(embedding_path, 'r', encoding = 'utf-8') as f:
                next(f)  # Skip the header or first line if any
                # Use the readlines() method to read all lines into a list
                lines = f.readlines()

                # Count the number of lines in the list
                num_rows = len(lines)

                step = num_rows // self.vocab_size
                for i, line in enumerate(lines):
                    if i >= self.vocab_size * step:  # Break if enough vectors are read
                        break
                    if i % step == 0:  # Only take every step-th vector
                        values = line.split()
                        vector = torch.tensor([float(val) for val in values[1:]])
                        embedding_vectors.append(vector)
                    
            embedding_matrix = embedding_vectors
        

        self.X = []  # List for the design matrix
        n_gen_seqs = 0  # Total Number of generated sequences

        # Process for standard positional encoding as Attention is All You Need
        max_pos = sequence_length
        position_enc = torch.tensor([[torch.sin(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) if i % 2 == 0 else torch.cos(torch.tensor(pos / (10000 ** (i // 2 * 2.0 / self.embedding_dim)), dtype=torch.float)) for i in range(self.embedding_dim)] for pos in range(max_pos)], dtype=torch.float)

    
        stuck_limit = self.seq_len * 5  # Number of iterations that determines if the sequence is cursed, and hence should be dropped
        
        
        # Loop to build the num_samples sequences

        pbar = tqdm(total=self.num_samples, desc='Generating dataset', unit='sample')  # Initialize progress bar

        while(n_gen_seqs < self.num_samples):

            sequence = []  # Initialise the seq to an empty list
            length = 0  # Variable for the length of the sequence
            stuck = 0  # Counter that establish if the algorithm in stuck
            
            while(length < self.seq_len):  # While loop that generates the sequence
                
                
                # If it is the first token, then sample it uniformily from the vocabulary
                if length == 0:
                    
                    while True:
                        # Sample from the distribution
                        number = rs.zipf(distr_param)

                        # Check if the number is within the range
                        if number < self.vocab_size:
                            break  # Exit the loop if the number is within the range
                    
                    sequence.append(self.vocab[number - 1])
                    length += 1
                    

                else:  #if it is another token then choose it from similarity
                    
                    # Combine token and positional embeddings 
                    combined_embedding = 0

                    for i in range(length - 1, max(length - context_window - 1, -1), -1):
                        token_index = self.vocab.index(sequence[i])
                        combined_embedding += embedding_matrix[token_index] + position_enc[i]

                            
                        
                    #combined_embedding = combined_embedding + position_enc[length - 1]

                    # Calculate similarity with previous tokens and select the most similar one
                    similarities = [torch.dot(embedding_matrix[k], combined_embedding) for k in range(self.vocab_size)]
                    similarities = torch.tensor([similarities])
                    _, next_token_i = torch.topk(similarities, self.vocab_size)
                
                
                    # Stochastic step
                    while True:
                        # Sample from the distribution
                        number = rs.zipf(distr_param)

                        # Check if the number is within the range
                        if number < self.vocab_size:
                            break  # Exit the loop if the number is within the range
                    
                                
                    next_token = self.vocab[next_token_i[0][number - 1]]
                        
                    
                    sequence.append(next_token)
                    length += 1
                            
                    
                                  
                stuck += 1
                
                # I assumed that if took more then stuck limit iterations to build the sequence, then the seq is cursed
                if(stuck == stuck_limit):
                    stuck = 0
                    sequence = []
                    length = 0
                    combined_embedding = 0
                    
                    
            # Check if the built sequence is already in the dataset
            if(n_gen_seqs != 0):
                is_in_matrix = sequence in self.X
            else:
                is_in_matrix = False  # If it is the first sequence add it in X
            
            
            # If the generated sequence is not already present, build the padded seqs 
            if(not is_in_matrix):
                
                self.X.append(sequence)
                n_gen_seqs += 1
                pbar.update(1)

        pbar.close()
        print("\n")
        
        self.X = np.array(self.X)
        self.y = np.hstack((self.X[:, 1:], np.full((self.X[:, 1:].shape[0], 1), self.vocab_size)))  # shift target sequence to the right

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx],dtype = torch.long), torch.tensor(self.y[idx], dtype = torch.long)