import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

# Define the device to work on
if(torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'

#device = 'cuda'

######################################################
#
# DOTPRODUCTATTENTION
#
######################################################

class DotProductAttention(nn.Module):
    def __init__(self, model_dim, attention_input = 'both'):
        super(DotProductAttention, self).__init__()

        """
            Class for the dot product attention.

            Args:
            - model_dim (int): Hidden dimension.
            - attention_input (string): Wether the model should look at
            only the positional info, the semantic info or both.
        """
        
        # Check if the attention_input variable makes sense
        if attention_input not in ['both','only_sem','only_pos']:
            raise ValueError

        # Assign the arguments
        self.attention_input = attention_input
        self.model_dim = model_dim

        # Initialize the matrix that will mask pos or sem info
        self.F = torch.zeros(model_dim, model_dim, device = device)

        # Define the dimensions to look at
        if self.attention_input == 'both':
            a = model_dim
        elif self.attention_input == 'only_sem' or self.attention_input == 'only_pos':
            a = int(model_dim / 2)

        # Complete the definition of F
        with torch.no_grad():
            if self.attention_input in ['both', 'only_sem']:
                self.F[torch.arange(0, a), torch.arange(0, a)] = 1.0
            elif self.attention_input in ['both', 'only_pos']:
                self.F[torch.arange(a, 2 * a), torch.arange(a, 2 * a)] = 1.0

        # Define queriess, keys and values
        self.Q = nn.Parameter(torch.empty(model_dim, model_dim, device = device))
        self.K = nn.Parameter(torch.empty(model_dim, model_dim, device = device))
        self.V = nn.Parameter(torch.empty(model_dim, model_dim, device = device))

        # Initialize them with the kaiming mathod
        nn.init.kaiming_uniform_(self.Q.T, a = math.sqrt(5))
        nn.init.kaiming_uniform_(self.K.T, a = math.sqrt(5))
        nn.init.kaiming_uniform_(self.V.T, a = math.sqrt(5))


    def forward(self, x):

        """
            Function for the feed forward.

            Args:
            - x (Tensor): Batch of sequences.

            Returns:
            - x (Tensor): Batch of processed sequences
        """

        # Compute the Qs, Ks and Vs
        Qx = x @ self.F @ self.Q
        Kx = x @ self.F @ self.K
        Vx = x @ self.V

        # Compute the attention scores
        attn_scores = torch.matmul(Qx,Kx.transpose(-2,-1)) / math.sqrt(self.model_dim)

        # Create an upper triangular mask
        batch_size, seq_length, _ = attn_scores.size()
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal = 1).bool().to(device)
    
        # Fill masked positions with -inf
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
    
        # Compute the attention probabilities
        self.attn_probs = torch.softmax(attn_scores,dim=-1)
        
        # Compute the processed sequences
        x = torch.matmul(self.attn_probs, Vx)

        return x
    

######################################################
#
# LEARNEDPOSTIONALENCODING
#
######################################################
    
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        """
            Class for learning the positional encodings.

            Args:
            - d_model (int): Embedding dimension.
            - max_seq_length (int): The maximum lenght of a sequence.
        """

        # Define the positional encodings up to the maximum sequence length
        pe = torch.arange(0, max_seq_length)

        # Create an embedding layer for the positional encodings
        # Each position index is embedded into a d_model-dimensional vector
        self.embedding = nn.Embedding(max_seq_length, d_model)

        # Register the positional encodings as a buffer
        # This ensures that the positional encodings are moved with the module
        # and are part of its state, but they are not updated during training
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):

        """
            Function for the forward pass.

            Args:
            - x (Tensor): Batch of sequences.

            Returns: 
            - torch.tile(e, (x.shape[0], 1, 1)) (Tensor): Positional encodings for the input sequences.
        """

        # Retrieve the positional encodings for all positions in the sequence
        e = self.embedding(self.pe)

        # Tile the positional encodings to match the batch size of the input
        # This replicates the positional encodings for each example in the batch
        # It creates a tensor of shape (batch_size, max_seq_length, d_model)
        return torch.tile(e, (x.shape[0], 1, 1))
    
    
######################################################
#
# TRANSFORMERSEQ2SEQ
#
######################################################
    
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, attention_input):
        super(TransformerSeq2Seq, self).__init__()

        """
            Class for transformer model.

            Args:
            - vocab_size (int): Size of the vocabulary.
            - model_dim (int): Hidden dimension for the attention.
            - hidden_dimension_fc (int): Hidden dimension for the fc layer.
            - n_classes (int): Number of classes.
            - seq_len (int): Lenght of the sequence.
            - attention_input (string): Wether the model should look at
            only the positional info, the semantic info or both.
        """

        # Check if model dim is even
        # if it is not, then some passages 
        # in the forward pass are impossible
        if model_dim % 2 == 1:
            raise ValueError()

        # Assign the args
        self.model_dim = model_dim
        self.attention_input = attention_input
        embedding_dim = model_dim
        self.seq_len = seq_len

        # Create semantic and postional embeddings
        self.semantic_emb = nn.Embedding(vocab_size, embedding_dim) 
        self.positional_emb = LearnedPositionalEncoding(embedding_dim, seq_len)

        # Create the transformer architecture (A -> LN -> FC1 -> FC2)
        self.attention = DotProductAttention(model_dim, attention_input = attention_input)
        self.norm = nn.LayerNorm(model_dim)
        self.fc1 = nn.Linear(model_dim, hidden_dimension_fc)
        self.activ = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dimension_fc, n_classes)

    def forward(self, x): # Batch x Lenght

        """
            Function for the forward pass.

            Args:
            - x (Tensor): batch of sequences.

            Returns:
            - x (Tensor): new representation of the sequences.
        """

        # Maps input tokens to semantic embeddings
        x_sem = self.semantic_emb(x) # Batch x Lenght x d/2 or Batch x Lenght x d

        # Maps positions to positional embeddings
        x_pos = self.positional_emb(x)

        # Apply masking if attention_input is set to 'only_sem' or 'only_pos'
        if self.attention_input in ['only_sem', 'only_pos']:
            
            # Set zeros on the first self.model_dim / 2 entries
            x_sem[...,int(self.model_dim / 2):] = 0.0
            # Set zeros on the last self.model_dim / 2 entries
            x_pos[..., :int(self.model_dim / 2)] = 0.0

        # Combine semantic and positional embeddings
        x = x_sem + x_pos

        # Apply the transformer architecture
        a = self.attention(x)
        x = self.norm(a)
        x = self.fc2(self.activ(self.fc1(x)))

        return x

   
######################################################
#
# TRAIN
#
######################################################

def train(model, train_dataset, val_dataset, lr = 0.001, n_epochs = 100, n_classes = 10):


    """
        Function for training the model.

        Args:
        - model (nn.Module): Size of the vocabulary.
        - train_dataset (numpy ndarray): Hidden dimension for the attention.
        - val_dataset (numpy ndarray): Hidden dimension for the fc layer.
        - n_epochs (int): Number of epochs.
        - n_classes (int): Number of classes.

        Returns:
        - transformer (nn.Module): Trained model.
        - train_losses (list): List of training losses for each epoch.
        - val_losses (list): List of validation losses for each epoch.
        - val_acc (list): List of validation accuracies for each epoch.
    """

    # Learning rate
    lr = 0.001

    # Load the datasets
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 64, shuffle = True)

    # Define loss and optimisation method
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, betas = (0.9, 0.98), eps = 1e-9)

    # Define lists for the losses on training and validation set, and one for the accuracy on the validation set
    train_losses = []
    val_losses = []
    val_acc = []

    # Train the model for the specified number of epochs
    for epoch in range(n_epochs):

        # Set the loss for the epoch at an initial value of zero
        epoch_loss = 0.0
        train_losses.append(epoch_loss)
          
        # Evaluate on the test set every epoch
        with torch.no_grad():

            # Set the loss on the validation set, and the accuracy on an initial value of zero
            val_loss = 0.0
            acc = 0.0

            # Evaluate the model on the validation set
            for X, y in val_dataloader:

                # Set the devide for the dataset
                X = X.to(device)
                y = y.to(device)

                # Compute the result of the forward process for a batch
                output = model(X)

                # Compute the prediction
                pred = output.argmax(axis = -1)

                # Compute the loss and the accuracy
                loss = criterion(output.view(-1, n_classes), y.view(-1))
                acc += torch.mean((pred.view(-1) == y.view(-1)).float()).item()
                val_loss += loss.item()

            # Normalize the loss and accuracy for the number of data points, and store their values
            val_loss /= len(val_dataloader)
            acc /= len(val_dataloader)

            val_losses.append(val_loss)
            val_acc.append(acc)

        # Train the model
        for X, y in train_dataloader:

            # Set the device for the datasets
            X = X.to(device)
            y = y.to(device)

            # Set the gradient to None for each pass
            optimizer.zero_grad()

            # Compute the output of the model for a batch
            output = model(X)

            # Compute the loss
            loss = criterion(output.contiguous().view(-1, n_classes), y.contiguous().view(-1))
            epoch_loss += loss.item()

            # Backprop
            loss.backward()  # Autograd
            optimizer.step()  # Weights's update

        # Normalize the loss over the number of datapoints and store it
        epoch_loss /= len(train_dataloader)
        train_losses.append(epoch_loss)
          
        # Print the progress
        if(epoch % 10 == 0):
            print(f'[Epoch {epoch:02}] Train loss = {epoch_loss:.5f} :: Val loss {val_loss:.5f} :: Val accuracy {acc * 100:.2f}')

            
    return model, train_losses, val_losses, val_acc


######################################################
#
# REPARAMETRIZE
#
######################################################

def reparameterize(orig_transformer, vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len):

    """
        Function for reparameterizing a transformer model.

        Args:
        - orig_transformer (nn.Module): Original transformer model to reparameterize.
        - vocab_size (int): Size of the vocabulary.
        - model_dim (int): Dimensionality of the model.
        - hidden_dimension_fc (int): Hidden dimension for the fc layer.
        - n_classes (int): Number of classes.
        - seq_len (int): Length of input sequences.

        Returns:
        - new_transformer (nn.Module): Reparameterized transformer model.
    """

    # Don't compute the gradients for this process
    with torch.no_grad():

        # Copy the state dictionary of the original transformer
        a = orig_transformer.state_dict()

        # Create a new transformer model with the same architecture as the original
        new_transformer = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, orig_transformer.attention_input)
        new_transformer.load_state_dict(a)

        # Reparameterize the attention mechanism
        new_transformer.attention.Q.data = new_transformer.attention.F @ new_transformer.attention.Q
        new_transformer.attention.K.data = new_transformer.attention.F @ new_transformer.attention.K

        # Zero out the second half of the semantic embedding weights
        new_transformer.semantic_emb.weight[...,int(model_dim/2):] = 0.0

        # Zero out the first half of the positional embedding weights
        new_transformer.positional_emb.embedding.weight[...,:int(model_dim/2)] = 0.0

        # Add small random noise to Q, K, and the embedding weights
        new_transformer.attention.Q.data += 0.001 * torch.randn_like(new_transformer.attention.Q.data)
        new_transformer.attention.K.data += 0.001 * torch.randn_like(new_transformer.attention.K.data)
        new_transformer.semantic_emb.weight.data += 0.001 * torch.randn_like(new_transformer.semantic_emb.weight.data)
        new_transformer.positional_emb.embedding.weight.data += 0.001 * torch.randn_like(new_transformer.positional_emb.embedding.weight.data)

        # Copy the state dictionary of the new transformer
        a = new_transformer.state_dict()

        # Create a new transformer model with the same architecture as the original
        # and set attention input to 'both'
        new_transformer = TransformerSeq2Seq(vocab_size, model_dim, hidden_dimension_fc, n_classes, seq_len, 'both')
        new_transformer.load_state_dict(a)

    return new_transformer


######################################################
#
# TRAIN LOCAL
#
######################################################

def train_local(transformer, train_dataset, val_dataset, n_epochs, n_classes):

    """
        Function for training the model to test the stability of the solution.

        Args:
        - transformer (nn.Module): Model to be trained.
        - train_dataset (DataLoader): DataLoader containing the training dataset.
        - val_dataset (DataLoader): DataLoader containing the validation dataset.
        - n_epochs (int): Number of epochs for training.
        - n_classes (int): Number of classes.

        Returns:
        - transformer (nn.Module): Trained model.
        - train_losses (list): List of training losses for each epoch.
        - val_losses (list): List of validation losses for each epoch.
        - val_acc (list): List of validation accuracies for each epoch.
    """

    # Learning rate
    lr = 0.001

    # DataLoader for training and validation datasets
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 64, shuffle = True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(transformer.parameters(), lr=lr)
    optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    # Lists to store losses and accuracies
    train_losses = []
    val_losses = []
    val_acc = []

    # Train loop
    for epoch in range(n_epochs):

        # Initialize epoch loss
        epoch_loss = 0.0
        train_losses.append(epoch_loss)
        
        # Iterate over training dataset
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            output = transformer(X)
            loss = criterion(output.contiguous().view(-1, n_classes), y.contiguous().view(-1))

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        # Normalize epoch loss
        epoch_loss /= len(train_dataloader)
        train_losses.append(epoch_loss)

        # Validation
        with torch.no_grad():

            val_loss = 0.0
            acc = 0.0

            for X, y in val_dataloader:

                X = X.to(device)
                y = y.to(device)

                output = transformer(X)

                pred = output.argmax(axis = -1)
                loss = criterion(output.view(-1, n_classes), y.view(-1))

                acc += torch.mean((pred.view(-1) == y.view(-1)).float()).item()
                val_loss += loss.item()

            val_loss /= len(val_dataloader)
            acc /= len(val_dataloader)

            val_losses.append(val_loss)
            val_acc.append(acc)

        # Print progress
        if(epoch % round((n_epochs * 0.1)) == 0):
            print(f'[Epoch {epoch:02}] Train loss = {epoch_loss:.5f} :: Val loss {val_loss:.5f} :: Val accuracy {acc*100:.2f}')

    return transformer, train_losses, val_losses, val_acc
