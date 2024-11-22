# Import modules and libraries
import json
from nltk_funcs import clean_tokenize, bag_of_words, sentence_to_embedding
from nltk.corpus import stopwords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork
##import torchtext.vocab as vocab

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Initialize dataset based off of torch dataset
class ChatDataset(Dataset):
    # Constructor for samples and input/output vectors
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # Function to get an (input,output) pair
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Function to return the number of samples
    def __len__(self):
        return self.n_samples

# Open JSON file containing data, read through, and save data to dictionary
with open('data/movieReviewsTrain.json', 'r') as f:
    data = json.load(f)

# Cleaning, tokenizing, lemmatizing, and creating vocab
all_words = []
tags = []
pattern_tags = []

for data_line in data['data']:
    tag = data_line['tag']
    tags.append(tag)
    for pattern in data_line['patterns']:
        words = clean_tokenize(pattern)
        all_words.extend(words)
        pattern_tags.append((words, tag))

all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Generate embeddings if using GloVe
##glove = vocab.GloVe(name='6B', dim=100)
##x_train = [sentence_to_embedding(" ".join(pattern_sentence), glove, 100) for pattern_sentence, _ in pattern_tags]
##x_train = np.array(x_train)
##y_train = np.array([tags.index(tag) for _, tag in pattern_tags])


# Initialize input/output vectors
x_train = []
y_train = []

for (pattern_sentence, tag) in pattern_tags:
    # Generate bag of words with the cleaned vocabulary
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    # Map tag to an index in `tags`
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


# Initialize super parameters
batch_size = 3
output_size = len(tags)
input_size = len(all_words)
learning_rate = 0.005
num_epochs = 1000

# Create dataset for chatbot
dataset = ChatDataset()
# List of tuples containing data for chatbot
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)

# Initialize the updated NeuralNetwork model
model = NeuralNetwork(input_size, hidden_size=64, output_size=len(tags))
# OR, if using LSTM
# model = NeuralNetworkLSTM(input_size, hidden_size=64, output_size=len(tags), num_layers=2, dropout_prob=0.5)


# Initialize variable for Cross Entropy Loss and Softmax functions to be used for error calculations
cross_entropy = nn.CrossEntropyLoss()
# Set up process for updating weights/gradient
optimizer = torch.optim.Adam(model.parameters(),learning_rate)

# For an epoch
for epoch in range(num_epochs):
    # For each training pair
    for (words, labels) in train_loader:
        outputs = model(words) # Feedforward output (y_predicted)
        loss = cross_entropy(outputs, labels) # Calculate cross entropy loss ((y_predicted - y)**2).mean(), where y_predicted is the expected output and y is actual output, and softmax of output
        optimizer.zero_grad() # Clears gradient for this iteration
        loss.backward() # Calculates gradient using chain rule
        optimizer.step() # Updates weights based on back propagation, ready for next epoch
    
    # Prints out current Cross Entropy Loss every 100 epochs for user
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.10f}')

print(f'Final loss, loss = {loss.item():.10f}')

# Initialize data dictionary to save for running chatbot application
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": 64,         # This should match the value used in model initialization
    "all_words": all_words,
    "tags": tags
}

# Save data to .pth file for later use
FILE = "model.pth"
torch.save(data, FILE)

print(f'Training complete, file saved to {FILE}')
