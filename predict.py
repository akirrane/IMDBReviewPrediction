# Import modules and libraries
import random
import json
import torch
from model import NeuralNetwork
from nltk_funcs import clean_tokenize, bag_of_words

# Open data file, save data to dictionary
with open('data/movieReviewsTest.json', 'r') as f:
    data = json.load(f)

# Load the trained model parameters from the file
FILE = "model.pth"
net = torch.load(FILE)

# Retrieve the trained model parameters
input_size = net["input_size"]
hidden_size = net["hidden_size"]
output_size = net["output_size"]
all_words = net["all_words"]
tags = net["tags"]
model_state = net["model_state"]

# Initialize the model with updated parameters
model = NeuralNetwork(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Function for conversation with chatbot
def predict_response(msg):
    sentence = clean_tokenize(msg) # Tokenize user input
    x = bag_of_words(sentence, all_words) # Create a bag of words with user input
    x = x.reshape(1, x.shape[0]) # Reshape to one row
    x = torch.from_numpy(x) # Convert numpy array to torch tensor

    output = model(x) # Feed user input data through model
    _, prediction = torch.max(output, dim = 1) # Returns prediction 
    tag = tags[prediction.item()] # Return predicted tag

    # Calculate probability of output being correct
    probs = torch.softmax(output, dim = 1)
    prob = probs[0][prediction.item()] 

    if prob.item() > 0.60: # Threshold of 75%, else bot does not understand
        # For tag in data dictionary
        for keyword in data["data"]:
            # If tag matches predicted tag
            if tag == keyword["tag"]:
                return random.choice(keyword['responses']) # Return a random response from the list of responses

    return "I'm not sure if this is positive or negative..." # If under 75% probablility, print "I do not understand..."
