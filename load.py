import json
import nltk
import os
import random
import re
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
nltk.download('wordnet')


def preprocess(message):
    """
    This function takes a string as input, then performs these operations: 
        - lowercase
        - remove URLs
        - remove ticker symbols 
        - removes punctuation
        - tokenize by splitting the string on whitespace 
        - removes any single character tokens
    
    Parameters
    ----------
        message : The text message to be preprocessed.
        
    Returns
    -------
        tokens: The preprocessed text into tokens.
    """ 
    #TODO: Implement 
    
    # Lowercase the twit message
    text = message.lower()
    
    # Replace URLs with a space in the message
    # Pattern thanks to Asad user
    # reply in https://stackoverflow.com/questions/6718633/python-regular-expression-again-match-url
    url_pat = re.compile('((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*')
    text = re.sub(url_pat, ' ', text)
    
    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub('\$[a-zA-Z0-9]+', ' ', text)
    
    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    text = re.sub('\@[a-zA-Z0-9]+', ' ', text)

    # Replace everything not a letter with a space
    #text = re.sub('(?:[^a-zA-Z0-9.\s]|[^[0-9]+[\.]*[0-9]+]|([\.]{2,}))', ' ', text)
    text = re.sub('[^a-z]', ' ', text)
    
    # Tokenize by splitting the string on whitespace into a list of words
    tokens = text.split()

    # Lemmatize words using the WordNetLemmatizer. You can ignore any word that is not longer than one character.
    wnl = nltk.stem.WordNetLemmatizer()
    tokens = [wnl.lemmatize(w) for w in tokens if len(w)  > 1]
    
    return tokens
def predict(text, model, vocab):
    """ 
    Make a prediction on a single sentence.

    Parameters
    ----------
        text : The string to make a prediction on.
        model : The model to use for making the prediction.
        vocab : Dictionary for word to word ids. The key is the word and the value is the word id.

    Returns
    -------
        pred : Prediction vector
    """    
    
    # TODO Implement
    
    tokens = preprocess(text)
    
    # Filter non-vocab words
    tokens = [word for word in tokens if word in vocab]
    # Convert words to ids
    tokens = [vocab[word] for word in tokens]
        
    # Adding a batch dimension
    text_input = torch.tensor(tokens).unsqueeze(1)
    # Get the NN output
    hidden = model.init_hidden(text_input.size(1))
    logps, _ = model.forward(text_input, hidden)
    # Take the exponent of the NN output to get a range of 0 to 1 for each label.
    pred = torch.round(logps.squeeze())
    prob = torch.exp(logps)
    #max_prob = prob.argmax(dim = 0)
    #predictions = max_prob.argmax(dim = 0)
    predictions = np.argmax(prob.detach().numpy())
    #print(f'pippo: {pippo}')
    #print(pred, prob, max_prob, predictions)
    # Taking the step towards an interpretation of the datas instedad of giving probability distribution 
    if(predictions.item() > 2):
        print("Positive review detected!")
    elif (predictions.item() < 2):
        print("Negative review detected.")      
    else:
        print("Neutral review detected.")      
    
    return prob.detach().numpy()

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        """
        Initialize the model by setting up the layers.
        
        Parameters
        ----------
            vocab_size : The vocabulary size.
            embed_size : The embedding layer size.
            lstm_size : The LSTM layer size.
            output_size : The output size.
            lstm_layers : The number of LSTM layers.
            dropout : The dropout probability.
        """
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        
        # TODO Implement

        # Setup embedding layer
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # Setup additional layers
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size = self.lstm_size, num_layers = self.lstm_layers, batch_first = False, dropout = self.dropout)
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc = nn.Linear(in_features = self.lstm_size, out_features = self.output_size)
        self.log_smax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        """ 
        Initializes hidden state
        
        Parameters
        ----------
            batch_size : The size of batches.
        
        Returns
        -------
            hidden_state
            
        """
        
        # TODO Implement 
        
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # I thought cuda was useful in this step
        #hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda(), weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_().cuda())
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(), weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
        return hidden


    def forward(self, nn_input, hidden_state):
        """
        Perform a forward pass of our model on nn_input.
        
        Parameters
        ----------
            nn_input : The batch of input to the NN.
            hidden_state : The LSTM hidden state.

        Returns
        -------
            logps: log softmax output
            hidden_state: The new hidden state.

        """
        
        # TODO Implement 
        embeds = self.embedding(nn_input.long())
        
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        
        lstm_out = lstm_out[-1, :, :]
        
        out = self.dropout(lstm_out)
        
        fully_connected = self.fc(out)
        
        logps = self.log_smax(fully_connected)
        
        return logps, hidden_state

# Here we have loaded in a model that trained over 3 epochs `rnn_20_epoch.net`
with open('vocab.json', 'rb') as f:
    vocab  = json.load(f)
with open('textsentiment_3_epoch.net', 'rb') as f:
    checkpoint = torch.load(f, map_location=torch.device('cpu'))
    
loaded = TextClassifier(checkpoint['vocab_size'], checkpoint['embed_size'], checkpoint['lstm_size'], checkpoint['output_size'], checkpoint['lstm_layers'])
loaded.load_state_dict(checkpoint['state_dict'])
loaded.eval()

text = "Google is working on self driving cars, I'm bullish on $goog"
#text = "Google is shit company they will never achieve nothing, I'm bearish on $goog"
print(text)
pred = predict(text, loaded, vocab)
print(f'The model predicts that for the tweet\n"{text}"\n the class is {np.argmax(pred)-2} in a range from -2 to 2\n with an accuracy of {np.max(pred)*100}%\n')

text = "Google is working on self driving cars, I'm bullish on $goog debt bankrupt"
print(text)
pred = predict(text, loaded, vocab)
print(f'The model predicts that for the tweet\n"{text}"\n the class is {np.argmax(pred)-2} in a range from -2 to 2\n with an accuracy of {np.max(pred)*100}%\n')

text = "I really think $tsla is overvalued and has a dumb CEO"
print(text)
pred = predict(text, loaded, vocab)
print(f'The model predicts that for the tweet\n"{text}"\n the class is {np.argmax(pred)-2} in a range from -2 to 2\n with an accuracy of {np.max(pred)*100}%\n')

'''
For GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.embedding.weight.data.uniform_(-1, 1)
model.to(device)
'''

