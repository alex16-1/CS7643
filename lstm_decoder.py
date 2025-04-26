import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super(LSTMDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        # Embed captions
        embeddings = self.dropout(self.embed(captions))

        # Concatenate image features with embedded captions
        # We need to expand features to match batch_size x 1 x embed_size
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings), dim=1)

        # Pass through LSTM
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs

    def sample(self, features, states=None, max_len=20):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            # Forward propagation
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            # Get predicted word id
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted)

            # Early stopping if EOS token is predicted
            if predicted.item() == 2:  # <EOS> token
                break

            # Prepare input for next time step
            inputs = self.embed(predicted).unsqueeze(1)

        return torch.stack(sampled_ids, dim=1)
