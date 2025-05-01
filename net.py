# https://github.com/LaurentVeyssier/Image-Captioning-Project-with-full-Encoder-Decoder-model/tree/master
# Credit: Laurent Veyssier

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)  # added Batchnorm

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)  # added Batchnorm
        return features


class ResNetEncoderPatch(nn.Module):
    def __init__(self, embed_size):
        super(ResNetEncoderPatch, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        # self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)  # added Batchnorm
        self.conv = nn.Conv2d(2048, embed_size, 1)  # project to embed_size

    def forward(self, images):
        features = self.backbone(images)  # [B, 2048, H/32, W/32]
        features = self.conv(features)  # [B, d_model, H/32, W/32]
        B, C, H, W = features.shape
        features = features.view(B, C, H * W)  # [B, d_model, num_patches]
        features = features.permute(0, 2, 1)  # [B, num_patches, d_model]
        return features


class CaptionDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers=6, nhead=8, dropout=0.1, max_len=50):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(max_len, embed_size)
        self.max_len = max_len

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embed_size, vocab_size)

    def forward(self, encoder_output, tgt_tokens):
        B, T = tgt_tokens.shape
        positions = torch.arange(0, T, device=tgt_tokens.device).unsqueeze(0).expand(B, T)
        tgt = self.token_embed(tgt_tokens) + self.pos_embed(positions)

        # Mask: causal mask so decoder only sees past tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)

        out = self.decoder(
            tgt=tgt,
            memory=encoder_output,
            tgt_mask=tgt_mask
        )
        return self.output(out)

    def sample(self, encoder_output, bos_token_id=1, eos_token_id=2):
        device = encoder_output.device
        generated = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)  # [1, 1]

        for _ in range(self.max_len - 1):
            pos = torch.arange(generated.size(1), device=device).unsqueeze(0)  # [1, T]
            tgt = self.token_embed(generated) + self.pos_embed(pos)  # [1, T, d_model]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)

            out = self.decoder(
                tgt=tgt,
                memory=encoder_output,
                tgt_mask=tgt_mask
            )  # [1, T, d_model]

            next_token_logits = self.output(out[:, -1, :])  # [1, vocab_size]
            # probs = torch.softmax(next_token_logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1).item()
            next_token = next_token_logits.argmax(dim=-1).item()  # scalar int

            generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=1)

            if next_token == eos_token_id:
                break

        return generated.squeeze(0).tolist()  # [seq_len]


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, device="cpu"):
        super(DecoderRNN, self).__init__()

        # Decoder parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Decoder layers definition
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Add drop out in case of multiple LSTM layers
        if num_layers > 1:
            self.lstm = nn.LSTM(
                embed_size, hidden_size, num_layers, dropout=0.5, batch_first=True
            )
        else:
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # Drop out layer
        self.dropout = nn.Dropout(0.5)

        # Final layer to output prediction vector accross the vocab space
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Initialize decoder layers using Xavier's initialization
        self.model_init()

        # To provide lisibility, print tensor dimensions when instantiating the model
        self.print_tensor_dimensions = True

        # init device
        self.device = device

    def forward(self, features, captions):

        # Decoder forward useful parameters
        self.batch_size = features.shape[0]
        self.seq_length = captions.shape[1]

        # Initializing the hidden and cell states and flushing out previous hidden states
        # We don't want the previous batch to influence the output of next image-caption input.
        self.hidden = self.init_hidden(self.batch_size)

        # Remove ending token since we will not require a next word from <end>
        if self.print_tensor_dimensions:
            print("Captions dimensions before removing <END> token: ", captions.shape)
        captions = captions[:, :-1]  # shape (batch_size, caption_length - 1)
        if self.print_tensor_dimensions:
            print("Captions dimensions after removing <END> token: ", captions.shape)

        # we embed all words into a embed_size space. Each word gets represented by a vector of embed_size
        # Shape of captions - from : batch_size x seq_length (of words)
        # to ==> batch_size x seq_length x word_embedding_dimension (embed_size)
        captions = captions.long()  # ensure captions are of type long
        embeddings = self.embedding(captions)
        if self.print_tensor_dimensions:
            print("Captions dimensions after embedding : ", embeddings.shape)

        # we prepare features tensor into appropriate format: (batchsize x embed_size) ==> (batchsize x 1 x embed_size)
        if self.print_tensor_dimensions:
            print("features dimensions in: ", features.shape)
        features = features.view(self.batch_size, 1, self.embed_size)
        if self.print_tensor_dimensions:
            print("features dim after expanding: ", features.shape)

        # We use similar embedding size for the features and the captions so we can concatenate both
        # The feature vector will be sumitted first to the RNN followed by the sequence of words
        # Tensor shape: batchsize x caption_length x embed_size
        embeddings = torch.cat((features, embeddings), dim=1)
        if self.print_tensor_dimensions:
            print(
                "Dimensions of combined (features, captions) inputs to LSTM : ",
                embeddings.shape,
            )

        # Input is passed on the the LSTM
        lstm_out, hidden = self.lstm(embeddings, self.hidden)
        lstm_out = self.dropout(lstm_out)

        # To chain multiple LSTM layers
        lstm_out = lstm_out.contiguous()

        # We ensure appropriate shape before input to the fully connected and final layer
        # Tensor shape: batchsize x caption_length x embed_size
        lstm_out = lstm_out.view(self.batch_size, self.seq_length, self.hidden_size)
        if self.print_tensor_dimensions:
            print("lstm output dimensions: ", lstm_out.shape)

        # We submit the output of the LSTM to the fully connected layer for predictions
        outputs = self.fc(lstm_out)
        if self.print_tensor_dimensions:
            print("Prediction output dimensions: ", outputs.shape)

        # Turn off printing dimensions after instantiation
        self.print_tensor_dimensions = False

        return outputs

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized randomly, for hidden state and cell state of LSTM

        hidden = (
            torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device),
            torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device),
        )

        return hidden

    def model_init(self):
        # We initialize the decoder parameters using Xavier's approach
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        pass

    def sample(self, inputs, states=None, max_len=20):
        "accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)"

        # Make sure input feature is of the expected shape by the LSTM
        inputs = inputs.view(1, 1, self.embed_size)

        # Initialize the caption output list
        caption = []

        # Initialize the hidden state
        if states is None:
            states = self.init_hidden(1)

        # Loop over the max length sequence to produce the caption
        for word in range(max_len):

            # Pass the input to the LSTM (first the extracted features then the previous word generated and previous hidden state)
            lstm_out, states = self.lstm(inputs, states)
            lstm_out = lstm_out.contiguous()

            # Obtain the prediction from the fully-connected prediction layer (shape 1 x 1 x vocab_size)
            word_pred = self.fc(lstm_out)
            # Extract the index determining the position of the predicted word in vocab
            prediction = word_pred.argmax(dim=2)
            # Append the word index to the caption list
            caption.append(prediction.item())
            # prepare next input to the LSTM : Embed the last word to predict the next
            inputs = self.embedding(prediction.view(1, -1)).view(1, 1, -1)

        return list(caption)
