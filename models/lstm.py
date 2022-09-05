import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_layers=1, vocab_size=100, use_gpu=True, use_attention=True):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.use_gpu = use_gpu
        
        self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = True)
        
        self.fc = nn.fc(hidden_size, self.vocab_size)        
        self.embed = nn.Embedding(self.vocab_size, embed_dim)

        
    def forward(self, image_features, image_captions, lengths):
        image_features = image_features.unsqueeze(1)
        
        embedded_captions = self.embed(image_captions)

        embeddings = torch.cat((image_features, embedded_captions), 1)

        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted= False)

        lstm_outputs, _ = self.lstm(packed_seq)        
        decoder_outputs = self.fc(lstm_outputs.data) 
        
        return decoder_outputs

    def forward_test(self, features, hidden=None):
        if(hidden is not None):
            features = self.embed(features).unsqueeze(1)
        output, hidden = self.lstm(features, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden