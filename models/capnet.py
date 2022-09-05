import torch.nn as nn

from models.resnet import ResNet50
from models.lstm import LSTM

class CapNet(nn.Module):
    '''
        Main Image Captioning model.
        Encoder-Decoder style architecture
    '''
    def __init__(self, embedding_dim=128, lstm_size=256, vocab_size=100, use_gpu=True):
        super().__init__()
        
        self.encoder = ResNet50(
            in_channels=3, 
            out_dim=embedding_dim,
        )

        self.decoder = LSTM(
            embedding_dim,
            lstm_hidden_size=lstm_size,
            vocab_size=vocab_size,
            use_gpu=use_gpu,
            use_attention=True,
        )

    def forward(self, imgs, caps, lengths):

        embedding = self.encoder(imgs)

        pred_caps = self.decoder(embedding, caps, lengths)

        return pred_caps





