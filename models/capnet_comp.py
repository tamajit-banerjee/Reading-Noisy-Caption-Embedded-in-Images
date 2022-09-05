from models.lstm import LSTM
import torchvision
import torch

class PretrainedEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained = True)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1]) # This rmeoves the last layer of the ResNet
        self.linear = torch.nn.Linear(list(resnet.children())[-1].in_features, out_dim) 
        self.bn = torch.nn.BatchNorm1d(out_dim, momentum=0.01)
        self.relu = torch.nn.ReLU(inplace=True)

        
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
            x = self.linear(x.view((x.size(0), -1)))
            x = self.bn(self.relu(x))
        return x
    


class CapNetComp(torch.nn.Module):
    def __init__(self, embedding_dim=128, lstm_size=256, vocab_size=100, use_gpu=True):
        super().__init__()

        self.encoder = PretrainedEncoder(
            in_channels=3, 
            out_dim=embedding_dim,
        )
        self.encoder.eval()

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
        
        
        