import torch
from torch import nn
from torchvision import models

class EncoderCNN(nn.Module):
    def __init__(self,embed_size,train_CNN=False):
        super(EncoderCNN,self).__init__()
        self.train_CNN=train_CNN
        self.inception=models.inception_v3(pretrained=True,aux_logits=False)
        self.inception.fc=nn.Linear(self.inception.fc.in_features,embed_size)
        self.relu=nn.ReLU()
        self.times=[]
        self.dropout=nn.Dropout(p=0.5)
    def forward(self,images):
        features=self.inception(images)
        return self.dropout(self.relu(features))
    
class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(DecoderRNN,self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers)
        self.linear=nn.Linear(hidden_size,vocab_size)
        self.dropout=nn.Dropout(p=0.5)
    def forward(self,features,captions):
        embedding=self.dropout(self.embed(captions))
        embedding=torch.cat((features.unsqueeze(0),embedding),dim=0)
        hiddens,_=self.lstm(embedding)
        outputs=self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(CNNtoRNN,self).__init__()
        self.encoder=EncoderCNN(embed_size)
        self.decoder=DecoderRNN(embed_size,hidden_size,vocab_size,num_layers)
    def forward(self,images,captions):
        features=self.encoder(images)
        outputs=self.decoder(features,captions)
        return outputs
    def caption_image(self,image,vocablulary,max_length=50):
        result_caption=[]
        with torch.no_grad():
            x=self.encoder(image).unsqueeze(0)
            states=None
            for _ in range(max_length):
                hiddens,states=self.decoder.lstm(x,states)
                output=self.decoder.linear(hiddens.squeeze(0))
                predicted=output.argmax(1)
                result_caption.append(predicted.item())
                x=self.decoder.embed(predicted).unsqueeze(0)
                if vocablulary.itos[predicted.item()]=='<EOS>':
                    break
        return [vocablulary.itos[idx] for idx in result_caption]
