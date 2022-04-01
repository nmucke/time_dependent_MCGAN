import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                    -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Example(nn.Module):
    def __init__(self, vocab_size, feature_size):
        super(Example, self).__init__()
        self.embedding = nn.Embedding(vocab_size, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        self.layers = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8)
        self.transformer = nn.TransformerEncoder(self.layers, num_layers=6)
        self.decoder = nn.Linear(feature_size, vocab_size)

    def forward(self, x):
        # X shape: [seq_len, batch_size]
        print("Input size [seq_len, batch_size]")
        print(x.shape)
        x = self.embedding(x)
        print("Embedding size [seq_len, batch_size, feature_size]")
        print(x.shape)
        x = self.pos_encoder(x)
        # X shape: [seq_len, batch_size, feature_size]
        print(x.shape)
        x = self.transformer(x)
        # X shape: [seq_len, batch_size, feature_size]
        print(x.shape)
        x = self.decoder(x)
        # X shape: [seq_len, batc_size, vocab_size]
        print(x.shape)
        return x


ntokens = 10000
data = torch.arange(0, ntokens, 1)

feature_size = 30
sequence_size = 100
src = torch.as_strided(data, (sequence_size, feature_size), (1, 1))
tgt = torch.as_strided(data[1:], (sequence_size, feature_size), (1, 1))
print(src.shape)
print(tgt.shape)

model = Example(ntokens, 512)

# Hello world: A=0, Z=25
# src = torch.LongTensor([[7,4,11,11,14,22,14,17,11]]).view(9,1) #helloworl
# tgt = torch.LongTensor([[4,11,11,14,22,14,17,11,3]]).view(9,1) #elloworld

# 1234
# src = torch.LongTensor([[1,2,3,4,5,6,7,8,9]]).view(9,1)
# tgt = torch.LongTensor([[2,3,4,5,6,7,8,9,10]]).view(9,1)

criterion = nn.CrossEntropyLoss()
lr = 1.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(100):
    out = model(src)
    outv = out.view(-1, ntokens)

    print(outv)
    print(outv.shape)

    optimizer.zero_grad()
    loss = criterion(out.view(-1, ntokens), tgt.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
out = model(src)
print(out.shape)
outvals = torch.argmax(out, 2)
print(outvals)
print(outvals.shape)

test1 = torch.arange(0, 30, 1).unsqueeze(1)
test2 = torch.arange(500, 600, 1).unsqueeze(1)
test3 = torch.arange(10, 50, 1).unsqueeze(1)

print(test1.shape)
print(test2.shape)
print(test3.shape)

testout1 = model(test1)
testout2 = model(test2)
testout3 = model(test3)

print(testout1.shape)
print(testout2.shape)
print(testout3.shape)
testout1 = torch.argmax(testout1, 2)
testout2 = torch.argmax(testout2, 2)
testout3 = torch.argmax(testout3, 2)

print(testout1.shape)
print(testout2.shape)
print(testout3.shape)

print(testout1)
print(testout2)
print(testout3)