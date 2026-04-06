
from torch import (Tensor,
                   relu as torch_relu,
                   max as torch_max,
                   cat as torch_cat,
                   )
from torch.nn import (Module, Embedding, Dropout,
                      ModuleList, Linear, Conv1d,
                      LSTM
                      )
import torch
from torch.nn.utils.rnn import (PackedSequence,
                                pack_padded_sequence,
                                )

class CNNTextClassifier(Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_filters: int,
                 kernel_sizes: tuple[int, ...],
                 dropout: float,
                 pad_idx: int,
                 num_classes: int) -> None:

        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = Dropout(dropout)
        self.convs = ModuleList([Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.rep_dropout = Dropout(dropout)
        self.fc = Linear(num_filters * len(kernel_sizes), num_classes)
        self.num_classes = num_classes

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        emb = self.emb_dropout(self.embedding(x)).transpose(1, 2)
        pooled = [torch.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        rep = self.rep_dropout(torch.cat(pooled, dim=1))
        return self.fc(rep)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMTextClassifier(Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float,
                 pad_idx: int,
                 num_classes: int,
                 bidirectional: bool,
                 ) -> None:

        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = Dropout(dropout)
        self.lstm = LSTM(input_size=embed_dim,
                         hidden_size=hidden_dim,
                         num_layers=num_layers,
                         batch_first=True,
                         dropout=dropout if num_layers > 1 else 0.0,
                         bidirectional=bidirectional)
        self.rep_dropout = Dropout(dropout)
        rep_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = Linear(rep_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        emb = self.emb_dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(packed)

        if self.lstm.bidirectional:
            h_last = torch_cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            h_last = h_n[-1]

        rep = self.rep_dropout(h_last)
        return self.fc(rep)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
