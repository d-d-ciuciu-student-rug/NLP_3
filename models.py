
from torch import (Tensor,
                   relu as torch_relu,
                   max as torch_max,
                   cat as torch_cat,
                   )
from torch.nn import (Module, Embedding, Dropout,
                      ModuleList, Linear, Conv1d,
                      LSTM,
                      )
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
                 num_classes: int,
                 ) -> None:

        super().__init__()

        # Layer-by-layer construction of the model.
        self.embedding: Embedding = Embedding(vocab_size,
                                              embed_dim,
                                              padding_idx = pad_idx)

        self.emb_dropout: Dropout = Dropout(dropout)

        self.convs: ModuleList = ModuleList([Conv1d(in_channels = embed_dim,
                                                    out_channels = num_filters,
                                                    kernel_size = kernel_size)
                                             for kernel_size in kernel_sizes])

        self.rep_dropout: Dropout = Dropout(dropout)

        self.fc: Linear = Linear(num_filters * len(kernel_sizes),
                                 num_classes)


    def forward(self,
                x: Tensor,
                lengths: Tensor
                ) -> Tensor:

        emb: Tensor = self.emb_dropout(self.embedding(x))
        emb_t: Tensor = emb.transpose(1, 2)

        pooled: list[Tensor] = []

        for conv in self.convs:
            z: Tensor = torch_relu(conv(emb_t))
            p: Tensor = torch_max(z, dim = 2).values

            pooled.append(p)

        rep: Tensor = torch_cat(pooled, dim = 1)
        rep = self.rep_dropout(rep)

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

        # Layer-by-layer construction of the model.
        self.embedding: Embedding = Embedding(vocab_size,
                                              embed_dim,
                                              padding_idx = pad_idx)

        self.emb_dropout: Dropout = Dropout(dropout)

        self.lstm: LSTM = LSTM(input_size = embed_dim,
                               hidden_size = hidden_dim,
                               num_layers = num_layers,
                               batch_first = True,
                               dropout = dropout if num_layers > 1 else 0.0,
                               bidirectional = bidirectional,
                               )

        self.rep_dropout: Dropout = Dropout(dropout)

        rep_dim: int = hidden_dim * (2 if bidirectional else 1)
        self.fc = Linear(rep_dim, num_classes)


    def forward(self,
                x: Tensor,
                lengths: Tensor
                ) -> Tensor:

        emb: Tensor = self.emb_dropout(self.embedding(x))

        packed: PackedSequence = pack_padded_sequence(emb,
                                                      #lengths.cpu(),
                                                      lengths,
                                                      batch_first = True,
                                                      enforce_sorted = False)
        output: PackedSequence
        h_n: Tensor
        c_n: Tensor
        output, (h_n, c_n) = self.lstm(packed)

        h_last: Tensor
        if self.lstm.bidirectional:
            h_last = torch_cat((h_n[-2, :, :], h_n[-1, :, :]), dim = 1)
        else:
            h_last = h_n[-1]

        rep: Tensor = self.rep_dropout(h_last)

        return self.fc(rep)


    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
