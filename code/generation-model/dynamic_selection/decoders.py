from codecs import EncodedFile
import torch.nn as nn
from torch import Tensor
import torch
from helpers import freeze_params, ConfigurationError, subsequent_mask, uneven_subsequent_mask
from transformer_layers import PositionalEncoding, \
    TransformerDecoderLayer, TransformerEncoderLayer
import random


class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size


class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 trg_size: int = 97,
                 decoder_trg_trg_: bool = True,
                 **kwargs):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(TransformerDecoder, self).__init__()

        self._hidden_size = hidden_size

        # Dynamic output size depending on the target size
        self._output_size = trg_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([TransformerDecoderLayer(
            size=hidden_size, ff_size=ff_size, num_heads=num_heads,
            dropout=dropout, decoder_trg_trg=decoder_trg_trg_) for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # Output layer to be the size of joints vector + 1 for counter (total is trg_size)
        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(self,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                **kwargs):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        # add position encoding to word embedding
        x = self.pe(trg_embed)
        # Dropout if given
        x = self.emb_dropout(x)

        padding_mask = trg_mask
        # Create subsequent mask for decoding
        sub_mask = subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        # Apply each layer to the input
        for layer in self.layers:
            x = layer(x=x, memory=encoder_output,
                      src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)

        # Apply a layer normalisation
        x = self.layer_norm(x)
        # Output layer turns it back into vectors of size trg_size
        output = self.output_layer(x)

        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)


class DynamicTransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 trg_size: int = 97,
                 decoder_trg_trg_: bool = True,
                 **kwargs):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(DynamicTransformerDecoder, self).__init__()

        self._hidden_size = hidden_size

        # Dynamic output size depending on the target size
        self._output_size = trg_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([TransformerDecoderLayer(
            size=hidden_size, ff_size=ff_size, num_heads=num_heads,
            dropout=dropout, decoder_trg_trg=decoder_trg_trg_) for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.softmax = nn.Softmax(dim=2)
        # Output layer to be the size of joints vector + 1 for counter (total is trg_size)
        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size // 2, 1))

        if freeze:
            freeze_params(self)

    def forward(self,
                trg_embed: Tensor = None,
                encoder1_output: Tensor = None,
                encoder2_output: Tensor = None,
                src1_mask: Tensor = None,
                src2_mask: Tensor = None,
                trg_mask: Tensor = None,
                T=None,
                balance_weight=None,
                epoch_num=0,
                **kwargs):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        self.balance_weight = balance_weight
        self.T = T
        # add position encoding to word embedding
        x = self.pe(trg_embed)

        # Dropout if given
        x1 = self.emb_dropout(x)
        x2 = x1.clone()
        

        padding_mask = trg_mask
        # Create subsequent mask for decoding
        sub_mask = subsequent_mask(
            trg_embed.size(1)).type_as(trg_mask)

        # Apply each layer to the input
        for layer in self.layers:

            x1 = layer(x=x1, memory=encoder1_output,
                       src_mask=src1_mask, trg_mask=sub_mask, padding_mask=padding_mask)
            x2 = layer(x=x2, memory=encoder2_output,
                       src_mask=src2_mask, trg_mask=sub_mask, padding_mask=padding_mask)

        # Have three views and at each time step, do a weighted result.

        # G step:
        if epoch_num % 1 == 0:
            #print("update expert picker")
            p1 = self.mlp(x1)
            p2 = self.mlp(x2)

            Hw = torch.cat([p1, p2], dim=2)

            balance_weight = self.softmax(Hw)

            # # weighted.
            # all_attns = torch.cat(
            #     [x1.unsqueeze(2), x2.unsqueeze(2)], dim=2)
            # new_weight = balance_weight.unsqueeze(-1).expand_as(all_attns)
            # combined_x = all_attns.mul(new_weight)
            # combined_x = combined_x.sum(dim=2).squeeze(2)

            # # one hot.
            a, new_indexes = torch.max(balance_weight, dim=2)
            mask1 = new_indexes == 0
            mask1 = mask1.unsqueeze(-1).expand_as(x1)
            mask2 = new_indexes == 1
            
            mask2 = mask2.unsqueeze(-1).expand_as(x2)
            
            combined_x = x1 * mask1 + x2*mask2
        else:
            choice = random.choice([0, 1])
            combined_x = [x1, x2][choice]

        combined_x = self.layer_norm(combined_x)
        # Output layer turns it back into vectors of size trg_size
        output = self.output_layer(combined_x)

        return output, x1, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)
