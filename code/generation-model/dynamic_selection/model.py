# coding: utf-8
"""
Module to represents whole models
"""

import numpy as np
import torch.nn as nn
from torch import Tensor
import torch
from torch.nn.modules.normalization import LayerNorm

from initialization import initialize_model
from embeddings import Embeddings
from encoders import Encoder, TransformerEncoder
from decoders import Decoder, DynamicTransformerDecoder
from constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD
from search import greedy
from vocabulary import Vocabulary
from batch import Batch


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder1: Encoder,
                 encoder2: Encoder,
                 decoder: Decoder,
                 src1_embed: Embeddings,
                 src2_embed: Embeddings,
                 
                 trg_embed: Embeddings,
                 src1_vocab: Vocabulary,
                 src2_vocab: Vocabulary,
                 
                 trg_vocab: Vocabulary,
                 cfg: dict,
                 in_trg_size: int,
                 out_trg_size: int,
                 ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]

        self.src1_embed = src1_embed
        self.src2_embed = src2_embed
        
        self.trg_embed = trg_embed

        self.encoder1 = encoder1
        self.encoder2 = encoder2
        
        self.decoder = decoder
        self.src1_vocab = src1_vocab
        self.src2_vocab = src2_vocab
       
        self.trg_vocab = trg_vocab
        self.bos_index = self.src1_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src1_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src1_vocab.stoi[EOS_TOKEN]
        self.target_pad = TARGET_PAD

        self.use_cuda = cfg["training"]["use_cuda"]
        self.T = cfg['training']['T']
        self.balance_weght = cfg['training']['balance']

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        self.count_in = model_cfg.get("count_in", True)
        # Just Counter
        self.just_count_in = model_cfg.get("just_count_in", False)
        # Gaussian Noise
        self.gaussian_noise = model_cfg.get("gaussian_noise", False)
        # Gaussian Noise
        if self.gaussian_noise:
            self.noise_rate = model_cfg.get("noise_rate", 1.0)

        # Future Prediction - predict for this many frames in the future
        self.future_prediction = model_cfg.get("future_prediction", 0)

        # Weight for different views.
        self.w_proj_layer_norm = LayerNorm(512)
        self.w_proj = nn.Linear(512, 512)
        self.w_context_vector = nn.Linear(512, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
    # pylint: disable=arguments-differ

    def forward(self,
                src1: Tensor,
                src2: Tensor,
                
                trg_input: Tensor,
                src1_mask: Tensor,
                src2_mask: Tensor,
                
                src1_lengths: Tensor,
                src2_lengths: Tensor,
               
                epoch_num,
                trg_mask: Tensor = None,
                src1_input: Tensor = None,
                src2_input: Tensor = None,
                
                ) -> (
            Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src1: source input for text
        :param src2: source input for gloss
        :param trg_input: target input
        :param src1_mask: source mask for text
        :param src2_mask: source mask for gloss
        :param src1_lengths: length of source inputs for text
        :param src2_lengths: length of source inputs for gloss
        :param trg_mask: target mask
        :return: decoder outputs
        """

        # Encode the source sequence
        encoder1_output, encoder1_hidden = self.encode1(src=src1,
                                                        src_length=src1_lengths,
                                                        src_mask=src1_mask)
        encoder2_output, encoder2_hidden = self.encode2(src=src2,
                                                        src_length=src2_lengths,
                                                        src_mask=src2_mask)
        
        # # pooling to reduce the dimension to B x D
        # pool1 = torch.mean(encoder1_output, dim=1).unsqueeze(1)
        # pool2 = torch.mean(encoder2_output, dim=1).unsqueeze(1)
        # 
        # src_input = torch.cat([pool1, pool2, pool3], dim=1)

        # # Weights for different views. original code was from
        # # https://github.com/GT-SALT/Multi-View-Seq2Seq/blob/fe4d4d4c8dd25b5784d2bfea8fabbf494fb453f2/fairseq_multi_view/fairseq/models/bart/model.py#L282-L305.
        # Hw = torch.tanh(self.w_proj_layer_norm(self.w_proj(src_input)))
        # balance_weight = self.softmax(self.w_context_vector(Hw).squeeze(-1))
        # if self.balance_weght:
        #     balance_weight = balance_weight ** (1/self.T)
        #     balance_weight = balance_weight / \
        #         balance_weight.sum(dim=1, keepdim=True)
        # print(balance_weight)

        # TODO
        # Hadamard
        # a = encoder1_output.repeat(1, encoder2_output.shape[1], 1)
        # b = encoder2_output.repeat(1, encoder1_output.shape[1], 1)
        # encoder_output = torch.mul(a, b)
        # unroll_steps = trg_input.size(1)

        # Add gaussian noise to the target inputs, if in training
        if (self.gaussian_noise) and (self.training) and (self.out_stds is not None):

            # Create a normal distribution of random numbers between 0-1
            noise = trg_input.data.new(trg_input.size()).normal_(0, 1)
            # Zero out the noise over the counter
            noise[:, :, -1] = torch.zeros_like(noise[:, :, -1])

            # Need to add a zero on the end of
            if self.future_prediction != 0:
                self.out_stds = torch.cat((self.out_stds, torch.zeros_like(self.out_stds)))[
                    :trg_input.shape[-1]]

            # Need to multiply by the standard deviations
            noise = noise * self.out_stds

            # Add to trg_input multiplied by the noise rate
            trg_input = trg_input + self.noise_rate*noise

        # # Decode the target sequence
        # # TODO: I increased src_mask to have same shape as hadamard output
        # skel_out, dec_hidden, _, _ = self.decode(encoder_output=encoder_output,
        #                                          src_mask=src1_mask.repeat(
        #                                              1, 1, encoder2_output.shape[1]),
        #                                          trg_input=trg_input,
        #                                          trg_mask=trg_mask)
        skel_out, dec_hidden, _, _ = self.decode(encoder1_output=encoder1_output,
                                                 encoder2_output=encoder2_output,
                                                
                                                 src1_mask=src1_mask,
                                                 src2_mask=src2_mask,
                                                 
                                                 trg_input=trg_input,
                                                 trg_mask=trg_mask,
                                                 epoch_num=epoch_num)
        gloss_out = None

        return skel_out, gloss_out

    def encode1(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        # Encode an embedded source
        encode_output = self.encoder1(
            self.src1_embed(src), src_length, src_mask)

        return encode_output

    def encode2(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        # Encode an embedded source
        encode_output = self.encoder2(
            self.src2_embed(src), src_length, src_mask)

        return encode_output

    def encode3(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
            -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        # Encode an embedded source

        return encode_output

    def decode(self, encoder1_output: Tensor, encoder2_output: Tensor, 
               src1_mask: Tensor, src2_mask: Tensor,  trg_input: Tensor, epoch_num,
               trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder1_output: encoder states for attention computation
        :param encoder2_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        # Enbed the target using a linear layer
        trg_embed = self.trg_embed(trg_input)

        # Apply decoder to the embedded target
        decoder_output = self.decoder(trg_embed=trg_embed,
                                      encoder1_output=encoder1_output,
                                      encoder2_output=encoder2_output,
                                    
                                      src1_mask=src1_mask,
                                      src2_mask=src2_mask,
                                     
                                      trg_mask=trg_mask,
                                      T=self.T,
                                      balance_weight=self.balance_weght,
                                      epoch_num=epoch_num)

        return decoder_output

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module, epoch_num: int) \
            -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        skel_out, _ = self.forward(
            src1=batch.src1, src2=batch.src2,  trg_input=batch.trg_input,
            src1_mask=batch.src1_mask, src2_mask=batch.src2_mask, src1_lengths=batch.src1_lengths,
            src2_lengths=batch.src2_lengths, 
            trg_mask=batch.trg_mask, epoch_num=epoch_num)

        # compute batch loss using skel_out and the batch target
        batch_loss = loss_function(skel_out, batch.trg)

        # If gaussian noise, find the noise for the next epoch
        if self.gaussian_noise:
            # Calculate the difference between prediction and GT, to find STDs of error
            with torch.no_grad():
                noise = skel_out.detach() - batch.trg.detach()

            if self.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                noise = noise[:, :, :noise.shape[2] //
                              (self.future_prediction)]

        else:
            noise = None

        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss, noise

    def run_batch(self, batch: Batch, max_output_length: int,) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        # First encode the batch, as this can be done in all one go
        encoder1_output, encoder1_hidden = self.encode1(
            batch.src1, batch.src1_lengths,
            batch.src1_mask)
        encoder2_output, encoder2_hidden = self.encode2(
            batch.src2, batch.src2_lengths,
            batch.src2_mask)
        
        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(
                max(batch.src1_lengths.cpu().numpy()) * 1.5)

        # Then decode the batch separately, as needs to be done iteratively
        # greedy decoding

        stacked_output, stacked_attention_scores = greedy(
            encoder1_output=encoder1_output,
            encoder2_output=encoder2_output,
            
            src1_mask=batch.src1_mask,
            src2_mask=batch.src2_mask,
            
            embed=self.trg_embed,
            decoder=self.decoder,
            trg_input=batch.trg_input,
            model=self)

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed, self.trg_embed)


def build_model(cfg: dict = None,
                src1_vocab: Vocabulary = None,
                src2_vocab: Vocabulary = None,
               
                trg_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """

    full_cfg = cfg
    cfg = cfg["model"]

    src1_padding_idx = src1_vocab.stoi[PAD_TOKEN]
    src2_padding_idx = src2_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"] + 1
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"] + 1

    just_count_in = cfg.get("just_count_in", False)
    future_prediction = cfg.get("future_prediction", 0)

    #  Just count in limits the in target size to 1
    if just_count_in:
        in_trg_size = 1

    # Future Prediction increases the output target size
    if future_prediction != 0:
        # Times the trg_size (minus counter) by amount of predicted frames, and then add back counter
        out_trg_size = (out_trg_size - 1) * future_prediction + 1

    # Define source embedding (for text and gloss separately)
    src1_embed = Embeddings(
        **cfg["encoder1"]["embeddings"], vocab_size=len(src1_vocab),
        padding_idx=src1_padding_idx)
    src2_embed = Embeddings(
        **cfg["encoder2"]["embeddings"], vocab_size=len(src2_vocab),
        padding_idx=src2_padding_idx)
   

    # Define target linear
    # Linear layer replaces an embedding layer - as this takes in the joints size as opposed to a token
    # Because we have continuous coordinates we don't use embeddings
    trg_linear = nn.Linear(
        in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])

    # Encoder 1 for TEXT-------
    enc_dropout = cfg["encoder1"].get("dropout", 0.)  # Dropout
    enc_emb_dropout = cfg["encoder1"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder1"]["embeddings"]["embedding_dim"] == \
        cfg["encoder1"]["hidden_size"], \
        "for transformer, emb_size must be hidden_size"

    # Transformer Encoder 1 for TEXT
    encoder1 = TransformerEncoder(**cfg["encoder1"],
                                  emb_size=src1_embed.embedding_dim,
                                  emb_dropout=enc_emb_dropout)

    # Encoder 2 for GLOSS-------
    enc_dropout = cfg["encoder2"].get("dropout", 0.)  # Dropout
    enc_emb_dropout = cfg["encoder2"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder2"]["embeddings"]["embedding_dim"] == \
        cfg["encoder2"]["hidden_size"], \
        "for transformer, emb_size must be hidden_size"

    # Transformer Encoder 2 for GLOSS
    encoder2 = TransformerEncoder(**cfg["encoder2"],
                                  emb_size=src2_embed.embedding_dim,
                                  emb_dropout=enc_emb_dropout)


    # Decoder -------
    dec_dropout = cfg["decoder"].get("dropout", 0.)  # Dropout
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    decoder_trg_trg = cfg["decoder"].get("decoder_trg_trg", True)
    # Transformer Decoder
    decoder = DynamicTransformerDecoder(
        **cfg["decoder"], vocab_size=len(trg_vocab),
        emb_size=trg_linear.out_features, emb_dropout=dec_emb_dropout,
        trg_size=out_trg_size, decoder_trg_trg_=decoder_trg_trg)

    # Define the model
    model = Model(encoder1=encoder1,
                  encoder2=encoder2,
                  
                  decoder=decoder,
                  src1_embed=src1_embed,
                  src2_embed=src2_embed,
                 
                  trg_embed=trg_linear,
                  src1_vocab=src1_vocab,
                  src2_vocab=src2_vocab,
                  
                  trg_vocab=trg_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size)

    # Custom initialization of model parameters
    initialize_model(model, cfg, src1_padding_idx, trg_padding_idx)

    return model
