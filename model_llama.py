import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import sys
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append("transformers/src/") #put outside the file
from transformers.models.llama import LlamaModel, LlamaForCausalLM
from transformers.models.llama.configuration_llama import PretrainedConfig, LlamaConfig


class LlamaConfig_small(PretrainedConfig):
  model_type = "llama"
  keys_to_ignore_at_inference = ["past_key_values"]

  def __init__(
      self,
      vocab_size=50,
      hidden_size=1024,
      intermediate_size=1000,
      num_hidden_layers=8,
      num_attention_heads=8,
      num_key_value_heads=None,
      hidden_act="silu",
      max_position_embeddings=2048,
      initializer_range=0.02,
      rms_norm_eps=1e-6,
      use_cache=True,
      pad_token_id=None,
      bos_token_id=1,
      eos_token_id=2,
      pretraining_tp=1,
      tie_word_embeddings=False,
      rope_theta=10000.0,
      rope_scaling=None,
      attention_bias=False,
      **kwargs,
  ):
      self.vocab_size = vocab_size
      self.max_position_embeddings = max_position_embeddings
      self.hidden_size = hidden_size
      self.intermediate_size = intermediate_size
      self.num_hidden_layers = num_hidden_layers
      self.num_attention_heads = num_attention_heads

      # for backward compatibility
      if num_key_value_heads is None:
          num_key_value_heads = num_attention_heads

      self.num_key_value_heads = num_key_value_heads
      self.hidden_act = hidden_act
      self.initializer_range = initializer_range
      self.rms_norm_eps = rms_norm_eps
      self.pretraining_tp = pretraining_tp
      self.use_cache = use_cache
      self.rope_theta = rope_theta
      self.rope_scaling = rope_scaling
      self._rope_scaling_validation()
      self.attention_bias = attention_bias

      super().__init__(
          pad_token_id=pad_token_id,
          bos_token_id=bos_token_id,
          eos_token_id=eos_token_id,
          tie_word_embeddings=tie_word_embeddings,
          **kwargs,
      )
  def _rope_scaling_validation(self):
      """
      Validate the `rope_scaling` configuration.
      """
      if self.rope_scaling is None:
          return

      if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
          raise ValueError(
              "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
              f"got {self.rope_scaling}"
          )
      rope_scaling_type = self.rope_scaling.get("type", None)
      rope_scaling_factor = self.rope_scaling.get("factor", None)
      if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
          raise ValueError(
              f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
          )
      if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
          raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")

def describe_model(net):
    nparams = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if type(net) is LLAMA:
        print('\nLLAMA specs:')
        print(' nparams=',nparams)
        #print(' nlayers_encoder=',net.nlayers_encoder)
        # print(' nlayers_decoder=',net.nlayers_decoder)
        # print(' nhead=',net.nhead)
        # print(' hidden_size=',net.hidden_size)
        # print(' dim_feedforward=',net.dim_feedforward)
        # print(' act_feedforward=',net.act)
        # print(' dropout=',net.dropout_p)
        print(' ')
        print('')
    else:
        print('Network type ' + str(type(net)) + ' not found...')

class PositionalEncoding(nn.Module):
    #
    # Adds positional encoding to the token embeddings to introduce word order
    #
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000.) / emb_size) # size emb_size/2
        pos = torch.arange(0, maxlen).reshape(maxlen, 1) # maxlen x 1
        pos_embedding = torch.zeros((maxlen, emb_size)) # maxlen x emb_size
        pos_embedding[:, 0::2] = torch.sin(pos * den) # maxlen x emb_size/2
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2) # maxlen x 1 x emb_size
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        #  Input
        #    token_embedding: [seq_len, batch_size, embedding_dim] list of embedded tokens
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class LLAMA(nn.Module):
    #
    # Transformer trained for meta seq2seq learning
    #
    def __init__(self, hidden_size: int, input_size: int, PAD_idx_input: int, PAD_idx_output: int,
        nlayers_decoder: int=3, nhead: int=8,
        dropout_p: float=0.1, ff_mult: int=4, activation='gelu'):
        # #
        # # Input
        # #  hidden_size : embedding size
        # #  input_size  : number of input symbols
        # #  output_size : number of output symbols
        # #  PAD_idx_input : index of padding in input sequences
        # #  PAD_idx_output : index of padding in output sequences
        # #  nlayers_encoder : number of transformer encoder layers
        # #  nlayers_decoder : number of transformer decoder layers
        # #  nhead : number of heads for multi-head attention
        # #  dropout_p : dropout applied to symbol embeddings and transformer layers
        # #  ff_mult : multiplier for hidden size of feedforward network
        # #  activation: string either 'gelu' or 'relu'
        # #
        super(LLAMA, self).__init__()
        llama_config = LlamaConfig_small(
          vocab_size=input_size,
          hidden_size=128,
          intermediate_size=1000,
          num_hidden_layers=nlayers_decoder,
          num_attention_heads=nhead,
          num_key_value_heads=None,
          hidden_act="silu",
          max_position_embeddings=2048,
          initializer_range=0.02,
          rms_norm_eps=1e-6,
          use_cache=True,
          pad_token_id=input_size-1,
          bos_token_id=input_size - 3,  # TODO verify it is correct in datasets line 180
          eos_token_id=input_size - 2,
          pretraining_tp=1,
          tie_word_embeddings=False,
          rope_theta=10000.0,
          rope_scaling=None,
          attention_bias=False)
        # assert activation in ['gelu','relu']
        # self.hidden_size = hidden_size
        # self.input_size = input_size
        # self.output_size = output_size
        self.PAD_idx_input = PAD_idx_input
        self.PAD_idx_output = PAD_idx_output
        # self.nlayers_encoder = nlayers_encoder
        self.nlayers_decoder = nlayers_decoder
        self.nhead = nhead
        self.dropout_p = dropout_p
        # self.dim_feedforward = hidden_size*ff_mult
        # self.act = activation
        # self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=nlayers_encoder, num_decoder_layers=nlayers_decoder,
        #     dim_feedforward=self.dim_feedforward, dropout=dropout_p, batch_first=True, activation=activation)
        self.positional_encoding = PositionalEncoding(emb_size=hidden_size, dropout=dropout_p)
        self.input_embedding = nn.Embedding(input_size, hidden_size)
        self.output_embedding = nn.Embedding(input_size, hidden_size)
        #self.out = nn.Linear(hidden_size, output_size)
        self.llama = LlamaForCausalLM(llama_config)


    # def prep_encode(self, xq_context_padded):
    #     # Embed source sequences and make masks
    #     #
    #     # Input
    #     #  xq_context_padded : source sequences via token index # b*nq (batch_size) x maxlen_src
    #     xq_context_embed = self.input_embedding(xq_context_padded) # batch_size x maxlen_src x emb_size
    #
    #     # Add positional encoding to input embeddings
    #     src_embed = self.positional_encoding(xq_context_embed.transpose(0,1))
    #     src_embed = src_embed.transpose(0,1) # batch_size x maxlen_src x emb_size
    #
    #     # Create masks for padded source sequences
    #     src_padding_mask = xq_context_padded==self.PAD_idx_input # batch_size x  maxlen_src
    #         # value of True means ignore
    #     return src_embed, src_padding_mask

    def prep_decode(self, z_padded):
        # Embed target sequences and make masks
        #
        # Input
        #  z_padded : b*nq (batch_size) x maxlen_tgt
        #  z_lengths : b*nq list
        maxlen_tgt = z_padded.size(1)
        z_embed = self.input_embedding(z_padded) # batch_size x maxlen_tgt x emb_size

        # Add positional encoding to target embeddings
        tgt_embed = self.positional_encoding(z_embed.transpose(0,1))
        tgt_embed = tgt_embed.transpose(0,1) # batch_size x maxlen_tgt x emb_size

        # create mask for padded targets

        tgt_padding_mask = z_padded!=self.PAD_idx_input # batch_size x maxlen_tgt
        tgt_padding_mask = tgt_padding_mask.int()
            # in llama value of 0 means ignore

        # create diagonal mask for autoregressive control
        #tgt_mask = self.transformer.generate_square_subsequent_mask(maxlen_tgt) # maxlen_tgt x maxlen_tgt
        #tgt_mask = tgt_mask.to(device=DEVICE)
        return tgt_embed, tgt_padding_mask, #tgt_mask

    def forward(self, z_padded, batch):
        # Forward pass through encoder and decoder
        #
        # Input
        #  z_padded : tensor of size [b*nq (batch_size), maxlen_target] : decoder input via token index
        #  batch : struct via datasets.make_biml_batch(), which includes source sequences
        #
        # Output
        #   output : [b*nq x maxlen_target x output_size]
        # z_padded is shifted to the right by one
        if self.training:
            xy_support_query_padded = z_padded #batch['xy_support_query_padded'] # n_samples x sample_x_batch_ [maxlen_src]
        else:
            xy_support_query_padded = z_padded #batch['xy_support_xquery_padded']
        #src_embed, src_padding_mask = self.prep_decode(xy_support_query_padded)
        tgt_embed, tgt_padding_mask,  = self.prep_decode(z_padded) #always use input dict that is combined input+output+special symbols
        out = self.llama.forward(input_ids=xy_support_query_padded,
                                 attention_mask=tgt_padding_mask)
        # trans_out = self.transformer(src_embed, tgt_embed, tgt_mask=tgt_mask,
        #     src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask,
        #     memory_key_padding_mask=src_padding_mask)
        logits = out['logits'] # CausalLMOutputWithPast

        return logits

    def encode(self, batch):
        # Forward pass through encoder only
        #
        # Output
        #  memory : [b*nq (batch_size) x maxlen_src x hidden_size]
        #  memory_padding_mask : [b*nq (batch_size) x maxlen_src] binary mask
        xq_context_padded = batch['xq_context_padded'] # batch_size x maxlen_src
        src_embed, src_padding_mask = self.prep_encode(xq_context_padded)
        memory = self.transformer.encoder(src_embed, src_key_padding_mask=src_padding_mask)
        memory_padding_mask = src_padding_mask
        return memory, memory_padding_mask

    def decode(self, z_padded, batch):
        # Forward pass through decoder only
        #
        # Input
        #
        #  memory : [b*nq (batch_size) x maxlen_src x hidden_size] output of transformer encoder
        #  memory_padding_mask : [b*nq (batch_size) x maxlen_src x hidden_size] binary mask padding where False means leave alone
        #
        # Output
        #   output : [b*nq x maxlen_target x output_size]
        tgt_embed, tgt_padding_mask = self.prep_decode(z_padded)
        output = self.forward(tgt_embed, batch)
        # output = self.transformer.decoder(tgt_embed, memory,
        #         tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_padding_mask)

        return output