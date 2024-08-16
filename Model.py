import os
import glob
import pandas as pd
import numpy as np
import mne
from scipy.signal import iirnotch, filtfilt, butter, resample
import logging
import random
import gc
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Configs:
    def __init__(self):
        self.seq_len = 5000 # EEG sequence length
        self.enc_in = 19 # Number of EEG channels
        self.d_model=32
        self.d_ff = 64
        self.embed = 'fixed'
        self.n_heads = 4
        self.freq='s'
        self.dropout=0.1
        self.e_layers =2
        self.factor = 3
        self.output_attention =False
        self.activation = 'relu'
        self.num_class = 2

configs = Configs()

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # Embedding
        self.enc_embedding_t = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.enc_embedding_f = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder_t = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.encoder_f = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.d_model*configs.seq_len, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 256)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(configs.d_model*configs.seq_len, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 256)
        )

        # Define ResNet18 and modify the number of input channels
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(configs.enc_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Identity()  # Remove the last fully connected layer

        # The projection head maps the final features into a low-dimensional embedding space
        self.projection_head = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
                )


    def forward(self, data_t, data_f, spectrogram_data):
        batchsize, seq_len, num_channels = data_t.shape

        "Time"
        data_t_out = self.enc_embedding_t(data_t, None)
        data_t_out, attns = self.encoder_t(data_t_out, attn_mask=None)
        t_output = self.act(data_t_out)  # 输出的 Transformer 编码器/解码器嵌入不包括非线性处理
        t_output = self.dropout(t_output)
        t_output = t_output.reshape(t_output.shape[0], -1)  # (batch_size, seq_length * d_model)
        t_output = self.projector_t(t_output)

        "Fre"
        data_f_out = self.enc_embedding_f(data_f, None)
        data_f_out, attns = self.encoder_f(data_f_out, attn_mask=None)
        f_output = self.act(data_f_out)  # 输出的 Transformer 编码器/解码器嵌入不包括非线性处理
        f_output = self.dropout(f_output)
        f_output = f_output.reshape(f_output.shape[0], -1)  # (batch_size, seq_length * d_model)
        f_output = self.projector_f(f_output)

        "Spectrogram"
        resnet_features = self.resnet(spectrogram_data)  # 通过ResNet18计算特征

        "CONCAT"
        t_f_concat = torch.cat((t_output, f_output), dim=1)
        t_f_tf_concat = torch.cat((t_f_concat, resnet_features), dim=1)
        output = self.projection_head (t_f_tf_concat)  # (batch_size, num_classes)

        return output

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=128, output_dim=2):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)