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

#Reserve designated electrode channels
target_channels = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
                   'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
                   'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                   'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
                   'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

data_dir = '..'
num_epochs = 20
batch_size = 128
temperature = 0.5 # The paper in SimCLR recommends using temperature = 0.5 as the default value
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

# class Classifier(nn.Module):
#     def __init__(self, input_dim=128, num_classes=2):
#         super(Classifier, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         return self.classifier(x)

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._get_correlated_mask().type(torch.bool)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def _get_correlated_mask(self):
        diag = torch.eye(2 * self.batch_size)
        l1 = torch.roll(diag, shifts=self.batch_size, dims=1)
        mask = (1 - diag) * (1 - l1)
        return mask

    def forward(self, z_i, z_j):
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positives = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(-1, 1)
        negatives = sim[self.mask].reshape(2 * self.batch_size, -1)

        labels = torch.zeros(2 * self.batch_size).to(positives.device).long()
        logits = torch.cat((positives, negatives), dim=1)

        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss

def random_remove_frequencies(data_t, remove_ratio=0.1, min_freq_index=4):
    """
    Randomly remove some frequency components. Randomly remove a specified proportion of frequency components.

    :param data_t: Input time series data with shape (seq_len, num_channels).
    :param remove_ratio: The proportion of frequency components to be removed, with a value between 0 and 1.
    :param min_freq_index: The lowest frequency index from which frequencies can start being removed.
        When removing frequencies, selecting very low frequencies (close to the DC component) may affect the baseline level of the data.
    :return: The augmented time series data with some frequency components removed in the frequency domain.
    """
    freq_domain = torch.fft.fft(data_t, dim=0)
    seq_len, num_channels = freq_domain.shape
    num_freqs_to_remove = int((seq_len - min_freq_index) * remove_ratio)
    indices_to_remove = torch.randint(min_freq_index, seq_len, (num_freqs_to_remove,))
    freq_domain[indices_to_remove, :] = 0
    augmented_data_t = torch.fft.ifft(freq_domain, dim=0).real

    return augmented_data_t

#Dataset for pretrain
class EEGDataset(Dataset):
    def __init__(self, data_dir, n_fft=500, hop_length=125, win_length=500, window=None):
        """
        Parameters:
        - data_dir: The directory path containing all the .npy files.
        """
        self.data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')]

        # STFT参数
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window if window is not None else torch.hann_window(self.win_length)

        # For the nth pre-training, take the nth sub-sample from all files，
        self.samples = []
        for file_idx, file in enumerate(self.data_files):
            data = np.load(file)
            if data.shape[0] >= 2:
                self.samples.append((file_idx, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieve a sub-sample from the dataset based on the index and generate two augmented versions of the data.

        Parameters:
        - index: The index of the sample.

        Returns:
        - The original data sample
        - Augmented data sample 1
        - Augmented data sample 2
        """
        file_idx, sample_idx = self.samples[index]
        data = np.load(self.data_files[file_idx])
        data = torch.tensor(data[sample_idx], dtype=torch.float32)

        segment_tensor1 = random_remove_frequencies(data, remove_ratio=0.1)
        segment_tensor2 = random_remove_frequencies(data, remove_ratio=0.1)

        # FFT abs
        data_f1 = torch.fft.fft(segment_tensor1, dim=0).abs()
        data_f2 = torch.fft.fft(segment_tensor2, dim=0).abs()
        # STFT abs
        spectrogram_data1 = []
        for channel in range(segment_tensor1.shape[1]):
            Zxx1 = torch.stft(segment_tensor1[:, channel], n_fft=self.n_fft, hop_length=self.hop_length,
                              win_length=self.win_length, window=self.window, return_complex=True)
            spectrogram_data1.append(Zxx1.abs())
        spectrogram_data1 = torch.stack(spectrogram_data1, dim=0)

        spectrogram_data2 = []
        for channel in range(segment_tensor2.shape[1]):
            Zxx2 = torch.stft(segment_tensor2[:, channel], n_fft=self.n_fft, hop_length=self.hop_length,
                              win_length=self.win_length, window=self.window, return_complex=True)
            spectrogram_data2.append(Zxx2.abs())
        spectrogram_data2 = torch.stack(spectrogram_data2, dim=0)

        return (segment_tensor1, data_f1, spectrogram_data1), (segment_tensor2, data_f2, spectrogram_data2)


def setup_logger(log_file):
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # 创建一个文件处理器来记录日志
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 创建一个控制台处理器来输出日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

from torch.cuda.amp import autocast, GradScaler
def train_contrastive_learning(train_loader, model, batch_size, temperature, num_epochs, learning_rate, save_path="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, "datraining_log.txt")
    logger = setup_logger(log_file)

    criterion = NTXentLoss(batch_size=batch_size, temperature=temperature).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.2)

    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (augmented_pair1, augmented_pair2) in enumerate(train_loader):
            segment_tensor1, data_f1, spectrogram_data1 = augmented_pair1
            segment_tensor2, data_f2, spectrogram_data2 = augmented_pair2

            segment_tensor1, data_f1, spectrogram_data1 = segment_tensor1.to(device), data_f1.to(
                device), spectrogram_data1.to(device)
            segment_tensor2, data_f2, spectrogram_data2 = segment_tensor2.to(device), data_f2.to(
                device), spectrogram_data2.to(device)

            with autocast():
                z_i = model(segment_tensor1, data_f1, spectrogram_data1)
                z_j = model(segment_tensor2, data_f2, spectrogram_data2)

                loss = criterion(z_i, z_j)

            total_loss += loss.item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.6f}')


    final_model_save_path = os.path.join(save_path, "pretrain_model2.pth")
    torch.save(model.state_dict(), final_model_save_path)
    print(f"Final model saved to {final_model_save_path}")

    print("Training completed.")
    return model


dataset = EEGDataset(data_dir=data_dir)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=8, pin_memory=True)
pretrained_model_path = "final_model.pth"
model = Model(configs)
model.load_state_dict(torch.load(pretrained_model_path))
model = model.to(device)
trained_model = train_contrastive_learning(train_loader, model, batch_size=batch_size, temperature=temperature, num_epochs=num_epochs, learning_rate=lr)