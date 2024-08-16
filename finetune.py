import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding

data_dir = ''
num_epochs = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

log_file = 'finetune-evaluation.txt'

pretrained_model_path = 'pretrain_model2.pth'  # Path to pre-trained model

class Configs:
    def __init__(self):
        self.seq_len = 5000
        self.enc_in= 19
        self.d_model=32
        self.d_ff = 64
        self.embed = 'fixed'
        self.n_heads = 4
        self.freq='s'
        self.dropout=0.2
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

        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(configs.enc_in, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Identity()  # 移除最后的全连接层

        self.projection_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, data_t, data_f, spectrogram_data):

        data_t_out = self.enc_embedding_t(data_t, None)
        data_t_out, attns = self.encoder_t(data_t_out, attn_mask=None)
        t_output = self.act(data_t_out)  # 输出的 Transformer 编码器/解码器嵌入不包括非线性处理
        t_output = self.dropout(t_output)
        t_output = t_output.reshape(t_output.shape[0], -1)  # (batch_size, seq_length * d_model)
        t_output = self.projector_t(t_output)

        data_f_out = self.enc_embedding_f(data_f, None)
        data_f_out, attns = self.encoder_f(data_f_out, attn_mask=None)
        f_output = self.act(data_f_out)  # 输出的 Transformer 编码器/解码器嵌入不包括非线性处理
        f_output = self.dropout(f_output)
        f_output = f_output.reshape(f_output.shape[0], -1)  # (batch_size, seq_length * d_model)
        f_output = self.projector_f(f_output)

        resnet_features = self.resnet(spectrogram_data)

        t_f_concat = torch.cat((t_output, f_output), dim=1)
        t_f_tf_concat = torch.cat((t_f_concat, resnet_features), dim=1)
        output = self.projection_head (t_f_tf_concat)

        return output

#dataset for finetune
class EEGDataset(Dataset):
    def __init__(self, data_dir, test_subject=None, exclude_test_subject=False, n_fft=500, hop_length=125, win_length=500, window=None):
        self.data_dir = data_dir
        self.test_subject = test_subject
        self.exclude_test_subject = exclude_test_subject
        self.file_paths = glob.glob(os.path.join(data_dir, 'sub*_segment*.csv')) # 获取所有匹配 'sub*_segment*.csv' 的文件路径

        # STFT
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window if window is not None else torch.hann_window(self.win_length)

        if self.test_subject is not None:
            if self.exclude_test_subject:
                # Exclude test subject data
                self.file_paths = [fp for fp in self.file_paths if
                                    f'sub{self.test_subject:03d}' not in os.path.basename(fp)]
            else:
                # Only keep test subject data
                self.file_paths = [fp for fp in self.file_paths if
                                   f'sub{self.test_subject:03d}' in os.path.basename(fp)]


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = pd.read_csv(file_path).iloc[:, :19].values # Read the CSV file and get the first 19 columns of electrode data

        if len(data) < 5000:
            padding = np.zeros((5000 - len(data), data.shape[1]))
            data = np.vstack((data, padding))

        data_t = torch.tensor(data, dtype=torch.float32)

        # FFT
        data_f = torch.fft.fft(data_t, dim=0).abs()

        # STFT
        spectrogram_data = []
        for channel in range(data_t.shape[1]):
            Zxx = torch.stft(data_t[:, channel], n_fft=self.n_fft, hop_length=self.hop_length,
                             win_length=self.win_length, window=self.window, return_complex=True) #复数频谱，包含了每个时间窗口的频率成分。通过 Zxx.abs() 获取幅度谱。
            spectrogram_data.append(Zxx.abs())
        spectrogram_data = torch.stack(spectrogram_data, dim=0)

        # Extract subject number from file name
        file_name = os.path.basename(file_path)
        subject_id = int(file_name[3:6])

        # Generate labels: 001 to 036 are 1 (AD patients), 037 to 065 are 0 (healthy people)
        if 1 <= subject_id <= 36:
            label = 1
        elif 37 <= subject_id <= 65:
            label = 0
        else:
            raise ValueError(f"Unexpected subject id: {subject_id}")

        label = torch.tensor(label, dtype=torch.long)

        return data_t, data_f, spectrogram_data,label, subject_id

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=128, output_dim=2):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def train_model(dataloader, model,classifier, criterion, optimizer, num_epochs=num_epochs):
    model.train()
    classifier.train()

    train_loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data_t, data_f, spectrogram_data, label, _ in dataloader:

            if data_t.size(0) == 1:
                continue  # Skip batch size 1

            data_t, data_f, spectrogram_data, label = data_t.to(device), data_f.to(device), spectrogram_data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(data_t, data_f, spectrogram_data)

            outputs = classifier(outputs)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data_t.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        train_loss_history.append(epoch_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    return model,train_loss_history

def evaluate_model(dataloader, model, classifier, criterion):
    model.eval()
    classifier.eval()

    votes_for_1 = 0
    votes_for_0 = 0

    with torch.no_grad():
        for data_t, data_f, spectrogram_data, label, _ in dataloader:

            if data_t.size(0) == 1:
                continue

            data_t, data_f, spectrogram_data, label = data_t.to(device), data_f.to(device), spectrogram_data.to(device), label.to(device)
            outputs = model(data_t, data_f, spectrogram_data) # (batch_size, 128)
            outputs = classifier(outputs)  # (batch_size, num_classes)

            loss = criterion(outputs, label)

            _, preds = torch.max(outputs, 1)

            # Voting mechanism: determine the final predicted label
            votes_for_1 += (preds == 1).sum().item()
            votes_for_0 += (preds == 0).sum().item()

    # final voting results
    final_pred = 1 if votes_for_1 > votes_for_0 else 0

    # real label
    true_label = label[0].item()

    return true_label, final_pred


all_true_labels = []
all_pred_labels = []

with open(log_file, 'w') as log:
    for test_subject in range(1, 66):
        print(f"Evaluating for subject {test_subject}")
        log.write(f"Evaluating for subject {test_subject}\n")

        train_dataset = EEGDataset(data_dir, test_subject=test_subject,exclude_test_subject= True)
        train_loader = DataLoader(train_dataset, batch_size=192, shuffle=True,drop_last=False,num_workers=8,pin_memory=True)

        model = Model(configs)
        classifier = SimpleClassifier()
        model = model.to(device)
        classifier = classifier.to(device)


        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)

        model, train_loss_history = train_model(train_loader, model, classifier, criterion, optimizer, num_epochs=num_epochs)

        log.write(f"Training loss history for subject {test_subject}: {train_loss_history}\n")

        test_dataset = EEGDataset(data_dir, test_subject=test_subject,exclude_test_subject= False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,drop_last=False,num_workers=8,pin_memory=True)

        true_label, final_pred = evaluate_model(test_loader, model, classifier, criterion)

        all_true_labels.append(true_label)
        all_pred_labels.append(final_pred)

all_true_labels = np.array(all_true_labels)
all_pred_labels = np.array(all_pred_labels)


accuracy = accuracy_score(all_true_labels, all_pred_labels)
precision = precision_score(all_true_labels, all_pred_labels, zero_division=0)
recall = recall_score(all_true_labels, all_pred_labels, zero_division=0)
f1 = f1_score(all_true_labels, all_pred_labels, zero_division=0)
auc = roc_auc_score(all_true_labels, all_pred_labels)
conf_matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=[0, 1])
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
specificity = TN / (TN + FP)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

with open(log_file, 'a') as log:
    log.write(f"Final Results:\n")
    log.write(f"Accuracy: {accuracy:.4f}\n")
    log.write(f"Precision: {precision:.4f}\n")
    log.write(f"Recall (Sensitivity): {recall:.4f}\n")
    log.write(f"Specificity: {specificity:.4f}\n")
    log.write(f"F1 Score: {f1:.4f}\n")
    log.write(f"AUC: {auc:.4f}\n")