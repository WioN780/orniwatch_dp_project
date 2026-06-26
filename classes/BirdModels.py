import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Recurrent Neural Network

class BirdCRNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=28, hidden_size=256, gru_num_layers=1, cnn_channels=128, dropout=0.2):
        super().__init__()
        # --- CNN: keep T, reduce F ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5,3), padding=(2,1), bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)), nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=(5,3), padding=(2,1), bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)), nn.Dropout(dropout),

            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(128, cnn_channels, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.freq_reduction = 4  # due to two (2,1) pools

        # --- RNN over time (flatten C×F')
        feat_per_frame = cnn_channels * (n_mels // self.freq_reduction)
        self.gru = nn.GRU(
            input_size=feat_per_frame,
            hidden_size=hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # --- Heads ---
        self.head_time = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, n_classes),
        ) # -> (B,T,C) -> (B,C,T)

        # Seg head for tf -> (B,C,F',T), then upsample F'->F
        self.head_tf = nn.Sequential(
            nn.Conv2d(cnn_channels, cnn_channels//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels//2, cnn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels, n_classes, 1)
        )


    def forward(self, x, lengths=None, return_tf=True):
        """
        x: (B,1,F,T)
        returns:
          logits_time: (B,C,T)
          logits_tf:   (B,C,F,T) if return_tf
        """
        B, _, Fm, T = x.shape

        # --- CNN ---
        feat_map = self.cnn(x)                              # (B, Cc, F', T)
        B, Cc, Fp, Tp = feat_map.shape                      # Tp == T

        # --- RNN over time ---
        rnn_in = feat_map.permute(0, 3, 1, 2).contiguous()  # (B,T,Cc,F')
        rnn_in = rnn_in.view(B, Tp, Cc * Fp)                # (B,T, Cc*F')

        if lengths is not None:
            # lengths are in frames of original T, CNN kept T
            packed = nn.utils.rnn.pack_padded_sequence(rnn_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.gru(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,2H)
        else:
            rnn_out, _ = self.gru(rnn_in)                   # (B,T,2H)

        # --- time head ---
        time_logits = self.head_time(rnn_out)               # (B,T,C)
        time_logits = time_logits.transpose(1, 2).contiguous()  # (B,C,T)

        if not return_tf:
            return time_logits, None

        # --- tf head ---
        tf_low = self.head_tf(feat_map)                     # (B,C,F',T)
        # upsample along frequency back to F, keep time as-is
        tf_logits = F.interpolate(tf_low, size=(Fm, T), mode="bilinear", align_corners=False)  # (B,C,F,T)

        return time_logits, tf_logits