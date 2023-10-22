#########################

import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Hyperparameters
batch_size = 32
num_epochs = 10
lr = 0.0002
betas = (0.5, 0.999)
max_mfcc_time_length = 300  # New parameter

class VoiceDataset(Dataset):
    def __init__(self, directory, transform=None, segment_length=1.0, sr=16000):
        self.directory = directory
        self.segment_length = segment_length
        self.sr = sr
        self.file_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        self.transform = transform

    def __len__(self):
        # Here, the length is computed as the number of segments each audio file can be split into
        total_segments = 0
        for f in self.file_list:
            y, _ = librosa.load(os.path.join(self.directory, f), sr=self.sr)
            num_segments = int(len(y) / (self.segment_length * self.sr))
            total_segments += num_segments
        return total_segments

    def __getitem__(self, idx):
        # Find which file the idx belongs to and which segment of that file
        file_idx = 0
        while idx >= int(librosa.get_duration(path=os.path.join(self.directory, self.file_list[file_idx])) / self.segment_length):
            idx -= int(librosa.get_duration(path=os.path.join(self.directory, self.file_list[file_idx])) / self.segment_length)
            file_idx += 1

        start_sample = int(idx * self.segment_length * self.sr)
        end_sample = start_sample + int(self.segment_length * self.sr)

        y, _ = librosa.load(os.path.join(self.directory, self.file_list[file_idx]), sr=self.sr, offset=start_sample/self.sr, duration=self.segment_length)
        
        mfccs = preprocess_audio_from_array(y)
        return torch.Tensor(mfccs).unsqueeze(1)


def get_batched_data(loader, batch_size):
    iter_loader = iter(loader)
    data = []
    while len(data) < batch_size:
        try:
            chunks = next(iter_loader)
            data.extend(chunks)
        except StopIteration:  # reset the iterator if we run out of data
            iter_loader = iter(loader)
    return torch.stack(data[:batch_size], 0)  # Combine the data to form a batch

def preprocess_audio(y, max_len=max_mfcc_time_length, sr=16000, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    if (mfccs.shape[1] > max_len):
        mfccs = mfccs[:, :max_len]
    else:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), 'constant')
    
    return mfccs

def preprocess_audio_from_array(y, max_len=max_mfcc_time_length, sr=16000, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Truncate or pad the MFCCs
    if (mfccs.shape[1] > max_len):
        mfccs = mfccs[:, :max_len]
    else:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), 'constant')
    
    return mfccs


# Model Definitions

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=5):
        super(Generator, self).__init__()

        model = [
            nn.Conv2d(input_nc, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        for _ in range(n_residual_blocks):
            model.append(ResidualBlock(64))

        # Add a final convolutional layer to ensure output has the correct number of channels
        model += [
            nn.Conv2d(64, output_nc, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.InstanceNorm2d(output_nc),
            nn.ReLU(inplace=True)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



# Modified Discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(input_nc, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        self.model = nn.Sequential(*model)
        self.output_layer = nn.Conv2d(64, output_nc, kernel_size=(3, 3), stride=1, padding=(1, 1))

    def forward(self, x):
        x = self.model(x)
        return self.output_layer(x)
    
    def get_output_shape(self, input_shape):
        shape = self.model(torch.zeros(1, *input_shape))
        return shape.size()


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cat_data_directory = '.\\cat_data\\'
    human_data_directory = '.\\doctor\\'

    cat_dataset = VoiceDataset(directory=cat_data_directory)
    human_dataset = VoiceDataset(directory=human_data_directory)

    cat_loader = DataLoader(cat_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    human_loader = DataLoader(human_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    G_A2B = Generator(13, 13).to(device)
    G_B2A = Generator(13, 13).to(device)
    D_A = Discriminator(13, 1).to(device)
    D_B = Discriminator(13, 1).to(device)

    D_A_output_shape = D_A.get_output_shape((13, max_mfcc_time_length, max_mfcc_time_length))
    D_B_output_shape = D_B.get_output_shape((13, max_mfcc_time_length, max_mfcc_time_length))

    optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=lr, betas=betas)
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=betas)
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=betas)
    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_cycle = torch.nn.L1Loss()

    human_iter = iter(human_loader)

    for epoch in range(num_epochs):
      for i in range(len(cat_loader)):  # Assuming cat_loader has the maximum number of batches
          real_A = get_batched_data(cat_loader, batch_size)
          real_B = get_batched_data(human_loader, batch_size)

          print("real_A_shape", real_A.shape)
          print('real_B_shape', real_B.shape)

            # Inside the main loop, after retrieving real_A and real_B
          current_batch_size = real_A.size(0)  # Get the current batch size

          valid = torch.ones((current_batch_size, *D_B_output_shape[1:])).to(device)
          #fake = torch.zeros((current_batch_size, *D_B_output_shape[1:])).to(device)

          fake = torch.Tensor(real_A.size(0), 1, 1, 150).fill_(0.0).to(device)
          real = torch.Tensor(real_B.size(0), 1, 1, 150).fill_(1.0).to(device)


          #fake = fake.view(input.size())  # Reshape 'fake' to have same shape as 'input'

          optimizer_G.zero_grad()

          fake_B = G_A2B(real_A)
          valid = torch.ones_like(D_B(fake_B)).to(device)
          loss_GAN_A2B = criterion_GAN(D_B(fake_B), valid)
          fake_A = G_B2A(real_B)  # Ensure that G_B2A produces the right batch size
          valid = torch.ones_like(D_A(real_A)).to(device)
          loss_GAN_B2A = criterion_GAN(D_A(fake_A), valid)
          recovered_A = G_B2A(fake_B)
          loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
          recovered_B = G_A2B(fake_A)
          loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

          loss_G = loss_GAN_A2B + loss_GAN_B2A + 10.0 * (loss_cycle_ABA + loss_cycle_BAB)
          loss_G.backward()
          optimizer_G.step()
        
          optimizer_D_A.zero_grad()
          fake_A_ = G_B2A(real_B).detach()
          loss_D_A_real = criterion_GAN(D_A(real_A), valid)
          loss_D_A_fake = criterion_GAN(D_A(fake_A_), fake)
          loss_D_A = (loss_D_A_real + loss_D_A_fake) / 2
          loss_D_A.backward()
          optimizer_D_A.step()
        
          optimizer_D_B.zero_grad()
          fake_B_ = G_A2B(real_A).detach()
          loss_D_B_real = criterion_GAN(D_B(real_B), valid)
          loss_D_B_fake = criterion_GAN(D_B(fake_B_), fake)
          loss_D_B = (loss_D_B_real + loss_D_B_fake) / 2
          loss_D_B.backward()
          optimizer_D_B.step()
            
          print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(cat_loader)}], Loss G: {loss_G.item()}, Loss D_A: {loss_D_A.item()}, Loss D_B: {loss_D_B.item()}")


    print("Training Complete!")

if __name__ == '__main__':
    main()
