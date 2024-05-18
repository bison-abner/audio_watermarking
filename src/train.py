
import torch.optim as optim
import torch.nn as nn
import torch
from src.model import Model
from src.data_loader import load_data
from src.evaluate import signal_noise_ratio, calc_ber, pesq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(num_point=256, num_bit=128, n_fft=512, hop_length=256, use_recover_layer=True, num_layers=16)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_snr = -float('inf')
    best_model_path = ''

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, watermarks in train_loader:
            inputs, watermarks = inputs.to(device), watermarks.to(device)

            optimizer.zero_grad()

            outputs = model.encode(inputs, watermarks)

            loss = criterion(outputs, watermarks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # 验证模型
        model.eval()
        val_loss = 0.0
        snr_total = 0.0
        pesq_total = 0.0
        ber_total = 0.0

        with torch.no_grad():
            for inputs, watermarks in val_loader:
                inputs, watermarks = inputs.to(device), watermarks.to(device)

                outputs = model.encode(inputs, watermarks)

                loss = criterion(outputs, watermarks)
                val_loss += loss.item()

                snr = signal_noise_ratio(inputs.cpu().numpy(), outputs.cpu().numpy())
                pesq = pesq(inputs.cpu().numpy(), outputs.cpu().numpy(), 16000)
                ber = calc_ber(outputs, watermarks).item()

                snr_total += snr
                pesq_total += pesq
                ber_total += ber

        avg_val_loss = val_loss / len(val_loader)
        avg_snr = snr_total / len(val_loader)
        avg_pesq = pesq_total / len(val_loader)
        avg_ber = ber_total / len(val_loader)

        print(f"Validation Loss: {avg_val_loss:.4f}, SNR: {avg_snr:.4f}, PESQ: {avg_pesq:.4f}, BER: {avg_ber:.4f}")

        # 保存最佳模型
        if avg_snr > best_snr:
            best_snr = avg_snr
            best_model_path = f'step{epoch + 1}_snr{avg_snr:.2f}_pesq{avg_pesq:.2f}_ber{avg_ber:.2f}.pkl'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'snr': avg_snr,
                'pesq': avg_pesq,
                'ber': avg_ber
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")

    print(f"Training completed. Best SNR: {best_snr:.2f}, Model saved at: {best_model_path}")


# 加载训练和验证数据
train_loader = load_data('path/to/train_audio', 'path/to/train_watermarks.npy')
val_loader = load_data('path/to/val_audio', 'path/to/val_watermarks.npy')

# 开始训练
num_epochs = 100
train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
