import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import lightning.pytorch as pl
import os


class TestDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_filenames = sorted(os.listdir(folder_path))
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0,1] and shape (C, H, W)
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.folder_path, filename)

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)  # shape: (3, H, W), range: [0, 1]

        return filename, image_tensor


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (a smooth approximation to L1 loss)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = CharbonnierLoss()
        self.psnr_values = []
        self.train_losses = []
        self.val_losses = []
        self.log_file = "log.txt"  # log file path

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        self.train_losses.append(loss.item())
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # Clamp to [0, 1] to avoid invalid pixel values
        restored = torch.clamp(restored, 0.0, 1.0)
        clean_patch = torch.clamp(clean_patch, 0.0, 1.0)

        # Compute PSNR
        mse = torch.mean((restored - clean_patch) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
        self.psnr_values.append(psnr.item())

        # Compute and store loss
        loss = self.loss_fn(restored, clean_patch)
        self.val_losses.append(loss.item())

        return psnr

    def on_validation_epoch_end(self):
        avg_psnr = np.mean(self.psnr_values)
        avg_val_loss = np.mean(self.val_losses)

        self.log("val_psnr", avg_psnr, prog_bar=True)
        self.log("val_loss", avg_val_loss, prog_bar=True)

        # Save metrics for logging in on_train_epoch_end
        self.avg_val_psnr = avg_psnr
        self.avg_val_loss = avg_val_loss

        self.psnr_values.clear()
        self.val_losses.clear()

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        avg_train_loss = np.mean(self.train_losses)
        self.train_losses.clear()

        # Write to log.txt
        with open(self.log_file, "a") as f:
            f.write(
                f"Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {self.avg_val_loss:.4f} | "
                f"Val PSNR: {self.avg_val_psnr:.2f}\n"
            )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                                  warmup_epochs=15,
                                                  max_epochs=150)
        return [optimizer], [scheduler]


def run_inference(model, test_loader, device):
    model.eval()
    model.to(device)

    preds = {}

    with torch.no_grad():
        for batch in tqdm(test_loader):
            filenames, degraded_images = batch
            degraded_images = degraded_images.to(device)

            restored = model(degraded_images)
            restored = torch.clamp(restored, 0.0, 1.0)  # ensure [0, 1]

            restored = (restored * 255).byte().cpu().numpy()

            for fname, img in zip(filenames, restored):
                preds[fname] = img  # img: (3, H, W), uint8

    return preds


def save_predictions(preds_dict, output_path="pred.npz"):
    np.savez(output_path, **preds_dict)
    print(f"Saved predictions to {output_path}")


# Load trained model
model = PromptIRModel.load_from_checkpoint("epoch=54-step=87040.ckpt")
model.eval()

# Create test DataLoader
test_dataset = TestDataset("hw4_realse_dataset/test/degraded")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Run inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictions = run_inference(model, test_loader, device)

# Save to pred.npz
save_predictions(predictions, "pred.npz")

data = np.load("pred.npz")
print(data.files)          # ['0.png', '1.png', ...]
