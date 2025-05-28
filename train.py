from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import warnings
warnings.filterwarnings('ignore', message='.*TypedStorage is deprecated.*')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0


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


class CustomDataset(Dataset):
    def __init__(self,
                 root,
                 split='train',
                 patch_size=256,
                 augment=True,
                 val_ratio=0.2):
        self.degraded_dir = os.path.join(root, 'degraded')
        self.clean_dir = os.path.join(root, 'clean')
        self.patch_size = patch_size
        self.augment = augment
        self.split = split

        # Match files by ID
        all_ids = []
        for fname in os.listdir(self.degraded_dir):
            if fname.endswith(('.png', '.jpg')):
                base = os.path.splitext(fname)[0]  # e.g., "rain-1"
                clean_name = base.replace('-', '_clean-')\
                    + os.path.splitext(fname)[1]
                clean_path = os.path.join(self.clean_dir, clean_name)
                if os.path.exists(clean_path):
                    all_ids.append((fname, clean_name))

        # Shuffle and split
        random.shuffle(all_ids)
        split_idx = int(len(all_ids) * (1 - val_ratio))
        if split == 'train':
            self.ids = all_ids[:split_idx]
        else:
            self.ids = all_ids[split_idx:]

        # Map task names to integer labels
        tasks = sorted({p.split('-')[0] for p, _ in all_ids})
        self.task2id = {t: i for i, t in enumerate(tasks)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        degraded_name, clean_name = self.ids[idx]
        degraded_path = os.path.join(self.degraded_dir, degraded_name)
        clean_path = os.path.join(self.clean_dir, clean_name)

        degraded = Image.open(degraded_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')

        if self.augment and self.split == 'train':
            if random.random() > 0.5:
                degraded = T.functional.hflip(degraded)
                clean = T.functional.hflip(clean)

        to_tensor = T.ToTensor()

        base_name = os.path.splitext(degraded_name)[0]
        task = base_name.split('-')[0]
        de_id = self.task2id.get(task, 0)
        meta = [base_name, de_id]

        return meta, to_tensor(degraded), to_tensor(clean)


def main():
    print("Options")
    print(opt)
    logger = TensorBoardLogger(save_dir="logs/")

    root = "./train"
    train_set = CustomDataset(root, split='train',
                              patch_size=256,
                              augment=True)
    val_set = CustomDataset(root, split='val',
                            patch_size=256, augment=False)

    trainloader = DataLoader(train_set,
                             batch_size=2,
                             shuffle=True,
                             num_workers=4)
    valloader = DataLoader(val_set,
                           batch_size=1,
                           shuffle=False,
                           num_workers=2)

    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir,
                                          every_n_epochs=1,
                                          save_top_k=-1,
                                          save_last=True)

    model = PromptIRModel()

    trainer = pl.Trainer(max_epochs=opt.epochs,
                         accelerator="gpu",
                         devices=opt.num_gpus,
                         strategy="ddp_find_unused_parameters_true",
                         logger=logger,
                         callbacks=[checkpoint_callback])

    trainer.fit(model=model,
                train_dataloaders=trainloader,
                val_dataloaders=valloader)


if __name__ == '__main__':
    main()
