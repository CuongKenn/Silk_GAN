import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils import ensure_dir, load_yaml, seed_everything


class PairedDataset(Dataset):
    def __init__(self, edge_dir: str, rgb_dir: str, size: int):
        self.edges = sorted(glob.glob(os.path.join(edge_dir, "*.png")))
        self.rgb_dir = rgb_dir
        self.size = size
        if not self.edges:
            raise RuntimeError(f"No edge files found in {edge_dir}")

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        e_path = self.edges[idx]
        name = os.path.basename(e_path)
        r_path = os.path.join(self.rgb_dir, name)

        edge = cv2.imread(e_path, cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(r_path, cv2.IMREAD_COLOR)
        if edge is None or rgb is None:
            raise RuntimeError(f"Missing pair for {name}")

        edge = cv2.resize(edge, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)

        edge = torch.from_numpy(edge).unsqueeze(0).float() / 127.5 - 1.0
        rgb = torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 127.5 - 1.0
        return edge, rgb


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, base=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.e2 = nn.Sequential(nn.Conv2d(base, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.LeakyReLU(0.2, True))
        self.e3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.LeakyReLU(0.2, True))
        self.b = nn.Sequential(nn.Conv2d(base * 4, base * 8, 4, 2, 1), nn.ReLU(True))
        self.d3 = nn.Sequential(nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1), nn.BatchNorm2d(base * 4), nn.ReLU(True))
        self.d2 = nn.Sequential(nn.ConvTranspose2d(base * 8, base * 2, 4, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(True))
        self.d1 = nn.Sequential(nn.ConvTranspose2d(base * 4, base, 4, 2, 1), nn.BatchNorm2d(base), nn.ReLU(True))
        self.out = nn.Sequential(nn.ConvTranspose2d(base * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b = self.b(e3)
        d3 = self.d3(b)
        d2 = self.d2(torch.cat([d3, e3], dim=1))
        d1 = self.d1(torch.cat([d2, e2], dim=1))
        return self.out(torch.cat([d1, e1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=4, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 4, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def save_preview(edge, fake, out_dir, step):
    ensure_dir(out_dir)
    e = ((edge[0, 0].detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    f = fake[0].detach().cpu().clamp(-1, 1)
    f = ((f + 1.0) * 127.5).byte().numpy().transpose(1, 2, 0)
    e_rgb = np.repeat(e[:, :, None], 3, axis=2)
    vis = np.concatenate([e_rgb, f], axis=1)
    cv2.imwrite(os.path.join(out_dir, f"pix2pix_step_{step:06d}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/stage1/pix2pix.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PairedDataset(
        edge_dir=cfg["data"]["edge_dir"],
        rgb_dir=cfg["data"]["rgb_dir"],
        size=int(cfg["data"]["image_size"]),
    )
    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=2, drop_last=True)

    g = UNetGenerator(in_ch=1, out_ch=3, base=int(cfg["model"]["channels"])) .to(device)
    d = PatchDiscriminator(in_ch=4, base=int(cfg["model"]["channels"])) .to(device)

    opt_g = optim.Adam(g.parameters(), lr=float(cfg["train"]["lr"]), betas=(0.5, 0.999))
    opt_d = optim.Adam(d.parameters(), lr=float(cfg["train"]["lr"]), betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    step = 0
    out_ckpt = cfg["paths"]["checkpoint_dir"]
    out_samples = cfg["paths"]["sample_dir"]
    ensure_dir(out_ckpt)
    ensure_dir(out_samples)

    for epoch in range(int(cfg["train"]["epochs"])):
        pbar = tqdm(dl, desc=f"Epoch {epoch + 1}")
        for edge, real in pbar:
            edge = edge.to(device)
            real = real.to(device)

            # D
            with torch.no_grad():
                fake = g(edge)
            d_real = d(torch.cat([edge, real], dim=1))
            d_fake = d(torch.cat([edge, fake], dim=1))
            d_loss = 0.5 * (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # G
            fake = g(edge)
            d_fake = d(torch.cat([edge, fake], dim=1))
            g_adv = bce(d_fake, torch.ones_like(d_fake))
            g_l1 = l1(fake, real)
            g_loss = g_adv + float(cfg["train"]["lambda_l1"]) * g_l1
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            step += 1
            pbar.set_postfix(d_loss=float(d_loss.item()), g_loss=float(g_loss.item()))

            if step % int(cfg["train"]["sample_every"]) == 0:
                save_preview(edge, fake, out_samples, step)

            if step % int(cfg["train"]["checkpoint_every"]) == 0:
                torch.save({"g": g.state_dict(), "d": d.state_dict(), "step": step}, os.path.join(out_ckpt, f"pix2pix_{step:06d}.pt"))

    torch.save({"g": g.state_dict(), "d": d.state_dict(), "step": step}, os.path.join(out_ckpt, "pix2pix_last.pt"))
    print("Pix2Pix training complete")


if __name__ == "__main__":
    main()
