from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from baselines.nafnet_road import NAFNetRoad
from rcadnet.dataset import PairedRoadRestorationDataset
from rcadnet.losses import RCADLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train road restoration architecture baselines.")
    parser.add_argument("--data-root", action="append", required=True)
    parser.add_argument("--scenario", action="append", dest="scenarios", required=True)
    parser.add_argument("--model", choices=["nafnet"], default="nafnet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--init-weights", help="Optional baseline checkpoint to fine-tune from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    datasets = [PairedRoadRestorationDataset(root, args.scenarios, patch_size=args.patch_size, train=True) for root in args.data_root]
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    if args.model == "nafnet":
        model = NAFNetRoad(width=args.width).to(device)
    if args.init_weights:
        checkpoint = torch.load(args.init_weights, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        print(json.dumps({"loaded_init_weights": args.init_weights}), flush=True)
    criterion = RCADLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["gt"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
        row = {"epoch": epoch, "loss": running / max(len(loader), 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "arch": {"model": args.model, "width": args.width},
                "epoch": epoch,
                "args": vars(args),
            },
            out_dir / f"{args.model}_last.pth",
        )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
