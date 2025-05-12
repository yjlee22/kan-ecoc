import argparse
import os
import multiprocessing as mp
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from kans.efficient_kan import KAN as EfficientKAN
from kans.fastkan import FastKAN
from kans.fasterkan import FasterKAN
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.collections import MetricCollection

def get_model(args, in_features, out_features):
    layers = [in_features] + args.hidden_dims + [out_features]
    if args.model == 'efficient':
        return EfficientKAN(
            layers_hidden=layers,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            scale_noise=args.scale_noise,
            scale_base=args.scale_base,
            scale_spline=args.scale_spline,
            base_activation=torch.nn.SiLU,
            grid_eps=args.grid_eps,
            grid_range=[-1, 1]
        )
    elif args.model == 'fast':
        return FastKAN(
        layers_hidden=layers,
        num_grids=args.grid_size,
        use_base_update=True,
        base_activation=F.silu,         
        spline_weight_init_scale=args.scale_spline
        )
    else:  # faster
        return FasterKAN(
            layers_hidden=layers,
            num_grids=args.grid_size,
            spline_weight_init_scale=args.scale_spline
        )

def train_binary(model, loader, code_matrix, bit_idx, device, epochs, lr):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    for epoch in trange(epochs, desc=f"Bit {bit_idx}"):
        for x, y in loader:
            x = x.to(device)
            y_flat = y.view(-1).to(device)
            bit_vals = code_matrix[y_flat, bit_idx]
            bit_labels = (bit_vals == 1).float().unsqueeze(1)

            optimizer.zero_grad()
            out = model(x.view(x.size(0), -1))
            loss = criterion(out, bit_labels)
            loss.backward()
            optimizer.step()
    return model

def predict_ecoc(models, loader, code_matrix, device):
    code_matrix = code_matrix.to(device)
    for m in models:
        m.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).view(x.size(0), -1)
            logits = torch.cat([m(x) for m in models], dim=1)
            bits = (logits > 0).int() * 2 - 1
            for true_label, b_vec in zip(y.view(-1).tolist(), bits):
                dists = (code_matrix != b_vec).sum(dim=1)
                pred = torch.argmin(dists).item()
                y_true.append(int(true_label))
                y_pred.append(pred)
    return torch.tensor(y_true), torch.tensor(y_pred)

def train_multiclass(model, loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    for epoch in trange(epochs, desc="Multiclass"):
        for x, y in loader:
            x = x.to(device)
            y_flat = y.view(-1).to(device, dtype=torch.long)
            optimizer.zero_grad()
            out = model(x.view(x.size(0), -1))
            loss = criterion(out, y_flat)
            loss.backward()
            optimizer.step()
    return model

def predict_multiclass(model, loader, device):
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).view(x.size(0), -1)
            logits = model(x)
            preds = logits.argmax(dim=1)
            y_true.extend(y.view(-1).tolist())
            y_pred.extend(preds.cpu().tolist())
    return torch.tensor(y_true), torch.tensor(y_pred)

def train_binary_worker(bit_idx, args, in_features, code_matrix, train_ds):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    cm_dev = code_matrix.to(device)
    model = get_model(args, in_features, out_features=1)
    model = train_binary(model, loader, cm_dev, bit_idx,
                         device, epochs=args.epochs, lr=args.lr)
    return model.cpu()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="KAN on MedMNIST: ECOC or direct multiclass with model choice")
    parser.add_argument('--data', type=str, default='bloodmnist')
    parser.add_argument('--ecoc', action='store_true')
    parser.add_argument('--model', choices=['efficient', 'fast', 'faster'], default='efficient',
                        help="Choose 'efficient', 'fast', or 'faster' KAN variant")
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[5])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--spline_order', type=int, default=3)
    parser.add_argument('--scale_noise', type=float, default=0.1)
    parser.add_argument('--scale_base', type=float, default=1.0)
    parser.add_argument('--scale_spline', type=float, default=1.0)
    parser.add_argument('--grid_eps', type=float, default=0.02)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.5], [.5])])
    info = INFO[args.data.lower()]
    DataClass = getattr(medmnist, info['python_class'])
    train_ds = DataClass(split='train', transform=transform, download=True)
    test_ds  = DataClass(split='test',  transform=transform, download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=not args.ecoc, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False,     num_workers=4)

    sample, _    = train_ds[0]
    in_features  = sample.numel()
    num_classes  = len(info['label'])

    if args.ecoc:
        num_bits    = 2 * num_classes
        code_matrix = torch.randint(0, 2, (num_classes, num_bits)) * 2 - 1

        ctx    = mp.get_context('spawn')
        n_proc = min(num_bits, os.cpu_count())
        with ctx.Pool(processes=n_proc) as pool:
            worker_args = [
                (bit_idx, args, in_features, code_matrix, train_ds)
                for bit_idx in range(num_bits)
            ]
            models = pool.starmap(train_binary_worker, worker_args)

        y_true, y_pred = predict_ecoc(models, test_loader, code_matrix, device)
    else:
        model = get_model(args, in_features, out_features=num_classes)
        model = train_multiclass(model, train_loader, device,
                                 epochs=args.epochs, lr=args.lr)
        y_true, y_pred = predict_multiclass(model, test_loader, device)

    metrics = MetricCollection({
        "accuracy": Accuracy(task="multiclass", num_classes=num_classes, average='macro'),
        "precision": Precision(task="multiclass", num_classes=num_classes, average='macro'),
        "recall": Recall(task="multiclass", num_classes=num_classes, average='macro'),
        "f1": F1Score(task="multiclass", num_classes=num_classes, average='macro'),
    })

    metrics = metrics.to(device)
    metrics_dict = metrics(y_pred.to(device), y_true.to(device))

    out_dir = f"seed_{args.seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.model}_ecoc_{args.ecoc}_grid_{args.grid_size}_splineorder_{args.spline_order}_{args.hidden_dims}.txt"), "w") as f:
        for name, val in metrics_dict.items():
            f.write(f"{name}: {val:.4f}\n")

    for name, val in metrics_dict.items():
        print(f"{name.capitalize()}: {val:.4f}")
