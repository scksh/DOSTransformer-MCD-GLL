import sys
import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch_geometric.loader import DataLoader
import utils
from utils import test_phonon, build_data, load_data, train_valid_test_split, r2

# limit CPU usage
torch.set_num_threads(2)

# Default data type float 64 for phdos
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

# Load data
df, species = load_data('./data/processed/data.csv')
print("Dataset Loaded!")

r_max = 4.0  # cutoff radius
df['data'] = df.apply(lambda x: build_data(x, r_max), axis=1)
device = torch.device("cuda:0")

print("Build data")

# MCDropout을 활성화하는 함수
def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

# MCDropout을 이용한 예측 함수
def predict_with_uncertainty(f_model, batch, num_samples=10):
    enable_dropout(f_model)  # Dropout 활성화
    preds = []

    for _ in range(num_samples):
        preds_global, _, _ = f_model(batch)  # preds_global이 첫 번째 출력이라고 가정
        preds.append(preds_global.unsqueeze(0))

    preds = torch.cat(preds, dim=0)  # [num_samples, batch_size, ...] 형태
    mean_preds = preds.mean(dim=0)  # 예측값의 평균
    std_preds = preds.std(dim=0)    # 예측값의 표준편차 (불확실성)
    return mean_preds, std_preds

def compute_gaussian_log_likelihood(target, mean, std):
    variance = std ** 2
    log_likelihood = -0.5 * torch.log(2 * torch.pi * variance) - (target - mean) ** 2 / (2 * variance)
    return log_likelihood.mean()

def main():
    args = utils.parse_args()
    train_config = utils.training_config(args)
    configuration = utils.exp_get_name(train_config)
    print("{}".format(configuration))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # GPU 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터셋 분할
    idx_train, idx_valid, idx_test = train_valid_test_split(df, species, valid_size=0.1, test_size=0.1, seed=args.random_state, plot=False)

    batch_size = 1
    train_loader = DataLoader(df.iloc[idx_train]['data'].values, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(df.iloc[idx_valid]['data'].values, batch_size=batch_size)
    test_loader = DataLoader(df.iloc[idx_test]['data'].values, batch_size=batch_size)

    print("Dataset Loaded!")
    embedder = "DOSTransformer_phonon"
    n_hidden = args.hidden
    n_atom_feat = 118
    n_bond_feat = 4

    # 모델 생성
    from embedder_phDOS.DOSTransformer_phonon import DOSTransformer_phonon
    model = DOSTransformer_phonon(
    args.layers, args.transformer, n_atom_feat, n_bond_feat, n_hidden, attn_drop=args.attn_drop, device=device).to(device)

    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()
    criterion_2 = nn.L1Loss()

    best_rmse = float('inf')
    best_mae = float('inf')
    num_batch = len(df.iloc[idx_train]['data'].values) // batch_size

    for epoch in range(args.epochs):
        model.train()

        for bc, batch in enumerate(train_loader):
            batch.to(device)

            # MCDropout을 사용하여 불확실성을 계산하며 예측
            mean_preds, std_preds = predict_with_uncertainty(model, batch, num_samples=10)

            # MSE와 RMSE 계산
            mse_global = criterion(mean_preds, batch.phdos)
            rmse_global = torch.sqrt(mse_global).mean()

            # 가우시안 로그 가능도 계산
            gaussian_log_likelihood = compute_gaussian_log_likelihood(batch.phdos, mean_preds, std_preds)

            loss = rmse_global  # 학습 손실은 RMSE로 설정
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write(f'\r[Epoch {epoch + 1}/{args.epochs} | Batch {bc + 1}/{num_batch}] '
                             f'Loss: {loss:.4f} Uncertainty: {std_preds.mean().item():.4f} '
                             f'Log-Likelihood: {gaussian_log_likelihood.item():.4f}')
            sys.stdout.flush()

        if (epoch + 1) % args.eval == 0:
            valid_rmse, valid_mse, valid_mae, valid_r2, preds_y = test_phonon(model, valid_loader, criterion_2, r2, device)
            print(f"\n[Epoch {epoch + 1}] Validation RMSE: {valid_rmse:.4f}, MAE: {valid_mae:.4f}")

            if valid_rmse < best_rmse:
                best_rmse = valid_rmse
                print(f"Best RMSE so far: {best_rmse:.4f}")

    print("Training Finished!")

if __name__ == "__main__":
    main() 