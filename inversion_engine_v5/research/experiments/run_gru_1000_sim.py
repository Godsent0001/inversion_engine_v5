import os
import sys
import time
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project path in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from inversion_engine_v5.research.core.feature_engine_fast import build_all_features_fast
from inversion_engine_v5.research.core.gru_sim_engine import GRUClassifier, simulate_trading_numba

def load_one_year_m1(data_path):
    print("Step 1: Reading 1-year M1 index window via chunk stream...")
    chunks = []
    for chunk in pd.read_csv(data_path, chunksize=100000):
        time_str = chunk['time'].str[:10]
        m = (time_str >= '2024-10-01') & (time_str <= '2025-09-30')
        if m.any():
            chunks.append(chunk[m])

    df_period = pd.concat(chunks, ignore_index=True)
    df_period['time'] = pd.to_datetime(df_period['time'])
    del chunks
    gc.collect()

    return df_period.sort_values('time').reset_index(drop=True)

def run_simulation():
    data_path = 'inversion_engine_v5/research/data/raw/markets/XAUUSD/XAUUSD_M1.csv'
    markets_dir = 'inversion_engine_v5/research/data/raw/markets'
    output_dir = 'inversion_engine_v5/outputs'
    os.makedirs(output_dir, exist_ok=True)

    df_period = load_one_year_m1(data_path)
    print(f"Data period: {df_period['time'].min()} to {df_period['time'].max()}, Total bars: {len(df_period)}")

    print("Step 2: Building ultra-fast vectorized feature matrix...")
    t0 = time.time()
    X_np, cleaned_df = build_all_features_fast(df_period, markets_dir=markets_dir)
    t1 = time.time()
    print(f"Features built in {t1-t0:.2f}s, Matrix shape: {X_np.shape}, Memory: {X_np.nbytes / (1024*1024):.2f}MB")

    input_dim = X_np.shape[1]
    X_tensor = torch.from_numpy(X_np).float()

    # 2H window index integer
    window_ids = (cleaned_df['time'].dt.floor('2h') - cleaned_df['time'].min()).dt.total_seconds() // 7200
    window_ids = window_ids.values.astype(np.int64)

    # Dynamic month mapping for 1..12
    unique_ym = sorted(list(set([(t.year, t.month) for t in cleaned_df['time']])))
    months_map = {ym: idx + 1 for idx, ym in enumerate(unique_ym)}
    month_ids = np.array([months_map[(t.year, t.month)] for t in cleaned_df['time']], dtype=np.int32)

    # Day index mapping for daily Sharpe calculation
    day_ids_global = (cleaned_df['time'].dt.floor('D') - cleaned_df['time'].min()).dt.days.values.astype(np.int32)

    opens = cleaned_df['open'].values.astype(np.float64)
    highs = cleaned_df['high'].values.astype(np.float64)
    lows = cleaned_df['low'].values.astype(np.float64)
    closes = cleaned_df['close'].values.astype(np.float64)
    atrs = cleaned_df['high'].sub(cleaned_df['low']).rolling(14).mean().ffill().bfill().values.astype(np.float64)
    spreads = cleaned_df['spread'].values.astype(np.float64) if 'spread' in cleaned_df.columns else np.zeros(len(cleaned_df))

    del cleaned_df, df_period
    gc.collect()

    seq_len = 10
    num_models = 1000
    seeds = [1000 + i for i in range(num_models)]

    print("Step 3: Preparing monthly slice indices...")
    monthly_slices = {}
    num_months = len(unique_ym)

    for m in range(1, num_months + 1):
        m_mask = (month_ids == m)
        m_indices = np.where(m_mask)[0]
        if len(m_indices) == 0:
            continue

        start_idx = max(0, m_indices[0] - seq_len + 1)
        end_idx = m_indices[-1] + 1
        warmup_offset = m_indices[0] - start_idx

        m_day_ids = day_ids_global[m_indices]
        _, day_indices = np.unique(m_day_ids, return_inverse=True)
        num_days = int(len(np.unique(m_day_ids)))

        monthly_slices[m] = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'warmup_offset': warmup_offset,
            'm_indices': m_indices,
            'win_ids': window_ids[m_indices],
            'm_ids': month_ids[m_indices],
            'day_ids': day_indices.astype(np.int32),
            'num_days': num_days,
            'opens': opens[m_indices],
            'highs': highs[m_indices],
            'lows': lows[m_indices],
            'closes': closes[m_indices],
            'atrs': atrs[m_indices],
            'spreads': spreads[m_indices]
        }

    print(f"Step 4: Running simulation sequentially for {num_models} GRU models...")
    torch.set_num_threads(4)

    results = []

    t0 = time.time()
    for i in tqdm(range(num_models)):
        m_id = i
        seed = seeds[i]

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_classes=3)
        model.eval()

        disqualified = False
        months_survived = 0

        monthly_pnls = np.zeros(num_months + 1, dtype=np.float64)
        monthly_sharpes = np.zeros(num_months + 1, dtype=np.float64)
        monthly_counts = np.zeros(num_months + 1, dtype=np.int32)

        with torch.no_grad():
            for m in range(1, num_months + 1):
                if m not in monthly_slices:
                    continue

                ms = monthly_slices[m]
                X_m = X_tensor[ms['start_idx']:ms['end_idx']]
                X_m_seq = X_m.unfold(0, seq_len, 1).transpose(1, 2)

                probs = model(X_m_seq)
                preds = torch.argmax(probs, dim=-1).numpy()

                warmup_offset = ms['warmup_offset']
                m_preds = preds[warmup_offset:] if warmup_offset > 0 else preds

                m_disq, m_surv, m_pnls, m_sharpes, m_cnts = simulate_trading_numba(
                    m_preds,
                    ms['win_ids'],
                    ms['m_ids'],
                    ms['day_ids'],
                    ms['opens'],
                    ms['highs'],
                    ms['lows'],
                    ms['closes'],
                    ms['atrs'],
                    ms['spreads'],
                    num_days=ms['num_days']
                )

                pnl_m = m_pnls[m]
                sharpe_m = m_sharpes[m]
                cnt_m = m_cnts[m]

                monthly_pnls[m] = pnl_m
                monthly_sharpes[m] = sharpe_m
                monthly_counts[m] = cnt_m

                if pnl_m < 0.0 or m_disq:
                    disqualified = True
                    break

                months_survived += 1

        total_pnl = np.sum(monthly_pnls[1:months_survived + 1]) if months_survived > 0 else monthly_pnls[1]
        total_trades = np.sum(monthly_counts[1:months_survived + 1])

        if not disqualified and months_survived == num_months:
            valid_sharpes = monthly_sharpes[1:num_months + 1]
            avg_monthly_sharpe = float(np.mean(valid_sharpes))
            rank_score = avg_monthly_sharpe
        else:
            avg_monthly_sharpe = float(np.mean(monthly_sharpes[1:months_survived + 1])) if months_survived > 0 else -10.0
            rank_score = -100.0 + months_survived + (avg_monthly_sharpe * 0.01)

        res = {
            'model_id': m_id,
            'seed': seed,
            'months_survived': months_survived,
            'disqualified': disqualified,
            'total_trades': int(total_trades),
            'total_pnl': float(total_pnl),
            'avg_monthly_sharpe': float(avg_monthly_sharpe),
            'rank_score': float(rank_score)
        }

        for m in range(1, num_months + 1):
            res[f'pnl_m{m}'] = float(monthly_pnls[m])
            res[f'sharpe_m{m}'] = float(monthly_sharpes[m])
            res[f'trades_m{m}'] = int(monthly_counts[m])

        results.append(res)

        if (i + 1) % 100 == 0:
            gc.collect()

    t1 = time.time()
    print(f"\nSimulation completed in {t1-t0:.2f} seconds ({num_models} models evaluated)!")

    # Sort results by rank_score descending
    results.sort(key=lambda r: r['rank_score'], reverse=True)

    df_res = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'gru_models_metrics.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"Saved complete metrics for all {num_models} models to {csv_path}")

    # Top 10 models selection and weights saving
    top_10 = results[:10]
    top_dir = os.path.join(output_dir, 'top_10_gru_models')
    os.makedirs(top_dir, exist_ok=True)

    print("\n--- TOP 10 MODELS SUMMARY ---")
    for rank_idx, r in enumerate(top_10):
        m_id = r['model_id']
        seed = r['seed']
        survived = r['months_survived']
        score = r['rank_score']
        trades = r['total_trades']
        pnl = r['total_pnl']

        print(f"Rank {rank_idx+1}: Model ID {m_id} | Seed {seed} | Survived: {survived}/{num_months} months | Score: {score:.4f} | Trades: {trades} | PnL: {pnl:.4f}")

        # Instantiate and save PyTorch state_dict & config
        torch.manual_seed(seed)
        model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_classes=3)

        save_data = {
            'rank': rank_idx + 1,
            'model_id': m_id,
            'seed': seed,
            'input_dim': input_dim,
            'hidden_dim': 64,
            'num_classes': 3,
            'seq_len': seq_len,
            'state_dict': model.state_dict(),
            'metrics': r
        }
        model_path = os.path.join(top_dir, f'top_model_rank_{rank_idx+1}_id_{m_id}.pt')
        torch.save(save_data, model_path)

    print(f"\nSuccessfully saved top 10 model weights and configurations to {top_dir}")

if __name__ == '__main__':
    run_simulation()
