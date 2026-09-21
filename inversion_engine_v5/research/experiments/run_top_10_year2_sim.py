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

def load_year2_m1(data_path):
    print("Step 1: Reading Year 2 M1 data (Oct 2025 – Sep 2026)...")
    chunks = []
    for chunk in pd.read_csv(data_path, chunksize=100000):
        time_str = chunk['time'].str[:10]
        m = (time_str >= '2025-10-01') & (time_str <= '2026-09-30')
        if m.any():
            chunks.append(chunk[m])

    df_y2 = pd.concat(chunks, ignore_index=True)
    df_y2['time'] = pd.to_datetime(df_y2['time'])
    del chunks
    gc.collect()

    return df_y2.sort_values('time').reset_index(drop=True)

def run_year2_evaluation():
    data_path = 'inversion_engine_v5/research/data/raw/markets/XAUUSD/XAUUSD_M1.csv'
    markets_dir = 'inversion_engine_v5/research/data/raw/markets'
    output_dir = 'inversion_engine_v5/outputs'
    top_dir = os.path.join(output_dir, 'top_10_gru_models')

    df_y2 = load_year2_m1(data_path)
    print(f"Year 2 period: {df_y2['time'].min()} to {df_y2['time'].max()}, Total bars: {len(df_y2)}")

    print("Step 2: Building ultra-fast vectorized feature matrix for Year 2...")
    t0 = time.time()
    X_np, cleaned_df = build_all_features_fast(df_y2, markets_dir=markets_dir)
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

    opens = cleaned_df['open'].values.astype(np.float64)
    highs = cleaned_df['high'].values.astype(np.float64)
    lows = cleaned_df['low'].values.astype(np.float64)
    closes = cleaned_df['close'].values.astype(np.float64)
    atrs = cleaned_df['high'].sub(cleaned_df['low']).rolling(14).mean().ffill().bfill().values.astype(np.float64)
    spreads = cleaned_df['spread'].values.astype(np.float64) if 'spread' in cleaned_df.columns else np.zeros(len(cleaned_df))

    del cleaned_df, df_y2
    gc.collect()

    seq_len = 10
    num_months = len(unique_ym)

    print("Step 3: Preparing monthly slice indices...")
    monthly_slices = {}
    for m in range(1, num_months + 1):
        m_mask = (month_ids == m)
        m_indices = np.where(m_mask)[0]
        if len(m_indices) == 0:
            continue

        start_idx = max(0, m_indices[0] - seq_len + 1)
        end_idx = m_indices[-1] + 1
        warmup_offset = m_indices[0] - start_idx

        monthly_slices[m] = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'warmup_offset': warmup_offset,
            'm_indices': m_indices,
            'win_ids': window_ids[m_indices],
            'm_ids': month_ids[m_indices],
            'opens': opens[m_indices],
            'highs': highs[m_indices],
            'lows': lows[m_indices],
            'closes': closes[m_indices],
            'atrs': atrs[m_indices],
            'spreads': spreads[m_indices]
        }

    print("\nStep 4: Evaluating Top 10 Models on Year 2 Data...")
    pt_files = sorted([f for f in os.listdir(top_dir) if f.endswith('.pt')])

    results_y2 = []
    torch.set_num_threads(4)

    for pt_file in pt_files:
        pt_path = os.path.join(top_dir, pt_file)
        ckpt = torch.load(pt_path)

        rank = ckpt['rank']
        m_id = ckpt['model_id']
        seed = ckpt['seed']
        y1_metrics = ckpt['metrics']
        y1_score = y1_metrics['rank_score']

        # Instantiate model and load state_dict
        model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_classes=3)
        model.load_state_dict(ckpt['state_dict'])
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
                    ms['opens'],
                    ms['highs'],
                    ms['lows'],
                    ms['closes'],
                    ms['atrs'],
                    ms['spreads']
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
        else:
            avg_monthly_sharpe = float(np.mean(monthly_sharpes[1:months_survived + 1])) if months_survived > 0 else -10.0

        res = {
            'y1_rank': rank,
            'model_id': m_id,
            'seed': seed,
            'y1_score': y1_score,
            'y2_months_survived': months_survived,
            'y2_disqualified': disqualified,
            'y2_total_trades': int(total_trades),
            'y2_total_pnl': float(total_pnl),
            'y2_avg_monthly_sharpe': float(avg_monthly_sharpe)
        }

        for m in range(1, num_months + 1):
            res[f'y2_pnl_m{m}'] = float(monthly_pnls[m])
            res[f'y2_sharpe_m{m}'] = float(monthly_sharpes[m])
            res[f'y2_trades_m{m}'] = int(monthly_counts[m])

        results_y2.append(res)
        print(f"Rank {rank} (Model {m_id}): Y2 Survived = {months_survived}/{num_months} | Y2 PnL = {total_pnl:.4f} | Y2 Avg Sharpe = {avg_monthly_sharpe:.4f} | Y2 Trades = {total_trades}")

    # Sort results by y1_rank ascending
    results_y2.sort(key=lambda r: r['y1_rank'])

    df_y2_res = pd.DataFrame(results_y2)
    csv_y2_path = os.path.join(output_dir, 'top_10_models_year2_metrics.csv')
    df_y2_res.to_csv(csv_y2_path, index=False)
    print(f"\nSaved Year 2 out-of-sample metrics for top 10 models to {csv_y2_path}")

if __name__ == '__main__':
    run_year2_evaluation()
