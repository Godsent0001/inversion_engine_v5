import os
import sys
import time
import gc
import torch
import numpy as np
import pandas as pd

# Ensure project path in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from inversion_engine_v5.research.core.feature_engine_fast import build_all_features_fast
from inversion_engine_v5.research.core.gru_sim_engine import GRUClassifier, simulate_trading_numba

def load_year1_m1(data_path):
    print("Step 1: Reading Year 1 Out-Of-Sample M1 data (Oct 2024 – Sep 2025)...")
    chunks = []
    for chunk in pd.read_csv(data_path, chunksize=100000):
        time_str = chunk['time'].str[:10]
        m = (time_str >= '2024-10-01') & (time_str <= '2025-09-30')
        if m.any():
            chunks.append(chunk[m])

    df_y1 = pd.concat(chunks, ignore_index=True)
    df_y1['time'] = pd.to_datetime(df_y1['time'])
    del chunks
    gc.collect()

    return df_y1.sort_values('time').reset_index(drop=True)

def run_year1_oos_evaluation():
    data_path = 'inversion_engine_v5/research/data/raw/markets/XAUUSD/XAUUSD_M1.csv'
    markets_dir = 'inversion_engine_v5/research/data/raw/markets'
    output_dir = 'inversion_engine_v5/outputs'
    top_dir = os.path.join(output_dir, 'top_10_gru_models')

    df_y1 = load_year1_m1(data_path)
    print(f"Year 1 OOS period: {df_y1['time'].min()} to {df_y1['time'].max()}, Total bars: {len(df_y1)}")

    print("Step 2: Building vectorized feature matrix for Year 1 OOS...")
    t0 = time.time()
    X_np, cleaned_df = build_all_features_fast(df_y1, markets_dir=markets_dir)
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

    del cleaned_df, df_y1
    gc.collect()

    seq_len = 30
    num_months = len(unique_ym)

    print("Step 3: Preparing monthly slice indices and pre-unfolding sequence tensors...")
    monthly_slices = {}
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

    print("\nStep 4: Evaluating Top 10 Models on Year 1 OOS Data...")
    pt_files = sorted([f for f in os.listdir(top_dir) if f.endswith('.pt')])

    results_y1 = []
    torch.set_num_threads(4)

    for pt_file in pt_files:
        pt_path = os.path.join(top_dir, pt_file)
        ckpt = torch.load(pt_path)

        rank = ckpt['rank']
        m_id = ckpt['model_id']
        seed = ckpt['seed']
        y2_metrics = ckpt['metrics']
        y2_pnl = y2_metrics['total_pnl'] * 100.0

        # Instantiate model and load state_dict
        model = GRUClassifier(input_dim=input_dim, hidden_dim=64, num_classes=3)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

        disqualified = False
        months_survived = 0

        monthly_pnls = np.zeros(num_months + 1, dtype=np.float64)
        monthly_sharpes = np.zeros(num_months + 1, dtype=np.float64)
        monthly_counts = np.zeros(num_months + 1, dtype=np.int32)
        monthly_mdds = np.zeros(num_months + 1, dtype=np.float64)

        with torch.inference_mode():
            for m in range(1, num_months + 1):
                if m not in monthly_slices:
                    continue

                ms = monthly_slices[m]
                X_m = X_tensor[ms['start_idx']:ms['end_idx']]
                X_m_seq = X_m.unfold(0, seq_len, 1).transpose(1, 2)

                probs = model(X_m_seq)
                preds = torch.argmax(probs, dim=-1).numpy()
                del X_m, X_m_seq, probs

                warmup_offset = ms['warmup_offset']
                m_preds = preds[warmup_offset:] if warmup_offset > 0 else preds

                m_disq, m_surv, m_pnls, m_sharpes, m_cnts, m_mdds = simulate_trading_numba(
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
                mdd_m = m_mdds[m]

                monthly_pnls[m] = pnl_m
                monthly_sharpes[m] = sharpe_m
                monthly_counts[m] = cnt_m
                monthly_mdds[m] = mdd_m

                if pnl_m < 0.0 or m_disq:
                    disqualified = True

                months_survived += 1

        total_pnl = np.sum(monthly_pnls[1:13])
        total_trades = np.sum(monthly_counts[1:13])

        res = {
            'y2_rank': rank,
            'model_id': m_id,
            'seed': seed,
            'y2_is_pnl_pct': y2_pnl,
            'y1_oos_months_survived': months_survived,
            'y1_oos_disqualified': disqualified,
            'y1_oos_total_trades': int(total_trades),
            'y1_oos_total_pnl_pct': float(total_pnl * 100.0)
        }

        for m in range(1, num_months + 1):
            res[f'y1_pnl_m{m}'] = float(monthly_pnls[m])
            res[f'y1_pnl_pct_m{m}'] = float(monthly_pnls[m] * 100.0)
            res[f'y1_mdd_pct_m{m}'] = float(monthly_mdds[m])
            res[f'y1_sharpe_m{m}'] = float(monthly_sharpes[m])
            res[f'y1_trades_m{m}'] = int(monthly_counts[m])

        results_y1.append(res)
        print(f"Rank {rank} (Model {m_id}): Y2 IS PnL = {y2_pnl:.2f}% | Y1 OOS PnL = {total_pnl*100:.2f}% | Y1 Disqualified = {disqualified} | Y1 Trades = {total_trades}")

    results_y1.sort(key=lambda r: r['y2_rank'])

    df_y1_res = pd.DataFrame(results_y1)
    csv_y1_path = os.path.join(output_dir, 'top_10_models_year1_oos_metrics.csv')
    df_y1_res.to_csv(csv_y1_path, index=False)
    print(f"\nSaved Year 1 out-of-sample metrics for top 10 models to {csv_y1_path}")

if __name__ == '__main__':
    run_year1_oos_evaluation()
