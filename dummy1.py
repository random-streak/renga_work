import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


def find_existing_path(measurements):
    for p in measurements:
        pp = Path(p)
        if pp.exists():
            return pp
    return None


BASE = Path(__file__).resolve().parent


actual_measurements = [
    Path('/home/renga/Desktop/data_neoen/data/measurements_neoen_morcenx_sep2025.csv'),
]

actuals_path = find_existing_path(actual_measurements)
if actuals_path is None:
    print('ERROR: could not locate the actuals CSV. Looked in:')
    for c in actual_measurements:
        print('  -', c)
    sys.exit(1)

# Load actuals and infer datetime index column
df_actuals = pd.read_csv(actuals_path)

def infer_datetime_index(df):
    # prefer common names
    measurements = [c for c in df.columns if str(c).lower() in ('measure_date', 'date', 'timestamp', 'time')]
    if measurements:
        col = measurements[0]
    else:
        # fallback: try to parse any column that looks datetime-like on a sample
        col = None
        for c in df.columns:
            try:
                sample = pd.to_datetime(df[c].dropna().iloc[:10], errors='coerce')
                if sample.notna().any():
                    col = c
                    break
            except Exception:
                continue
    if col is None:
        raise RuntimeError('No datetime-like column found in actuals CSV; please inspect columns: ' + ','.join(map(str, df.columns)))
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df = df.set_index(col)
    return df

try:
    actuals = infer_datetime_index(df_actuals)
except Exception as e:
    print('Failed to process actuals file:', e)
    sys.exit(1)

# Find forecast files under the repository tree (robust rglob)
forecast_files = sorted([str(p) for p in BASE.rglob('forecast_neoen_morcenx_*.csv')])
if not forecast_files:
    print('No forecast files found under', BASE, "(pattern: 'forecast_neoen_morcenx_*.csv')")
    sys.exit(0)


def infer_column(cols, preferred):
    lc = {c.lower(): c for c in cols}
    for p in preferred:
        if p in lc:
            return lc[p]
    # otherwise try to return first match by partial
    for c in cols:
        if any(k in c.lower() for k in preferred):
            return c
    return None


def evaluate_forecast(forecast_file, actuals):
    p = Path(forecast_file)
    fname = p.name  # e.g. 'forecast_202507071515.csv'
    timestamp_str = fname.replace('forecast_', '').replace('.csv', '')
    run_time = None
    for fmt in ("%Y%m%d%H%M", "%Y%m%d%H%M%S"):
        try:
            run_time = datetime.strptime(timestamp_str, fmt)
            break
        except Exception:
            continue

    df = pd.read_csv(forecast_file)

    # find the target time column
    target_col = infer_column(df.columns, ['target_time', 'time', 'timestamp', 'date'])
    if target_col is None:
        raise RuntimeError(f'No target time column found in forecast file {forecast_file}; columns: {list(df.columns)}')
    df['target_time_parsed'] = pd.to_datetime(df[target_col], errors='coerce')

    # find forecast value columns
    det_col = infer_column(df.columns, ['deterministic', 'det'])
    p50_col = infer_column(df.columns, ['p50', 'median'])

    if det_col is None and p50_col is None:
        raise RuntimeError(f'No deterministic or p50/median forecast columns found in {forecast_file}; columns: {list(df.columns)}')

    # find actuals value column
    actual_col = infer_column(actuals.columns, ['actual', 'value', 'measurement'])
    if actual_col is None:
        # if only one column exists in actuals, use it
        if len(actuals.columns) == 1:
            actual_col = actuals.columns[0]
        else:
            raise RuntimeError('Could not find an "actual" column in actuals data; columns: ' + ','.join(map(str, actuals.columns)))

    # merge on equality of timestamps
    merged = df.merge(actuals, left_on='target_time_parsed', right_index=True, how='inner', suffixes=('_f', '_a'))
    if merged.empty:
        # no matching timestamps, skip
        return None

    summaries = {}
    if det_col and det_col in merged:
        summaries['mae_det'] = (merged[det_col] - merged[actual_col]).abs().mean()
    else:
        summaries['mae_det'] = None

    if p50_col and p50_col in merged:
        summaries['mae_prob'] = (merged[p50_col] - merged[actual_col]).abs().mean()
    else:
        summaries['mae_prob'] = None

    return {
        'forecast_run': run_time,
        'mae_det': summaries['mae_det'],
        'mae_prob': summaries['mae_prob']
    }


# Iterate over all forecast runs
results = []
for file in forecast_files:
    try:
        summary = evaluate_forecast(file, actuals)
        if summary is None:
            print('No matching timestamps for', file, '- skipping')
            continue
        results.append(summary)
    except Exception as e:
        print('Error processing', file, ':', e)
        continue

if not results:
    print('No results to summarize (no forecast had matching timestamps or all failed).')
    sys.exit(0)

results_df = pd.DataFrame(results)
results_df = results_df.dropna(subset=['forecast_run'])
results_df = results_df.sort_values('forecast_run')

plt.figure(figsize=(10, 5))
if results_df['mae_det'].notna().any():
    plt.plot(results_df['forecast_run'], results_df['mae_det'], label='Deterministic MAE', color='r')
if results_df['mae_prob'].notna().any():
    plt.plot(results_df['forecast_run'], results_df['mae_prob'], label='Probabilistic MAE (p50)', color='b')
plt.xlabel('Forecast Run Time')
plt.ylabel('MAE')
plt.title('Forecast Accuracy Over Time')
plt.legend()
plt.show()