"""Pull fresh daily prices for every asset in every tier and write
`data/default_prices.parquet`. Run once when you want a fresh cache.

The cached panel also includes Vanguard mutual-fund predecessors so the
"long-term model" toggle in the app can splice history without a live
network call. See `compose_lib/returns.py::SPLICE_MAP`.
"""

from __future__ import annotations

import sys

from compose_lib.data_fetch import all_tickers, fetch_prices, save_default_panel


# Vanguard mutual-fund predecessors used by the long-term-model splice path.
SPLICE_PREDECESSORS = ["VFINX", "VBMFX", "VFITX", "VWESX",
                       "VEIEX", "VGTSX", "VUSTX"]


def main() -> int:
    tickers = sorted(set(all_tickers()) | set(SPLICE_PREDECESSORS))
    print(f"Fetching {len(tickers)} tickers from Yahoo: {tickers}")
    df = fetch_prices(tickers, start="1980-01-01")
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        print(f"WARNING: missing from Yahoo response: {missing}", file=sys.stderr)
    save_default_panel(df)
    print(
        f"Saved {df.shape[0]} rows × {df.shape[1]} cols  "
        f"({df.index.min().date()} → {df.index.max().date()})"
    )
    per_col = {c: int(df[c].notna().sum()) for c in df.columns}
    for c, n in sorted(per_col.items(), key=lambda kv: kv[1]):
        print(f"  {c:8s}  {n} obs  first={df[c].first_valid_index().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
