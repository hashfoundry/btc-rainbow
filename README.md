# btc-rainbow

Bitcoin rainbow charts built from thirteen different regression models — one
chart per model, plus a ranked comparison page.

## Quick start

```bash
docker compose up
```

That refreshes the price history from a public exchange, refits every model and
writes the charts to `./output/`. Open [`output/index.html`](output/index.html)
for the comparison table, or any `output/<model>.html` for a single chart.

To browse them over HTTP instead of opening files directly:

```bash
docker compose --profile serve up     # http://localhost:8080
```

Rerun `docker compose up` any time to pull the latest daily closes and rebuild.

## Models

| Key | Model | Notes |
|---|---|---|
| `power_law` | Power Law Corridor | Log-log OLS; the standard Santostasi fit |
| `power_law_quantile` | Power Law – Quantile Corridor | Bands at residual quantiles |
| `power_law_robust` | Power Law – Robust (Theil–Sen) | Insensitive to bubble outliers |
| `log_time` | Logarithmic Regression (original) | This chart's original fit, kept as baseline |
| `stretched_log` | Modified Power Law | Adds a curvature exponent |
| `time_offset_log` | Time-Adjusted Log Regression | Fits the time origin |
| `log_linear` | Log-Linear (Exponential Trend) | Control case; fits poorly by design |
| `hyperbolic` | Hyperbolic (Finite-Time Singularity) | Degenerates — see the caveat in the model doc |
| `logistic` | Logistic Growth (S-Curve) | Saturating adoption |
| `gompertz` | Gompertz Growth | Asymmetric saturation |
| `s2f` | Stock-to-Flow | Scarcity model; staircase rainbow |
| `lppl` | Log-Periodic Power Law | Sornette bubble signature |
| `halving_cycle` | Halving-Cycle Regression | Best fit; rainbow undulates with the cycle |

The mathematics, fitting method and failure modes of each are documented in
[bitcoin_rainbow_chart_regression_models.md](bitcoin_rainbow_chart_regression_models.md).

## Configuration

Set these in the environment or in a `.env` file next to `docker-compose.yml`:

| Variable | Default | Purpose |
|---|---|---|
| `MODELS` | `all` | Comma-separated model keys, e.g. `power_law,lppl,s2f` |
| `EXTEND_MONTHS` | `9` | Length of the projection window |
| `EXCHANGES` | `binance,coinbase,kraken,bitstamp` | Tried in order until one answers |
| `SKIP_UPDATE` | `0` | Set to `1` to plot the CSV as-is, with no network access |
| `VIEWER_PORT` | `8080` | Port for the optional `serve` profile |

Binance returns HTTP 451 in some jurisdictions; the remaining exchanges are
fallbacks, so the default list works from most places.

```bash
MODELS=power_law,halving_cycle docker compose up
```

## Running without Docker

```bash
pip install -r requirements.txt
python src/main.py                    # all models -> ./output
python src/main.py --list-models
python src/main.py --models power_law,lppl --no-update --extend-months 24
```

## Layout

| Path | Purpose |
|---|---|
| [src/main.py](src/main.py) | CLI entry point; builds every chart and the index page |
| [src/models.py](src/models.py) | Model registry, fitting, and rainbow band construction |
| [src/data.py](src/data.py) | CSV loading and exchange top-up |
| [src/plot.py](src/plot.py) | Plotly rendering, shared by every model |
| [src/btc_supply.py](src/btc_supply.py) | Issuance schedule and stock-to-flow |
| `data/bitcoin_data.csv` | Price history; updated in place on each run |
| `output/` | Generated charts (git-ignored) |

The `bitcoin_rainbow_chart.html` at the repository root is the previous
single-chart artifact and is no longer regenerated. Its replacement is
`output/bitcoin_rainbow_chart.html`, which is rebuilt from the `log_time` model
on every run.

## Caveats

Every model here is fitted to history and assumes the future resembles it. They
disagree with each other by roughly a factor of two on present fair value and
diverge much further on projection. R² measures how closely a model tracks the
past, not how well it predicts — the most flexible curves score highest because
they overfit most eagerly. Not financial advice.
