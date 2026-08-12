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

## How the models are compared

Ranking by R² would be misleading, because R² can never fall when a nested model
gains a parameter. Every run therefore prints three tables to stdout and renders
the same three as HTML **below every chart** and on `index.html`.

Each table covers **all models**, with the chart's own model highlighted, so any
single chart can be read against its alternatives instead of on its own. The
footnote under the plot keeps only what is specific to that fit — its R², σ,
today's price against fair value, and the fitted parameter values.

**Fit and complexity.** R², adjusted R², σ, median absolute % error, and ΔAIC /
ΔAICc / ΔBIC. The information criteria are charged against an **effective sample
size**, not the raw row count — see below. Ranked by ΔAICc; a gap under ~2 is a
tie.

**Residual diagnostics and band calibration.** Durbin–Watson, lag-1
autocorrelation, skew and excess kurtosis, plus what share of history actually
lands inside the rainbow and how far each band's occupancy strays from its
nominal share. The σ bands assume log residuals are roughly normal, so skew is
not cosmetic: it means the bands are miscalibrated.

**Out-of-sample predictive power.** A walk-forward backtest with an expanding
window: at each origin every model is refitted on data up to that date only, then
scored on the price a year later. Reports RMSE and bias in log space, median
absolute % error, band coverage, and skill against a random-walk baseline, with a
moving-block bootstrap interval for each model's gap to the best.

### Why the effective sample size matters

These are daily observations of a slow trend, so residuals are enormously
autocorrelated — lag-1 above 0.99, Durbin–Watson near 0.01 instead of 2. Price
stays on one side of the fitted curve for months. Textbook AIC assumes
independent observations, so with n = 5,834 the complexity penalty is swamped:
measured ΔAIC between the 6-parameter halving-cycle model and the 2-parameter
power law is over 2,300, which is not a real preference — naive AIC reproduces
the R² ranking exactly and penalises nothing.

The likelihood is therefore discounted to the effective sample size implied by
the residual decorrelation time, which is about **22 observations, not 5,834**.
That single change flips the answer: the halving-cycle model has the best R² but
falls to sixth on ΔAICc, and the two-parameter power law comes first. Adjusted R²
does *not* fix this — with n in the thousands and k ≤ 6 it moves the fifth
decimal — but it is reported anyway, because it is the usual suggestion and
seeing it move nothing is the point.

The effective sample size comes from one fixed reference fit and is applied to
every model. Deriving it per-model is circular and perverse: a model that fits
badly leaves more autocorrelated residuals, shrinking its own n_eff and therefore
its own penalty. Scored that way, the two worst models here rank first.

### Read the rankings together, not separately

ΔAICc and the backtest disagree, and neither is the last word. On this data the
power law wins on ΔAICc while the log-periodic model forecasts best. The backtest
ordering is also unstable: starting it in 2013 instead of 2015 drops the
halving-cycle model out of the top five, and starting in 2017 puts it first. With
only ~22 overlapping forecast windows, 9 of 12 models are statistically
indistinguishable from the top one, and the output says which.

The defensible conclusion is the modest one: once complexity is charged against
anything like the real information content, the 6-parameter models lose their
apparently overwhelming margin and land in a tie — not that they are decisively
beaten. The ΔAICc winner does change if the effective sample size is estimated
above ~30, and some estimators of it are. That caveat is documented in
[the model reference](bitcoin_rainbow_chart_regression_models.md).

## Configuration

Set these in the environment or in a `.env` file next to `docker-compose.yml`:

| Variable | Default | Purpose |
|---|---|---|
| `MODELS` | `all` | Comma-separated model keys, e.g. `power_law,lppl,s2f` |
| `EXTEND_MONTHS` | `9` | Length of the projection window |
| `EXCHANGES` | `binance,coinbase,kraken,bitstamp` | Tried in order until one answers |
| `SKIP_UPDATE` | `0` | Set to `1` to plot the CSV as-is, with no network access |
| `SKIP_BACKTEST` | `0` | Set to `1` to skip the walk-forward backtest (the slowest step, ~45s) |
| `BACKTEST_HORIZON_DAYS` | `365` | Forecast horizon for the backtest |
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
| [src/metrics.py](src/metrics.py) | Fit statistics, information criteria, walk-forward backtest |
| [src/report.py](src/report.py) | Shared stylesheet, comparison tables, chart and index page shells |
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
