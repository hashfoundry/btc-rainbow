# Bitcoin Rainbow Chart Regression Models

This document describes every regression model implemented in this repository:
the mathematics, how each one is actually fitted, what it is good for, and where
it breaks down. Each model is rendered as its own chart — see
[`src/models.py`](src/models.py) for the implementations.

## Introduction

Bitcoin rainbow charts are long-horizon valuation tools. A regression is fitted
through the price history in log space, and coloured bands are laid around that
trend to indicate whether price is historically cheap or expensive relative to
it. The bands are a *description of past dispersion*, not a forecast.

## Common construction

Every model here produces a centre line in natural-log price space. Bands are
then built generically from the residuals, which makes the models directly
comparable:

- **σ bands** (default) — each band is 0.5 residual standard deviations tall,
  so the nine-band rainbow spans ±2.25σ around the trend.
- **Quantile bands** — band edges sit at empirical residual quantiles (1st to
  99th percentile), so each band holds a known share of history.

Time is measured as **days since the genesis block** (2009-01-03) unless stated
otherwise.

## Comparing the models honestly

R² measures how closely a model tracks the past, not how well it predicts, and it
can never fall when a nested model gains a parameter. Ranking on it alone rewards
flexibility. Three corrections are applied; the implementation is in
[`src/metrics.py`](src/metrics.py).

### The autocorrelation problem

These are daily observations of a slow trend, so residuals are enormously
autocorrelated: lag-1 runs 0.995–0.999 and Durbin–Watson sits at 0.002–0.011
instead of 2. Price stays on one side of the fitted curve for months at a time.

Textbook AIC/BIC assume independent observations. With n = 5,834 the complexity
penalty (2k, or ln(n)·k ≈ 8.7k) is swamped by a likelihood built from thousands
of points that are nothing like independent. Measured directly: naive ΔAIC ranks
the models in **exactly** the R² order with no inversions, and puts the
6-parameter halving-cycle model 2,367 units ahead of the 2-parameter power law —
against a penalty difference of 8. That is not a preference, it is an artefact.

**Adjusted R² does not fix it either.** The largest correction across all
thirteen models is 8.3 × 10⁻⁵, against an R² spread of 0.134 — three orders of
magnitude too small to change any ranking. It is reported anyway, because it is
the usual suggested remedy and watching it do nothing is informative.

### The correction: effective sample size

The likelihood is discounted to the number of *independent* observations the
residuals are worth, from the integrated autocorrelation time. That is about
**22, not 5,834** — residuals decorrelate over roughly 260 days, so sixteen years
of daily data carry about two dozen independent looks at the trend.

This single change reverses the answer. The halving-cycle model keeps the best R²
but drops to sixth on ΔAICc; the two-parameter power law comes first.

One implementation detail matters: n_eff is taken from **one fixed reference fit**
and applied to every model. Deriving it per-model is circular and actively
perverse — a model that fits badly leaves more autocorrelated residuals, which
shrinks its own n_eff and therefore its own penalty. Scored that way, the log-linear
and hyperbolic models (the two worst) rank first.

The conclusion is not an artefact of where the autocorrelation sum is truncated.
Varying the cutoff from 0.01 to 0.2 moves n_eff only between 22.4 and 23.2 and
leaves the ranking untouched; even at an implausibly loose 0.4 (n_eff = 28.7) the
power law still comes first and the halving-cycle model still does not.

**It is, however, sensitive to the choice of estimator, and that should be
stated plainly.** ΔAICc between the halving-cycle model and the power law crosses
zero at n_eff ≈ 30. Estimators that stay close to the integrated autocorrelation
time agree with the shipped value — Geyer initial-positive sequence 22.0,
first-zero crossing 22.4, batch means at 500 days 21.9 — and all sit well below
the crossover. But an AR(1) rule gives 11 (which would penalise complexity even
harder), while Sokal automatic windowing gives 50 and short-block batch means
give 40+, both of which would put the halving-cycle model first.

So the finding "the power law's simplicity wins" is robust across the
τ-estimators closest to what is being measured, but it is not a knife-edge-free
result. The defensible claim is the weaker one: once the penalty is charged
against anything like the real information content, the 6-parameter models lose
their apparently overwhelming margin and land in a statistical tie — not that
they are decisively beaten.

### Out-of-sample walk-forward

The only test immune to the parameter-count objection. An expanding window (these
models all claim to describe history since genesis, so discarding early data
would misrepresent them); origins every 182 days from 2015; each model refitted
on data up to the origin, then scored on the price 365 days later.

## Model reference

Measured on daily closes through 2026-08-06 (5,834 observations, n_eff ≈ 22.4),
sorted by ΔAICc. These shift as new data arrives.

| Key | Model | k | R² | σ | ΔAICc | ΔBIC | OOS RMSE |
|---|---|---|---|---|---|---|---|
| `power_law` | Power Law Corridor | 2 | 0.9598 | 0.694 | **0.00** | 0.00 | 0.718 |
| `power_law_quantile` | Power Law – Quantile Corridor | 2 | 0.9598 | 0.694 | **0.00** | 0.00 | 0.718 |
| `time_offset_log` | Time-Adjusted Logarithmic Regression | 3 | 0.9612 | 0.681 | 2.19 | 2.31 | 0.817 |
| `log_time` | Logarithmic Regression (original) | 3 | 0.9612 | 0.681 | 2.19 | 2.31 | 0.817 |
| `stretched_log` | Modified Power Law (Stretched Log) | 3 | 0.9609 | 0.684 | 2.33 | 2.45 | 0.763 |
| `halving_cycle` | Halving-Cycle Regression | 6 | **0.9732** | 0.566 | 5.33 | 3.31 | 0.644 |
| `power_law_robust` | Power Law – Robust (Theil–Sen) | 2 | 0.9486 | 0.695 | 5.51 | 5.51 | 0.618 |
| `gompertz` | Gompertz Growth | 4 | 0.9559 | 0.727 | 8.44 | 8.31 | 0.973 |
| `logistic` | Logistic Growth (S-Curve) | 4 | 0.9551 | 0.733 | 8.83 | 8.70 | 0.973 |
| `lppl` | Log-Periodic Power Law | 6 | 0.9646 | 0.651 | 11.60 | 9.59 | **0.587** |
| `s2f` | Stock-to-Flow | 2 | 0.9321 | 0.901 | 11.74 | 11.74 | 1.048 |
| `log_linear` | Log-Linear (Exponential Trend) | 2 | 0.8697 | 1.249 | 26.37 | 26.37 | 1.910 |
| `hyperbolic` | Hyperbolic (Finite-Time Singularity) | 3 | 0.8393 | 1.387 | 34.06 | 34.18 | 2.246 |

Three things are worth reading off this table.

**The complexity penalty bites.** `halving_cycle` and `lppl` have the two best
R² scores and rank 6th and 10th on ΔAICc. Their extra parameters do not pay for
themselves.

**But "more parameters ⇒ better R²" is a tendency, not a law.** Spearman(k, R²)
is only +0.55 (p = 0.051), and in 15 of 60 model pairs the model with *more*
parameters scored *worse*. These models are mostly not nested, so a larger k can
simply buy a worse functional form — `logistic` and `gompertz` (k = 4) both lose
to the k = 2 power law, and `hyperbolic` (k = 3) loses to `log_linear` (k = 2).

**The two rankings disagree, and that is the interesting part.** The models AICc
penalises hardest are the ones that forecast best: `lppl` has the worst ΔAICc of
the credible models and the *best* out-of-sample RMSE. Meanwhile
`power_law_robust` has a mediocre in-sample R² (0.9486, third-worst) and the
second-best out-of-sample error — robust estimation gives up in-sample fit
precisely to generalise better.

Treat the out-of-sample column as indicative rather than a verdict: **9 of 12
models are statistically indistinguishable from the top one** under a moving-block
bootstrap, and the ordering moves with the evaluation window. Starting the
backtest in 2013 instead of 2015 drops `halving_cycle` out of the top five;
starting in 2017 puts it first. Only `s2f`, `log_linear` and `hyperbolic` are
separated from the leader with any confidence.

The bootstrap interval is Bonferroni-widened across the comparisons. Each model
is measured against whichever one happened to win on the same 22 windows, so
uncorrected intervals suffer the winner's curse: under a null where every model
is equally good, the uncorrected procedure flags roughly 44% of thirteen models
as "significantly worse than the best". The correction errs toward declaring
ties, which is the right direction for a table this small.

---

### 1. Power Law Corridor — `power_law`

```
log₁₀(P) = a · log₁₀(d) + b        ⇔        P = A · d^a
```

Where `d` = days since the genesis block. A straight line in log-log space, and
the best-documented long-run description of Bitcoin's price. Popularised by
physicist Giovanni Santostasi; published parameters cluster around an exponent
of 5.7 and a log₁₀ intercept near −17. This repository's OLS fit gives
**a ≈ 5.61, b ≈ −16.23**.

A [mechanistic derivation](https://www.sciencedirect.com/science/article/abs/pii/S3050517826000675)
grounds the exponent in network adoption: if users grow as a power of time and
value scales as a power of users (generalised Metcalfe), price is necessarily a
power of time. The power law is therefore the reduced form of the Metcalfe
argument, not a competitor to it.

### 2. Power Law – Quantile Corridor — `power_law_quantile`

Same centre line, different bands. Instead of σ multiples, edges sit at the 1st,
12th, 23rd … 99th percentile of the residual distribution. Because log-price
residuals are not Gaussian — they are skewed by bubble tops — quantile bands
describe the shape of history more faithfully, and each band carries a directly
readable meaning: *price has spent ~11% of its life in this band*.

### 3. Power Law – Robust (Theil–Sen) — `power_law_robust`

Same functional form, fitted with the Theil–Sen estimator: the slope is the
median of all pairwise slopes rather than the least-squares minimiser. With a
breakdown point of ~29% it is largely unmoved by blow-off tops and capitulation
wicks, which is what "excluding extreme outliers" usually means in practice. It
returns a slightly steeper exponent (**5.73** vs 5.61) and a confidence interval
for it.

### 4. Log-Linear (Exponential Trend) — `log_linear`

```
ln(P) = a · d + b        ⇔        P = e^(a·d + b)
```

Constant compound growth against *linear* time. Included as the control case: at
R² 0.87 it is decisively beaten by every log-time model, which is the empirical
reason rainbow charts use log-time. The fit implies ~113% annualised growth
sustained forever, which is its own refutation.

### 5. Log-Log / Logarithmic Regression — `log_time`

```
ln(P) = a · ln(b + i) + c
```

The chart's original fit, kept as the reference baseline. Note it regresses on
the **row counter** `i` rather than calendar days; because the history contains
gaps, this is subtly different from the power law and, on this data, marginally
better (0.9612 vs 0.9598).

### 6. Modified Power Law (Stretched Log) — `stretched_log`

```
ln(P) = a · ln(d)^c + b
```

Adds a curvature exponent so the trend can bend as returns diminish, rather than
holding one fixed slope forever. `c < 1` bends the log-log line downward. Fits
slightly better than the plain power law and projects more conservatively.

### 7. Time-Adjusted Logarithmic Regression — `time_offset_log`

```
ln(P) = a · ln(d − t₀) + b
```

Fits the time origin instead of assuming the genesis block. Bitcoin had no
market price for its first 18 months, so a genesis-anchored clock distorts the
early log-time axis. The fitted offset is **t₀ ≈ +193 days**, and the result is
numerically almost identical to `log_time` — the two arrive at the same curve
from different directions.

### 8. Hyperbolic / Finite-Time Singularity — `hyperbolic`

```
P = a / (b − d)^c
```

Super-exponential growth diverging at a future date `b`. Fitted by scanning `b`
and solving `(ln a, c)` exactly by least squares at each node, since the model is
linear in those two once `b` is fixed.

**This model does not fit Bitcoin's full history, and that is informative.**
Residuals fall monotonically as `b` is pushed further out, with no interior
optimum: the form is convex in log-price while Bitcoin's log price *decelerates*.
Left free, it escapes to `b → ∞`, where `ln(b − d)` is near-linear and the model
silently collapses into the exponential trend (note its R² of 0.84 sits right
beside `log_linear`'s 0.87). The scan is therefore capped at 30 years past the
last observation. Read the reported singularity date as the edge of that cap, not
as an estimate. The model is properly used on individual blow-off segments, where
growth genuinely is super-exponential.

The scan is anchored to the end of the **observed history**, not to the plotted
grid. Anchoring it to the grid — as an earlier version did — let the projection
window, a pure display choice, move the fitted singularity by five years and
change R² in the third decimal, and made the backtest fit a different model than
the chart showed.

It is also the worst forecaster in the set by a wide margin: out-of-sample RMSE
2.25 in log space, with a skill score of −4.07 against a random walk.

### 9. Logistic Growth (S-Curve) — `logistic`

```
ln(P) = ceiling − S / (1 + e^(k(d − d₀)))
```

Adoption saturates toward a ceiling — the direct counterpoint to the unbounded
power law. Parameterised on the **asymptote itself** rather than the usual
amplitude-plus-offset pair: Bitcoin has not visibly saturated, so amplitude and
offset are only weakly identified and a naive fit drifts until it hits whatever
bound it is given. Fitting the ceiling directly keeps the reported number
meaningful. The search is capped at $10M/BTC; the current fit lands well inside
that, near **$130k**, which is the classic (and classically premature) logistic
result.

### 10. Gompertz Growth — `gompertz`

```
ln(P) = ceiling − S + S · e^(−B·e^(−C·τ)),   τ = d / 1000
```

An asymmetric saturation curve: fast early growth, long slow approach to the
ceiling. Research on
[oscillatory growth and lengthening cycles](https://www.tandfonline.com/doi/full/10.1080/23322039.2022.2087287)
finds Gompertz tracks Bitcoin's early data better than the symmetric logistic,
which grows too slowly at the start. It scores marginally higher here and puts
the ceiling somewhat higher too. Same capped-ceiling caveat as above.

### 11. Stock-to-Flow — `s2f`

```
ln(P) = a · ln(S2F) + b,    S2F = circulating supply / annual issuance
```

PlanB's scarcity model. Supply and issuance are computed from the consensus
schedule in [`src/btc_supply.py`](src/btc_supply.py) — block height is
interpolated between the four observed halvings and extrapolated at 144
blocks/day — so no external data source is needed. The fitted exponent is
**≈ 2.79**.

Because S2F is a step function of the halving epoch, the resulting rainbow is a
staircase that jumps at each halving rather than a smooth corridor.

**Known critique:** the model regresses market cap (price × supply) on
stock-to-flow (supply ÷ issuance), so supply appears on both sides and inflates
the apparent R². Circulating supply alone correlates about as strongly with
market cap. Treat the fit as descriptive.

### 12. Log-Periodic Power Law — `lppl`

```
ln(P) = a·ln(d) + b + (d/d_ref)^(−m) · A·cos(ω·ln(d) − φ)
```

The Johansen–Ledoit–Sornette bubble signature: a power-law trend plus an
oscillation that is periodic in *log*-time, which implies market cycles that
lengthen in calendar time as the asset matures.

Fitted by grid search over the log-frequency ω and damping m, solving the
remaining linear coefficients exactly at each node — far more stable than
optimising the full LPPLS form directly. ω is restricted to wavelengths of two
to seven years; unconstrained, the fit prefers a single ~16-year arc that scores
better (R² 0.977) but describes the slow bend of the trend rather than the market
cycle the model exists to capture.

**Measured finding:** across 2–12 year wavelengths the objective is remarkably
flat (R² 0.956–0.962 against the plain power law's 0.960). The log-periodic cycle
signal in Bitcoin's power-law residuals is weak — worth roughly half a percentage
point of R² — with a shallow interior optimum near a six-year present-day
wavelength.

### 13. Halving-Cycle Regression — `halving_cycle`

```
ln(P) = a·ln(d) + b + Σ(k=1,2) [α_k·cos(2πkφ) + β_k·sin(2πkφ)]
```

Where φ is the fraction of the way through the current halving epoch. A
power-law trend plus two Fourier harmonics of the *issuance* cycle, so the bands
visibly tilt with the four-year rhythm instead of averaging over it. The model
stays linear in its coefficients, so it is solved in a single least-squares step.

This is the **best-fitting model in the set (R² 0.973)** and produces the most
striking chart — a rainbow that undulates through each cycle. The fitted peak
sits about **36% of the way through each epoch**, i.e. roughly 1.3 years after a
halving, consistent with the historical top timing.

The obvious caveat: it is fitted on only four halving epochs, so it is the most
overfit model here. It assumes the cycle keeps its shape and amplitude as the
issuance change becomes economically negligible.

## Models considered and not implemented

**Metcalfe's Law / generalised Metcalfe** (`P ∝ U^β` for network size `U`)
requires an active-address or user-count series that no free, reliable, key-less
API provides. Fitting it against a *modelled* user curve rather than observed
network data would simply reproduce the logistic or the power law with extra
steps — the [mechanistic derivation](https://www.sciencedirect.com/science/article/abs/pii/S3050517826000675)
shows generalised Metcalfe scaling with power-law adoption collapses exactly into
`power_law`. It is documented here rather than faked.

**Full LPPLS with a critical time `t_c`** was reduced to the residual-oscillation
form in `lppl`. Fitting `t_c` jointly is notoriously unstable and normally
requires many restarts over rolling windows to produce a confidence indicator,
which is a different tool from a rainbow chart.

## Limitations

Every model here is fitted to history and assumes the future resembles it. They
disagree with each other by a factor of two on present fair value, and they
diverge far more on projection. That spread is the honest signal: use the
rainbow as a gauge of where price sits relative to its own past, not as a price
prediction.

Specific caveats on the statistics:

- **The effective sample size is an estimate, not a constant of nature.** It is
  robust to the truncation cutoff but not to the choice of τ-estimator, and the
  ΔAICc winner changes above n_eff ≈ 30 (see above). It is used to make the
  complexity penalty roughly the right order of magnitude, not to support
  fine-grained claims — which is why no Akaike weights or p-values are derived
  from it.
- **Five fits terminate on a bound**, so the nominal `k` overstates what the data
  actually identified: `stretched_log` (c at its lower bound), `logistic` and
  `gompertz` (log-span at 60), `lppl` (damping at 0) and `hyperbolic` (horizon
  cap). Each bound exists for a documented reason, but a parameter pinned by a
  constraint is not a parameter the data estimated.
- **The out-of-sample ordering is unstable.** 22 overlapping forecast windows is
  not many, most models are statistically indistinguishable from the leader, and
  the ranking shifts with the evaluation period.
- **Small residual leakage in the backtest.** `btc_supply.HEIGHT_ANCHORS`
  hard-codes the four observed halving dates, so a refit at a 2016 origin
  technically "knows" when the 2020 and 2024 halvings landed. This affects only
  `s2f` and `halving_cycle`, and only through the placement of future halvings —
  never through price. Halving dates are derivable from the block schedule years
  in advance, so the bias is days of timing, not knowledge of the outcome.
- **AIC assumes a Gaussian likelihood**, which sits awkwardly with
  `power_law_robust`: Theil–Sen does not minimise squared error, so it is
  scored by a criterion it was not built to win. Its poor ΔAICc and strong
  out-of-sample result should be read in that light.

None of this is financial advice.

## Sources

- [Bitcoin Power Law Corridor (Santostasi) — TradingView](https://www.tradingview.com/script/HaCjQX68-Bitcoin-Power-Law-Corridor-Santostasi/)
- [Bitcoin: Power Law — Bitcoin Magazine Pro](https://www.bitcoinmagazinepro.com/charts/bitcoin-power-law/)
- [Bitcoin Rainbow Chart — BlockchainCenter](https://www.blockchaincenter.net/bitcoin-rainbow-chart/)
- [A mechanistic derivation of the Bitcoin price power law — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S3050517826000675)
- [Are Bitcoin bubbles predictable? Generalised Metcalfe's Law and LPPLS — Royal Society Open Science](https://royalsocietypublishing.org/doi/10.1098/rsos.180538)
- [A Bitcoin price prediction model assuming oscillatory growth and lengthening cycles — Cogent Economics & Finance](https://www.tandfonline.com/doi/full/10.1080/23322039.2022.2087287)
- [Bitcoin Stock-to-Flow (S2F) Model Explained — CoinGecko](https://www.coingecko.com/learn/bitcoin-stock-to-flow-model-explained)
- [lppls — reference implementation of the LPPLS model](https://github.com/Boulder-Investment-Technologies/lppls)
