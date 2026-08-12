"""Build a Bitcoin rainbow chart for every regression model.

Run directly, or via ``docker compose up`` - the charts land in ``OUTPUT_DIR``
with an ``index.html`` that links and ranks them.
"""

import argparse
import html
import os
import sys
from pathlib import Path

import numpy as np

import metrics as metrics_mod
import models
import plot
import report
from data import DEFAULT_EXCHANGES, get_data
from models import build_input, fit_model, resolve_models
from plot import BAND_COLORS, current_band, extend_dates

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
LEGACY_CHART_MODEL = "log_time"
LEGACY_CHART_NAME = "bitcoin_rainbow_chart.html"

RANKING_NOTE = (
    "R² can never fall when a nested model gains a parameter, so on its own it "
    "cannot tell a better model from a merely more flexible one. AICc charges "
    "for every parameter against the effective sample size."
)


def build_charts(
    data_file: str,
    output_dir: str,
    selection: str = "all",
    extend_months: int = plot.EXTEND_MONTHS,
    exchanges=DEFAULT_EXCHANGES,
    update: bool = True,
    backtest: bool = True,
    horizon_days: int = metrics_mod.BACKTEST_HORIZON_DAYS,
):
    """Fit every selected model and write one chart each, plus an index.

    Returns:
        A list of per-model result dicts, ranked most-credible first.
    """
    specs = resolve_models(selection)
    if not Path(data_file).is_file():
        raise ValueError(
            f"Price history not found at {data_file}. "
            "Set DATA_FILE / --data-file, or mount the data directory."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_data = get_data(data_file, exchanges=exchanges, update=update)
    extended = extend_dates(raw_data, months=extend_months)
    model_input = build_input(raw_data, extended)

    print(f"\nFitting {len(specs)} model(s) on {len(raw_data):,} daily observations.")
    results = []
    for spec in specs:
        try:
            fit = fit_model(spec, model_input)
        except Exception as exc:
            print(f"  ✗ {spec.name}: {type(exc).__name__}: {exc}")
            results.append({"spec": spec, "fit": None, "metrics": None, "backtest": None,
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(f"  ✓ {spec.name:<40} R² = {fit.r2:>7.4f}   σ = {fit.sigma:.3f}")
        results.append({"spec": spec, "fit": fit, "metrics": None, "backtest": None, "error": None})

    fitted = [r for r in results if r["fit"]]

    # One common effective sample size for every model, so the information
    # criteria compare like with like.
    n_eff = metrics_mod.reference_n_eff(model_input)
    print(
        f"\nEffective sample size: {n_eff:.1f} of {len(raw_data):,} daily observations "
        f"(residuals decorrelate over ~{len(raw_data) / n_eff:.0f} days)."
    )
    for result in fitted:
        result["metrics"] = metrics_mod.evaluate(
            result["fit"], model_input, result["spec"].n_params, n_eff
        )
    metrics_mod.rank({r["spec"].key: r["metrics"] for r in fitted})

    if backtest and fitted:
        origins = metrics_mod.backtest_origins(raw_data, horizon_days)
        print(
            f"\nWalk-forward backtest: {len(origins)} origins × {len(fitted)} models, "
            f"{horizon_days}-day horizon, expanding window..."
        )
        scores = metrics_mod.walk_forward(
            [r["spec"] for r in fitted], raw_data, horizon_days=horizon_days
        )
        for result in fitted:
            result["backtest"] = scores.get(result["spec"].key)

    # Rank by the complexity-penalised criterion rather than raw R².
    results.sort(
        key=lambda r: (r["fit"] is None, r["metrics"].d_aicc if r["metrics"] else float("inf"))
    )

    print(f"\nRendering {len(fitted)} chart(s)...")
    for result in fitted:
        spec, fit = result["spec"], result["fit"]
        fig = plot.create_plot(
            raw_data, fit, extended, months=extend_months,
            stats=result["metrics"], backtest=result["backtest"],
        )
        # The figure is embedded as a fragment rather than written straight out,
        # so the comparison tables can sit underneath it on the same page.
        plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn", config=PLOTLY_CONFIG)
        page = report.chart_page(plot_div, spec, results, n_eff, len(raw_data))

        (out / f"{spec.key}.html").write_text(page, encoding="utf-8")
        if spec.key == LEGACY_CHART_MODEL:
            (out / LEGACY_CHART_NAME).write_text(page, encoding="utf-8")

    print_report(raw_data, results, n_eff)

    index_path = out / "index.html"
    index_path.write_text(
        render_index(raw_data, results, extend_months, n_eff, backtest_ran=backtest),
        encoding="utf-8",
    )
    print(f"\nWrote {index_path} and {len(fitted)} chart(s) to {out}/")
    return results


def _rank_key(backtest) -> float:
    """Sort key for the out-of-sample table.

    Uses the RMSE over the origins every model forecast, so a model that failed
    on the hard origins cannot post a flattering score on an easier subset and
    take the top row. Identical to rmse_log when nothing failed.
    """
    return backtest.rmse_common if np.isfinite(backtest.rmse_common) else backtest.rmse_log


def print_report(raw_data, results, n_eff: float) -> None:
    """Print the comparison tables that make up the container's output."""
    fitted = [r for r in results if r["metrics"]]
    if not fitted:
        return
    n = len(raw_data)

    width = 108

    def rule(w=width):
        print("─" * w)

    print(f"\n\n{'═' * width}")
    print("IN-SAMPLE FIT AND COMPLEXITY")
    print(f"{'═' * width}")
    print(
        f"Information criteria use the effective sample size n_eff = {n_eff:.1f}, not the "
        f"{n:,} daily rows."
    )
    print(
        "Residuals are autocorrelated over months, so the raw count massively overstates the "
        "evidence.\n"
    )
    print(
        f"{'model':<37}{'k':>3}{'R²':>9}{'adj R²':>9}{'σ':>7}{'MedAPE':>8}"
        f"{'ΔAIC':>9}{'ΔAICc':>9}{'ΔBIC':>9}"
    )
    rule()
    for r in fitted:
        m = r["metrics"]
        print(
            f"{r['spec'].name:<37}{m.n_params:>3}{m.r2:>9.4f}{m.adj_r2:>9.4f}"
            f"{m.sigma:>7.3f}{m.medape:>7.1f}%{m.d_aic:>9.2f}{m.d_aicc:>9.2f}"
            f"{m.d_bic:>9.2f}"
        )
    rule()
    print("Sorted by ΔAICc (0 = best; a gap under ~2 is a tie).")
    print(
        "R² can never fall when a nested model gains a parameter, so on its own it cannot "
        "tell a\nbetter model from a merely more flexible one. AICc charges for every "
        "parameter against the\neffective sample size. Adjusted R² is shown too because it "
        "is the usual suggestion, but note\nit barely moves: with n in the thousands and "
        "k ≤ 6 the correction lands in the fifth decimal."
    )

    best_r2 = max(fitted, key=lambda r: r["metrics"].r2)
    best_aicc = fitted[0]
    if best_r2 is not best_aicc:
        print(
            f"\n'{best_r2['spec'].name}' has the highest R² "
            f"({best_r2['metrics'].r2:.4f}, k={best_r2['metrics'].n_params})\nbut ranks "
            f"#{fitted.index(best_r2) + 1} here — its extra parameters do not pay for "
            f"themselves. Scored naively on all\n{len(raw_data):,} daily rows its ΔAIC would "
            f"be {best_r2['metrics'].d_aic_naive:.0f} against "
            f"{best_aicc['metrics'].d_aic_naive:.0f} for '{best_aicc['spec'].name}' — "
            "an apparently\noverwhelming margin, and exactly the illusion this correction "
            "removes."
        )

    # Needs a handful of models with differing k before the correlation says
    # anything; with one or two it is undefined.
    premise = metrics_mod.parameter_premise(
        [r["metrics"].n_params for r in fitted], [r["metrics"].r2 for r in fitted]
    )
    if len(fitted) >= 4 and premise["pairs"] and np.isfinite(premise["spearman"]):
        print(
            f"\nHow strong is the parameters-buy-fit effect here? Spearman(k, R²) = "
            f"{premise['spearman']:+.2f} (p = {premise['p_value']:.3f}), and in\n"
            f"{premise['inversions']} of {premise['pairs']} model pairs the one with MORE "
            "parameters scored WORSE. The tendency is real\nbut it is not a law: these models "
            "are mostly not nested, so a bigger k can still buy a\nworse functional form."
        )

    print(f"\n\n{'═' * width}")
    print("RESIDUAL DIAGNOSTICS AND BAND CALIBRATION")
    print(f"{'═' * width}")
    print(
        f"{'model':<37}{'Durbin-Watson':>15}{'ρ₁':>9}{'skew':>8}{'ex.kurt':>9}"
        f"{'in-rainbow':>12}{'calib.err':>11}"
    )
    rule(101)
    for r in fitted:
        m = r["metrics"]
        print(
            f"{r['spec'].name:<37}{m.durbin_watson:>15.4f}{m.rho1:>9.4f}{m.skew:>8.2f}"
            f"{m.excess_kurtosis:>9.2f}{m.coverage:>11.1f}%{m.calibration_error:>10.1f}pp"
        )
    rule(101)
    print(
        "Durbin-Watson ≈ 2 would mean independent residuals; these sit near 0, which is why\n"
        "the effective sample size above is a few dozen rather than a few thousand.\n"
        "'in-rainbow' is the share of history inside the bands; 'calib.err' is the mean gap\n"
        "between observed and nominal band occupancy — high values mean the bands lie."
    )

    scored = [r for r in fitted if r["backtest"]]
    if not scored:
        return
    horizon = scored[0]["backtest"].horizon_days
    counts = {r["backtest"].origins for r in scored}
    origins = f"{min(counts)}–{max(counts)}" if len(counts) > 1 else str(min(counts))
    print(f"\n\n{'═' * width}")
    print(f"OUT-OF-SAMPLE PREDICTIVE POWER  ({origins} origins, {horizon}-day horizon)")
    print(f"{'═' * width}")
    print(
        "Each model refitted on data up to each origin, then scored on what it did not see.\n"
        "Immune to the parameter-count objection — a parameter that only memorises the past\n"
        "earns nothing here — but it carries its own uncertainty, quantified below.\n"
    )
    print(
        f"{'model':<37}{'n':>4}{'RMSE(log)':>10}{'bias':>8}{'MedAPE':>8}{'in-bands':>10}"
        f"{'skill':>8}{'  vs best':<20}"
    )
    rule(107)
    ordered = sorted(scored, key=lambda x: _rank_key(x["backtest"]))
    for position, r in enumerate(ordered):
        b = r["backtest"]
        if position == 0:
            verdict = "  best"
        elif np.isfinite(b.rmse_delta_low):
            span = f"[{b.rmse_delta_low:+.2f}, {b.rmse_delta_high:+.2f}]"
            verdict = f"  {span:<17}{'tie' if b.tied_with_best else ''}"
        else:
            verdict = ""
        print(
            f"{r['spec'].name:<37}{b.origins:>4}{b.rmse_log:>10.3f}{b.bias_log:>+8.3f}"
            f"{b.medape:>7.1f}%{b.coverage:>9.1f}%{b.skill:>+8.2f}{verdict}"
        )
    rule(107)
    print(
        "bias > 0 means the model under-predicted (price came in above the curve).\n"
        "'skill' compares squared log error against a random walk (last price persists):\n"
        "positive beats it, negative loses to it."
    )

    # Everything below compares models to each other, so it only means anything
    # when there is more than one.
    if len(ordered) < 2:
        return

    ties = sum(1 for r in ordered[1:] if r["backtest"].tied_with_best)
    print(
        f"\nTreat this ordering as indicative, not a verdict. {ties} of {len(ordered) - 1} "
        "models are statistically\nindistinguishable from the top one — their bootstrap "
        "interval for the RMSE gap straddles zero.\nThat interval is Bonferroni-widened across "
        "the comparisons, because every model is measured\nagainst whichever one happened to "
        "win on this same small sample.\n"
        f"\nForecast windows overlap and there are only {min(counts)} of them, so the ordering "
        "moves when the\nevaluation window moves: over the full model set, starting in 2013 "
        "rather than 2015 drops\nthe halving-cycle model out of the top five, while starting "
        "in 2017 puts it first."
    )

    aicc_best = fitted[0]
    if ordered[0] is not aicc_best:
        print(
            f"\nNote that the two rankings disagree: '{aicc_best['spec'].name}' wins on ΔAICc "
            f"while\n'{ordered[0]['spec'].name}' forecasts best here. Penalising complexity and "
            "measuring\nforecast accuracy are different questions, and on this data they have "
            "different answers."
        )


def render_index(
    raw_data, results, extend_months: int, n_eff: float, backtest_ran: bool = True
) -> str:
    """Build the landing page: ranked comparison table plus links."""
    n_hist = len(raw_data)
    price = float(raw_data["Value"].iloc[-1])
    as_of = raw_data["Date"].max().date()

    rows = []
    for result in results:
        spec, fit = result["spec"], result["fit"]
        name = html.escape(spec.name)
        formula = html.escape(spec.formula)
        description = html.escape(spec.description)

        if fit is None:
            rows.append(
                f"""      <tr class="failed">
        <td><span class="name">{name}</span><span class="formula">{formula}</span></td>
        <td colspan="7" class="error">fit failed — {html.escape(result["error"])}</td>
      </tr>"""
            )
            continue

        fair = float(fit.price()[n_hist - 1])
        ratio = price / fair
        band = current_band(fit, price, n_hist)
        color = band_color(band)
        m = result["metrics"]
        b = result["backtest"]

        if b:
            tie = " · tied w/ best" if b.tied_with_best else ""
            oos = (
                f'<span class="ratio">{b.rmse_log:.3f}</span>'
                f'<span class="sub">skill {b.skill:+.2f}{tie}</span>'
            )
        else:
            oos = '<span class="sub">not run</span>'
        rows.append(
            f"""      <tr>
        <td>
          <a class="name" href="{spec.key}.html">{name}</a>
          <span class="formula">{formula}</span>
          <span class="desc">{description}</span>
        </td>
        <td class="num">{m.n_params}</td>
        <td class="num"><span class="ratio">{m.d_aicc:.2f}</span>
          <span class="sub">ΔBIC {m.d_bic:.1f}</span></td>
        <td class="num"><span class="ratio">{fit.r2:.4f}</span>
          <span class="sub">adj {m.adj_r2:.4f}</span></td>
        <td class="num">{fit.sigma:.3f}</td>
        <td class="num">{oos}</td>
        <td class="num">${fair:,.0f}</td>
        <td class="num"><span class="ratio">{ratio:.2f}×</span>
          <span class="band"><i style="background:{color}"></i>{html.escape(band)}</span></td>
      </tr>"""
        )

    fitted = [r for r in results if r["fit"]]
    scored = [r for r in fitted if r["backtest"]]
    if scored:
        counts = {r["backtest"].origins for r in scored}
        span = f"{min(counts)}–{max(counts)}" if len(counts) > 1 else str(min(counts))
        horizon = scored[0]["backtest"].horizon_days
        oos_note = (
            f"Out-of-sample RMSE comes from a walk-forward backtest: {span} forecast "
            f"origins, each model refitted on data up to that date and scored {horizon} "
            "days later on history it had not seen. Lower is better; skill is measured "
            "against a random walk, where positive beats it."
        )
        if len(counts) > 1:
            oos_note += (
                " Origin counts differ because some models failed to converge on shorter "
                "histories; ranking and intervals use only the origins every model scored."
            )
    elif backtest_ran:
        oos_note = (
            "The walk-forward backtest produced no usable forecast origins — the horizon "
            "leaves no room between the backtest start and the end of the data — so the "
            "out-of-sample column is empty."
        )
    else:
        oos_note = "The walk-forward backtest was disabled for this run."

    # Only assert that the rankings disagree when there are two rankings and they
    # actually do.
    aicc_best = fitted[0] if fitted else None
    oos_best = min(scored, key=lambda r: _rank_key(r["backtest"])) if scored else None
    if oos_best is not None and aicc_best is not None and oos_best is not aicc_best:
        ranking_para = (
            "<p>The two rankings disagree, and neither is the last word. ΔAICc asks whether "
            "extra parameters pay for themselves; the backtest asks what actually forecast "
            "well. Models marked <em>tied w/ best</em> are statistically indistinguishable "
            "from the top forecaster — with only a couple of dozen overlapping forecast "
            "windows, most of that ordering is noise, and it shifts when the evaluation "
            "period shifts.</p>"
        )
    elif oos_best is not None:
        ranking_para = (
            "<p>Both rankings put the same model first here, but they answer different "
            "questions: ΔAICc asks whether extra parameters pay for themselves, the backtest "
            "asks what actually forecast well. Models marked <em>tied w/ best</em> are "
            "statistically indistinguishable from the top forecaster — with so few "
            "overlapping forecast windows, most of that ordering is noise.</p>"
        )
    else:
        ranking_para = ""

    return INDEX_TEMPLATE.format(
        as_of=as_of,
        price=f"{price:,.0f}",
        observations=f"{n_hist:,}",
        model_count=len(fitted),
        extend_months=extend_months,
        first_date=raw_data["Date"].min().date(),
        n_eff=f"{n_eff:.1f}",
        decorrelation_days=f"{n_hist / n_eff:.0f}",
        ranking_note=html.escape(RANKING_NOTE),
        ranking_para=ranking_para,
        oos_note=html.escape(oos_note),
        rows="\n".join(rows),
        css=report.CSS,
        stats_tables=report.stats_section(results, n_eff, n_hist),
    )


def band_color(band_label: str) -> str:
    for color, label in BAND_COLORS:
        if label == band_label:
            return color
    return "#9a9a9a"


INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bitcoin Rainbow Charts — Model Comparison</title>
<style>{css}</style>
</head>
<body>
<main>
  <h1>Bitcoin Rainbow Charts</h1>
  <p class="sub">One chart per regression model, ranked by ΔAICc — fit quality after paying for complexity.</p>

  <div class="stats">
    <div class="stat"><div class="k">Price as of {as_of}</div><div class="v">${price}</div></div>
    <div class="stat"><div class="k">Observations</div><div class="v">{observations}</div>
      <div class="note">effective: {n_eff}</div></div>
    <div class="stat"><div class="k">Models</div><div class="v">{model_count}</div></div>
    <div class="stat"><div class="k">Projection</div><div class="v">{extend_months} mo</div></div>
  </div>

  <div class="callout">
    <strong>Why not rank by R²?</strong> {ranking_note}
    The penalty is charged against an <em>effective</em> sample size of {n_eff}, not the
    {observations} daily rows: residual autocorrelation decays over a
    timescale of roughly {decorrelation_days} days, so those rows carry far less
    independent evidence than their count suggests. Scored naively, every extra parameter looks free.
  </div>

  <div class="wrap">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th class="num">k</th>
        <th class="num">ΔAICc</th>
        <th class="num">R² (log)</th>
        <th class="num">σ</th>
        <th class="num">Out-of-sample<br>RMSE (log)</th>
        <th class="num">Fair value</th>
        <th class="num">Price vs. model</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
  </div>

{stats_tables}

  <footer>
    <p>Fitted on daily closes from {first_date} to {as_of}. R², σ and the information
    criteria are measured on natural-log price, so they are comparable across models.</p>
    <p><strong>ΔAICc</strong> is the distance from the best model; 0 is best and a gap
    below about 2 means the two are effectively tied. <strong>k</strong> is the number of
    free parameters in the trend. Adjusted R² is shown next to R² because it is the usual
    suggested remedy, but note it barely moves — with n in the thousands and k ≤ 6 the
    correction lands in the fifth decimal, so it does not fix anything here.</p>
    <p>{oos_note}</p>
    {ranking_para}
    <p>A high R² means a model tracks history closely, not that it forecasts well — the
    most flexible curves overfit the past most eagerly. Nothing here is financial advice.</p>
  </footer>
</main>
</body>
</html>
"""


def parse_args(argv=None):
    env = os.environ.get
    parser = argparse.ArgumentParser(description="Build Bitcoin rainbow charts.")
    parser.add_argument("--data-file", default=env("DATA_FILE", "data/bitcoin_data.csv"))
    parser.add_argument("--output-dir", default=env("OUTPUT_DIR", "output"))
    parser.add_argument(
        "--models",
        default=env("MODELS", "all"),
        help="Comma-separated model keys, or 'all'. "
        f"Available: {', '.join(models.MODELS_BY_KEY)}",
    )
    parser.add_argument("--extend-months", type=int, default=int(env("EXTEND_MONTHS", "9")))
    parser.add_argument("--exchanges", default=env("EXCHANGES", ",".join(DEFAULT_EXCHANGES)))
    parser.add_argument(
        "--no-update",
        action="store_true",
        default=env("SKIP_UPDATE", "0").lower() in {"1", "true", "yes"},
        help="Skip the exchange top-up and plot the CSV as-is.",
    )
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        default=env("SKIP_BACKTEST", "0").lower() in {"1", "true", "yes"},
        help="Skip the walk-forward out-of-sample backtest (the slowest step).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=int(env("BACKTEST_HORIZON_DAYS", str(metrics_mod.BACKTEST_HORIZON_DAYS))),
        help="Forecast horizon for the walk-forward backtest.",
    )
    parser.add_argument("--list-models", action="store_true", help="Print models and exit.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_models:
        for spec in models.MODELS:
            print(f"{spec.key:<20} {spec.name}\n{'':<20} {spec.formula}")
        return 0

    try:
        results = build_charts(
            data_file=args.data_file,
            output_dir=args.output_dir,
            selection=args.models,
            extend_months=args.extend_months,
            exchanges=[e.strip() for e in args.exchanges.split(",") if e.strip()],
            update=not args.no_update,
            backtest=not args.no_backtest,
            horizon_days=args.horizon_days,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    failed = [r for r in results if r["fit"] is None]
    if failed:
        print(f"\n{len(failed)} model(s) failed to fit.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
