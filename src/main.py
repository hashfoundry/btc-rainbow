"""Build a Bitcoin rainbow chart for every regression model.

Run directly, or via ``docker compose up`` - the charts land in ``OUTPUT_DIR``
with an ``index.html`` that links and ranks them.
"""

import argparse
import html
import os
import sys
from pathlib import Path

import models
import plot
from data import DEFAULT_EXCHANGES, get_data
from models import build_input, fit_model, resolve_models
from plot import BAND_COLORS, current_band, extend_dates

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
LEGACY_CHART_MODEL = "log_time"
LEGACY_CHART_NAME = "bitcoin_rainbow_chart.html"


def build_charts(
    data_file: str,
    output_dir: str,
    selection: str = "all",
    extend_months: int = plot.EXTEND_MONTHS,
    exchanges=DEFAULT_EXCHANGES,
    update: bool = True,
):
    """Fit every selected model and write one chart each, plus an index.

    Returns:
        A list of per-model result dicts, ranked best-fit first.
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
            results.append({"spec": spec, "fit": None, "error": f"{type(exc).__name__}: {exc}"})
            continue

        fig = plot.create_plot(raw_data, fit, extended, months=extend_months)
        target = out / f"{spec.key}.html"
        fig.write_html(target, full_html=True, include_plotlyjs="cdn", config=PLOTLY_CONFIG)

        if spec.key == LEGACY_CHART_MODEL:
            fig.write_html(
                out / LEGACY_CHART_NAME,
                full_html=True,
                include_plotlyjs="cdn",
                config=PLOTLY_CONFIG,
            )

        print(f"  ✓ {spec.name:<40} R² = {fit.r2:>7.4f}   σ = {fit.sigma:.3f}   → {target.name}")
        results.append({"spec": spec, "fit": fit, "error": None})

    results.sort(key=lambda r: (r["fit"] is None, -(r["fit"].r2 if r["fit"] else 0)))
    index_path = out / "index.html"
    index_path.write_text(render_index(raw_data, results, extend_months), encoding="utf-8")
    print(f"\nWrote {index_path}")
    return results


def render_index(raw_data, results, extend_months: int) -> str:
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
        <td colspan="4" class="error">fit failed — {html.escape(result["error"])}</td>
      </tr>"""
            )
            continue

        fair = float(fit.price()[n_hist - 1])
        ratio = price / fair
        band = current_band(fit, price, n_hist)
        color = band_color(band)
        rows.append(
            f"""      <tr>
        <td>
          <a class="name" href="{spec.key}.html">{name}</a>
          <span class="formula">{formula}</span>
          <span class="desc">{description}</span>
        </td>
        <td class="num">{fit.r2:.4f}</td>
        <td class="num">{fit.sigma:.3f}</td>
        <td class="num">${fair:,.0f}</td>
        <td class="num"><span class="ratio">{ratio:.2f}×</span>
          <span class="band"><i style="background:{color}"></i>{html.escape(band)}</span></td>
      </tr>"""
        )

    fitted = [r for r in results if r["fit"]]
    return INDEX_TEMPLATE.format(
        as_of=as_of,
        price=f"{price:,.0f}",
        observations=f"{n_hist:,}",
        model_count=len(fitted),
        extend_months=extend_months,
        first_date=raw_data["Date"].min().date(),
        rows="\n".join(rows),
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
<style>
  :root {{
    --surface: #ffffff; --raised: #f7f7f8; --ink: #16181d; --muted: #6b7280;
    --line: #e5e7eb; --accent: #b45309;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --surface: #101216; --raised: #181b21; --ink: #eceef2; --muted: #9aa1ad;
      --line: #2a2e37; --accent: #f0b429;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 40px 24px 80px; background: var(--surface); color: var(--ink);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  main {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size: 30px; margin: 0 0 6px; letter-spacing: -0.02em; }}
  .sub {{ color: var(--muted); margin: 0 0 28px; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 32px; }}
  .stat {{
    flex: 1 1 160px; background: var(--raised); border: 1px solid var(--line);
    border-radius: 10px; padding: 14px 16px;
  }}
  .stat .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
  .stat .v {{ font-size: 22px; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }}
  .wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }}
  table {{ border-collapse: collapse; width: 100%; min-width: 760px; }}
  th, td {{ padding: 14px 16px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }}
  th {{
    background: var(--raised); color: var(--muted); font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .06em; position: sticky; top: 0;
  }}
  tr:last-child td {{ border-bottom: none; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
  a.name {{ color: var(--ink); font-weight: 650; text-decoration: none; border-bottom: 2px solid var(--accent); }}
  a.name:hover {{ color: var(--accent); }}
  .name {{ display: inline-block; }}
  .formula {{ display: block; color: var(--muted); font-size: 12.5px; margin-top: 5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .desc {{ display: block; color: var(--muted); font-size: 13px; margin-top: 6px; max-width: 62ch; }}
  .ratio {{ display: block; font-weight: 600; }}
  .band {{ display: flex; align-items: center; gap: 6px; justify-content: flex-end;
    color: var(--muted); font-size: 12.5px; margin-top: 5px; }}
  .band i {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .failed td {{ color: var(--muted); }}
  .error {{ font-size: 13px; }}
  footer {{ color: var(--muted); font-size: 13px; margin-top: 28px; max-width: 76ch; }}
</style>
</head>
<body>
<main>
  <h1>Bitcoin Rainbow Charts</h1>
  <p class="sub">One chart per regression model, ranked by goodness of fit in log-price space.</p>

  <div class="stats">
    <div class="stat"><div class="k">Price as of {as_of}</div><div class="v">${price}</div></div>
    <div class="stat"><div class="k">Observations</div><div class="v">{observations}</div></div>
    <div class="stat"><div class="k">Models</div><div class="v">{model_count}</div></div>
    <div class="stat"><div class="k">Projection</div><div class="v">{extend_months} mo</div></div>
  </div>

  <div class="wrap">
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th class="num">R² (log)</th>
        <th class="num">σ</th>
        <th class="num">Fair value</th>
        <th class="num">Price vs. model</th>
      </tr>
    </thead>
    <tbody>
{rows}
    </tbody>
  </table>
  </div>

  <footer>
    Fitted on daily closes from {first_date} to {as_of}. R² and σ are measured on
    natural-log price, so they are comparable across models; a higher R² means the
    model tracks history more closely, not that it forecasts better — the most
    flexible curves overfit the past most eagerly. Nothing here is financial advice.
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
