import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from industrial_ml_projects.projects import dropbox, instacart, nvidia, stripe, uber


PROJECTS = {
    "Uber": {
        "tagline": "Predict when a ride will arrive.",
        "problem": "Uber needs to estimate arrival time before the trip happens. The app has to look at distance, traffic, weather, airport trips, driver supply, and time of day, then give a prediction that feels trustworthy to the rider.",
        "button": "Generate Uber ride data and run the model",
    },
    "Stripe": {
        "tagline": "Find fraud rings that look separate at first.",
        "problem": "Stripe needs to catch groups of risky merchant accounts. One account may not look obvious by itself, but a group can share bank accounts, devices, IP ranges, email domains, and business patterns.",
        "button": "Generate Stripe account data and find groups",
    },
    "Instacart": {
        "tagline": "Predict if a grocery item will actually be found.",
        "problem": "Instacart needs to decide what items to show before a shopper reaches the store. If unavailable items are shown too often, customers lose trust. If too many items are hidden, customers lose choice.",
        "button": "Generate Instacart item data and run the model",
    },
    "NVIDIA": {
        "tagline": "Use network patterns to spot financial fraud.",
        "problem": "Financial fraud often appears through shared accounts, cards, and devices. A single transaction row can miss the bigger picture, so the model also looks at how transactions are connected.",
        "button": "Generate NVIDIA graph data and detect fraud",
    },
    "Dropbox": {
        "tagline": "Improve search results with better labels.",
        "problem": "Dropbox needs search results that bring the best documents to the top. Human labels are useful but expensive, so the workflow starts with a small trusted set and expands it with teacher style labels.",
        "button": "Generate Dropbox search data and train relevance model",
    },
}


def intro_markdown(company: str) -> str:
    item = PROJECTS[company]
    return f"## {company}: {item['tagline']}\n\n{item['problem']}"


def empty_chart() -> str:
    return """
    <div class="placeholder">
      <div class="placeholder-title">Click the button to create fresh synthetic data.</div>
      <div class="placeholder-copy">The chart will appear here after the model runs.</div>
    </div>
    """


def bar_svg(labels, values, title, y_label, color="var(--primary)") -> str:
    values = [float(v) for v in values]
    max_value = max(values) if values else 1.0
    width, height = 760, 360
    left, right, top, bottom = 70, 30, 46, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    gap = 14
    bar_w = max(18, (plot_w - gap * (len(values) - 1)) / max(1, len(values)))
    bars = []
    ticks = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * (bar_w + gap)
        h = plot_h * value / max_value if max_value else 0
        y = top + plot_h - h
        bars.append(
            f"""
            <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="{color}" opacity="0.82">
              <title>{label}: {value:.2f}</title>
            </rect>
            <text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.1f}</text>
            <text x="{x + bar_w / 2:.1f}" y="{height - 34}" text-anchor="middle" class="axis">{label}</text>
            """
        )
    for step in range(5):
        val = max_value * step / 4
        y = top + plot_h - plot_h * step / 4
        ticks.append(
            f"""
            <line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>
            <text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{val:.1f}</text>
            """
        )
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
      <style>
        .title {{ fill: var(--foreground); font: 500 18px system-ui; }}
        .axis, .value {{ fill: var(--muted-foreground); font: 400 13px system-ui; }}
        .grid {{ stroke: var(--border); stroke-width: 1; }}
      </style>
      <text x="{left}" y="26" class="title">{title}</text>
      <text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" class="axis">{y_label}</text>
      {''.join(ticks)}
      {''.join(bars)}
    </svg>
    """


def line_svg(labels, values, title, y_label) -> str:
    values = [float(v) for v in values]
    max_value = max(values) if values else 1.0
    min_value = min(values) if values else 0.0
    span = max(max_value - min_value, 1e-6)
    width, height = 760, 360
    left, right, top, bottom = 72, 34, 48, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = []
    for idx, value in enumerate(values):
        x = left + plot_w * idx / max(1, len(values) - 1)
        y = top + plot_h - plot_h * (value - min_value) / span
        points.append((x, y, value))
    path = " ".join(("M" if idx == 0 else "L") + f"{x:.1f},{y:.1f}" for idx, (x, y, _) in enumerate(points))
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="var(--primary)"><title>{labels[idx]}: {value:.2f}</title></circle>'
        for idx, (x, y, value) in enumerate(points)
    )
    label_marks = "".join(
        f'<text x="{x:.1f}" y="{height - 34}" text-anchor="middle" class="axis">{labels[idx]}</text>'
        for idx, (x, _, _) in enumerate(points)
    )
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
      <style>
        .title {{ fill: var(--foreground); font: 500 18px system-ui; }}
        .axis, .value {{ fill: var(--muted-foreground); font: 400 13px system-ui; }}
        .grid {{ stroke: var(--border); stroke-width: 1; }}
      </style>
      <text x="{left}" y="28" class="title">{title}</text>
      <text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" class="axis">{y_label}</text>
      <line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="grid"/>
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="grid"/>
      <path d="{path}" fill="none" stroke="var(--primary)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      {circles}
      {label_marks}
    </svg>
    """


def network_svg(title, clusters=5, risky=2) -> str:
    width, height = 760, 360
    centers = [(150, 130), (310, 110), (490, 145), (250, 250), (590, 245)]
    edges = []
    nodes = []
    for c_idx, (cx, cy) in enumerate(centers[:clusters]):
        points = []
        for idx in range(7):
            angle = 2 * np.pi * idx / 7
            radius = 42 + (idx % 2) * 10
            points.append((cx + np.cos(angle) * radius, cy + np.sin(angle) * radius))
        for idx, (x1, y1) in enumerate(points):
            x2, y2 = points[(idx + 2) % len(points)]
            edges.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="edge"/>')
        fill = "var(--primary)" if c_idx < risky else "var(--muted-foreground)"
        opacity = "0.86" if c_idx < risky else "0.35"
        for x, y in points:
            nodes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{fill}" opacity="{opacity}"/>')
        nodes.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="{fill}" opacity="{opacity}"/>')
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
      <style>
        .title {{ fill: var(--foreground); font: 500 18px system-ui; }}
        .caption {{ fill: var(--muted-foreground); font: 400 13px system-ui; }}
        .edge {{ stroke: var(--border); stroke-width: 2; }}
      </style>
      <text x="46" y="34" class="title">{title}</text>
      <text x="46" y="60" class="caption">Darker groups are the ones the model would send for review.</text>
      {''.join(edges)}
      {''.join(nodes)}
    </svg>
    """


def wrap_visual(svg: str) -> str:
    return f'<div class="visual-card">{svg}</div>'


def run_uber(rows: int):
    result = uber.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    df = uber.make_eta_data(rows)
    df["distance_group"] = pd.cut(df["trip_miles"], bins=[0, 3, 6, 10, 40], labels=["short", "medium", "long", "very long"])
    grouped = df.groupby("distance_group", observed=False)["eta_minutes"].mean()
    chart = bar_svg(grouped.index.tolist(), grouped.values.tolist(), "Longer rides usually mean later arrivals", "average minutes")
    outputs = f"""
### What happened

The app created **{rows:,} synthetic ride requests** and trained a model to predict arrival time.

### Model output

* Average prediction error is about **{result.metrics['mae']:.1f} minutes**.
* The strongest signal is **trip distance**, followed by traffic and airport trips.
* The serving check passed, so the saved model gives the same answer when the same ride is scored again.
"""
    return wrap_visual(chart), outputs


def run_stripe(rows: int):
    result = stripe.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    chart = network_svg("The model connects accounts that share suspicious patterns", clusters=5, risky=min(4, result.metrics["clusters_found"]))
    outputs = f"""
### What happened

The app created synthetic merchant accounts, compared account pairs, and connected accounts that looked related.

### Model output

* Found **{result.metrics['clusters_found']} suspicious groups**.
* Largest group has **{result.metrics['largest_cluster_size']} accounts**.
* The model was very confident on this synthetic run because the hidden fraud rings share clear signals.
"""
    return wrap_visual(chart), outputs


def run_instacart(rows: int):
    result = instacart.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    df = instacart.make_availability_data(rows)
    grouped = df.groupby("category")["found"].mean().head(7)
    labels = [f"cat {int(x)}" for x in grouped.index.tolist()]
    chart = bar_svg(labels, (grouped.values * 100).tolist(), "Some item groups are easier to find than others", "found rate %")
    outputs = f"""
### What happened

The app created **{rows:,} synthetic grocery item requests** and trained a model to predict whether each item will be found.

### Model output

* The model chose to show about **{result.metrics['selection_rate'] * 100:.0f}%** of candidate items.
* Among shown items, about **{result.metrics['found_rate_when_displayed'] * 100:.0f}%** were actually found.
* The demo also creates adjustable decision thresholds, so business rules can change without retraining.
"""
    return wrap_visual(chart), outputs


def run_nvidia(rows: int):
    result = nvidia.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    chart = network_svg("Shared accounts, cards, and devices reveal risk patterns", clusters=5, risky=3)
    outputs = f"""
### What happened

The app created a transaction network and trained a fraud model using both transaction behavior and network connections.

### Model output

* The synthetic graph has **{result.metrics['nodes']:,} nodes** and **{result.metrics['edges']:,} links**.
* The model reached **{result.metrics['roc_auc']:.2f} ROC AUC** on this run.
* Network features helped the model notice risk that is hard to see from a single transaction alone.
"""
    return wrap_visual(chart), outputs


def run_dropbox(rows: int):
    result = dropbox.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    df = dropbox.make_docs_and_queries(max(250, rows // 20))
    grouped = df.groupby("human_label").size().reindex([1, 2, 3, 4, 5], fill_value=0)
    chart = line_svg([str(x) for x in grouped.index], grouped.values.tolist(), "Search examples spread across relevance levels", "examples")
    outputs = f"""
### What happened

The app created synthetic search queries, documents, small human style labels, and larger teacher labels.

### Model output

* Created **{result.metrics['human_seed_rows']:,} trusted seed labels**.
* Expanded to **{result.metrics['amplified_teacher_rows']:,} training examples**.
* The ranking score reached **{result.metrics['ranker_ndcg_at_10']:.2f} NDCG at 10**, which means useful results moved near the top.
"""
    return wrap_visual(chart), outputs


RUNNERS = {
    "Uber": run_uber,
    "Stripe": run_stripe,
    "Instacart": run_instacart,
    "NVIDIA": run_nvidia,
    "Dropbox": run_dropbox,
}


def update_project(company: str):
    return intro_markdown(company), gr.update(value=PROJECTS[company]["button"])


def generate(company: str, rows: int):
    chart, outputs = RUNNERS[company](int(rows))
    return intro_markdown(company), chart, outputs


css = """
.gradio-container { max-width: 1120px !important; margin: 0 auto; }
.hero { padding: 12px 0 2px; }
.hero h1 { margin-bottom: 8px; }
.visual-card {
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  padding: 14px;
  background: var(--background-fill-secondary);
}
.placeholder {
  border: 1px dashed var(--border-color-primary);
  border-radius: 8px;
  padding: 44px 20px;
  text-align: center;
}
.placeholder-title { font-weight: 600; margin-bottom: 8px; }
.placeholder-copy { color: var(--body-text-color-subdued); }
"""


with gr.Blocks(title="Usecase ML", css=css) as demo:
    gr.Markdown(
        """
<div class="hero">

# Usecase ML

Choose a company inspired project, generate synthetic data, and watch the model output come to life.

</div>
"""
    )
    with gr.Row():
        company = gr.Dropdown(choices=list(PROJECTS), value="Uber", label="Choose a project")
        rows = gr.Slider(500, 3000, value=1000, step=250, label="Synthetic examples to generate")
    intro = gr.Markdown(intro_markdown("Uber"))
    run_button = gr.Button(PROJECTS["Uber"]["button"], variant="primary")
    chart = gr.HTML(empty_chart())
    outputs = gr.Markdown("The model output will appear here after you click the button.")

    company.change(update_project, inputs=company, outputs=[intro, run_button])
    run_button.click(generate, inputs=[company, rows], outputs=[intro, chart, outputs])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
