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
        "workflow": ["Ride request", "Traffic signals", "ETA model", "Arrival estimate"],
    },
    "Stripe": {
        "tagline": "Find fraud rings that look separate at first.",
        "problem": "Stripe needs to catch groups of risky merchant accounts. One account may not look obvious by itself, but a group can share bank accounts, devices, IP ranges, email domains, and business patterns.",
        "button": "Generate Stripe account data and find groups",
        "workflow": ["Merchant accounts", "Shared signals", "Similarity model", "Fraud groups"],
    },
    "Instacart": {
        "tagline": "Predict if a grocery item will actually be found.",
        "problem": "Instacart needs to decide what items to show before a shopper reaches the store. If unavailable items are shown too often, customers lose trust. If too many items are hidden, customers lose choice.",
        "button": "Generate Instacart item data and run the model",
        "workflow": ["Customer basket", "Store signals", "Availability model", "Show or hide item"],
    },
    "NVIDIA": {
        "tagline": "Use network patterns to spot financial fraud.",
        "problem": "Financial fraud often appears through shared accounts, cards, and devices. A single transaction row can miss the bigger picture, so the model also looks at how transactions are connected.",
        "button": "Generate NVIDIA graph data and detect fraud",
        "workflow": ["Transactions", "Account graph", "Risk model", "Fraud review"],
    },
    "Dropbox": {
        "tagline": "Improve search results with better labels.",
        "problem": "Dropbox needs search results that bring the best documents to the top. Human labels are useful but expensive, so the workflow starts with a small trusted set and expands it with teacher style labels.",
        "button": "Generate Dropbox search data and train relevance model",
        "workflow": ["Search query", "Human labels", "Teacher labels", "Better ranking"],
    },
}


def intro_markdown(company: str) -> str:
    item = PROJECTS[company]
    return f"## {company}: {item['tagline']}\n\n{item['problem']}"


def empty_chart() -> str:
    return """
    <div class="demo-layout">
      <div class="placeholder">
        <div class="placeholder-title">Click the button to create fresh synthetic data.</div>
        <div class="placeholder-copy">A chart, workflow diagram, and model results will appear here.</div>
      </div>
    </div>
    """


def bar_svg(labels, values, title, y_label, color="#2563eb") -> str:
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
        .title {{ fill: #172033; font: 700 18px system-ui; }}
        .axis, .value {{ fill: #526070; font: 500 13px system-ui; }}
        .grid {{ stroke: #dbe4f0; stroke-width: 1; }}
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
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"><title>{labels[idx]}: {value:.2f}</title></circle>'
        for idx, (x, y, value) in enumerate(points)
    )
    label_marks = "".join(
        f'<text x="{x:.1f}" y="{height - 34}" text-anchor="middle" class="axis">{labels[idx]}</text>'
        for idx, (x, _, _) in enumerate(points)
    )
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
      <style>
        .title {{ fill: #172033; font: 700 18px system-ui; }}
        .axis, .value {{ fill: #526070; font: 500 13px system-ui; }}
        .grid {{ stroke: #dbe4f0; stroke-width: 1; }}
      </style>
      <text x="{left}" y="28" class="title">{title}</text>
      <text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" class="axis">{y_label}</text>
      <line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" class="grid"/>
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="grid"/>
      <path d="{path}" fill="none" stroke="#2563eb" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
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
        fill = "#2563eb" if c_idx < risky else "#94a3b8"
        opacity = "0.88" if c_idx < risky else "0.46"
        for x, y in points:
            nodes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{fill}" opacity="{opacity}"/>')
        nodes.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="{fill}" opacity="{opacity}"/>')
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
      <style>
        .title {{ fill: #172033; font: 700 18px system-ui; }}
        .caption {{ fill: #526070; font: 500 13px system-ui; }}
        .edge {{ stroke: #cbd7e6; stroke-width: 2; }}
      </style>
      <text x="46" y="34" class="title">{title}</text>
      <text x="46" y="60" class="caption">Darker groups are the ones the model would send for review.</text>
      {''.join(edges)}
      {''.join(nodes)}
    </svg>
    """


def workflow_3d(company: str) -> str:
    steps = PROJECTS[company]["workflow"]
    cards = []
    for idx, step in enumerate(steps, start=1):
        cards.append(
            f"""
            <div class="workflow-step">
              <div class="workflow-number">{idx}</div>
              <div class="workflow-text">{step}</div>
            </div>
            """
        )
    return f"""
    <div class="workflow-card">
      <div class="section-label">3D workflow</div>
      <div class="workflow-scene">
        {''.join(cards)}
      </div>
    </div>
    """


def result_card(title: str, body: str, metrics) -> str:
    metric_cards = "".join(
        f"""
        <div class="metric-card">
          <div class="metric-value">{value}</div>
          <div class="metric-label">{label}</div>
        </div>
        """
        for label, value in metrics
    )
    return f"""
    <div class="result-card">
      <div>
        <div class="section-label">Model output</div>
        <h3>{title}</h3>
        <p>{body}</p>
      </div>
      <div class="metric-grid">{metric_cards}</div>
    </div>
    """


def wrap_visual(company: str, svg: str) -> str:
    return f"""
    <div class="demo-layout">
      <div class="visual-card">
        <div class="section-label">Model visual</div>
        {svg}
      </div>
      {workflow_3d(company)}
    </div>
    """


def run_uber(rows: int):
    result = uber.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    df = uber.make_eta_data(rows)
    df["distance_group"] = pd.cut(df["trip_miles"], bins=[0, 3, 6, 10, 40], labels=["short", "medium", "long", "very long"])
    grouped = df.groupby("distance_group", observed=False)["eta_minutes"].mean()
    chart = bar_svg(grouped.index.tolist(), grouped.values.tolist(), "Longer rides usually mean later arrivals", "average minutes")
    outputs = result_card(
        "Uber ride ETA model",
        "The app created synthetic ride requests, trained an arrival time model, and checked that repeated scoring gives the same answer. The model mainly learns that distance, traffic, and airport trips push arrival time higher.",
        [
            ("synthetic rides", f"{rows:,}"),
            ("average error", f"{result.metrics['mae']:.1f} min"),
            ("serving check", "passed"),
        ],
    )
    return wrap_visual("Uber", chart), outputs


def run_stripe(rows: int):
    result = stripe.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    chart = network_svg("The model connects accounts that share suspicious patterns", clusters=5, risky=min(4, result.metrics["clusters_found"]))
    outputs = result_card(
        "Stripe fraud ring model",
        "The app created merchant accounts, compared account pairs, and connected accounts that shared suspicious signals. The final groups are the ones an analyst would review first.",
        [
            ("groups found", f"{result.metrics['clusters_found']}"),
            ("largest group", f"{result.metrics['largest_cluster_size']} accounts"),
            ("review path", "ready"),
        ],
    )
    return wrap_visual("Stripe", chart), outputs


def run_instacart(rows: int):
    result = instacart.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    df = instacart.make_availability_data(rows)
    grouped = df.groupby("category")["found"].mean().head(7)
    labels = [f"cat {int(x)}" for x in grouped.index.tolist()]
    chart = bar_svg(labels, (grouped.values * 100).tolist(), "Some item groups are easier to find than others", "found rate %")
    outputs = result_card(
        "Instacart availability model",
        "The app created grocery item requests, predicted which items shoppers are likely to find, and used a threshold to decide what should be shown to customers.",
        [
            ("items shown", f"{result.metrics['selection_rate'] * 100:.0f}%"),
            ("found after shown", f"{result.metrics['found_rate_when_displayed'] * 100:.0f}%"),
            ("decision rules", "active"),
        ],
    )
    return wrap_visual("Instacart", chart), outputs


def run_nvidia(rows: int):
    result = nvidia.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    chart = network_svg("Shared accounts, cards, and devices reveal risk patterns", clusters=5, risky=3)
    outputs = result_card(
        "NVIDIA graph fraud model",
        "The app created a transaction network, turned shared cards and devices into graph signals, and used those signals to score fraud risk.",
        [
            ("network points", f"{result.metrics['nodes']:,}"),
            ("network links", f"{result.metrics['edges']:,}"),
            ("fraud score", f"{result.metrics['roc_auc']:.2f} AUC"),
        ],
    )
    return wrap_visual("NVIDIA", chart), outputs


def run_dropbox(rows: int):
    result = dropbox.run(ROOT / "artifacts" / "live_demo", n_rows=rows)
    df = dropbox.make_docs_and_queries(max(250, rows // 20))
    grouped = df.groupby("human_label").size().reindex([1, 2, 3, 4, 5], fill_value=0)
    chart = line_svg([str(x) for x in grouped.index], grouped.values.tolist(), "Search examples spread across relevance levels", "examples")
    outputs = result_card(
        "Dropbox search relevance model",
        "The app created search examples, started with trusted labels, expanded the training set with teacher labels, and trained a ranker that pushes useful documents upward.",
        [
            ("trusted labels", f"{result.metrics['human_seed_rows']:,}"),
            ("training examples", f"{result.metrics['amplified_teacher_rows']:,}"),
            ("ranking quality", f"{result.metrics['ranker_ndcg_at_10']:.2f}"),
        ],
    )
    return wrap_visual("Dropbox", chart), outputs


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
html, body, #root, gradio-app, .gradio-container, .main, .app, footer {
  background: #eef6ff !important;
}
body {
  margin: 0 !important;
}
.gradio-container,
.contain,
.app,
main {
  width: 100% !important;
}
.gradio-container {
  max-width: none !important;
  min-height: 100vh;
  padding: 0 !important;
  background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
  color: #172033;
}
.main {
  max-width: none !important;
}
.block {
  border-color: #d9e5f3 !important;
}
.prose, .prose p, .prose li, .prose h1, .prose h2, .prose h3 {
  color: #172033 !important;
}
.wrap {
  max-width: 100% !important;
}
.app-shell {
  width: 100%;
  max-width: none;
  margin: 0 auto;
  padding: 24px clamp(18px, 3vw, 42px) 42px;
  box-sizing: border-box;
}
.hero {
  padding: 18px 4px 8px;
  color: #172033;
}
.hero h1 { margin-bottom: 8px; color: #172033; }
.hero p { color: #526070; }
.demo-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.55fr) minmax(340px, 0.85fr);
  gap: 24px;
  align-items: stretch;
  width: 100%;
}
.visual-card {
  border: 1px solid #d9e5f3;
  border-radius: 14px;
  padding: 18px;
  background: #ffffff;
  box-shadow: 0 18px 45px rgba(37, 99, 235, 0.08);
  overflow: hidden;
}
.visual-card svg {
  width: 100%;
  height: auto;
  display: block;
}
.section-label {
  color: #2563eb;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.workflow-card {
  border: 1px solid #d9e5f3;
  border-radius: 14px;
  padding: 18px;
  background: linear-gradient(145deg, #ffffff 0%, #e9f5ff 100%);
  box-shadow: 0 18px 45px rgba(14, 165, 233, 0.10);
  min-height: 100%;
  color: #172033 !important;
}
.workflow-scene {
  min-height: 300px;
  display: grid;
  gap: 16px;
  perspective: 900px;
  align-content: center;
}
.workflow-step {
  display: grid;
  grid-template-columns: 42px 1fr;
  gap: 12px;
  align-items: center;
  padding: 14px 14px;
  border-radius: 14px;
  color: #172033 !important;
  background: linear-gradient(135deg, #fefefe 0%, #dbeafe 100%);
  border: 1px solid #9fc0ee;
  box-shadow: 12px 14px 0 rgba(37, 99, 235, 0.18), 0 18px 28px rgba(15, 23, 42, 0.12);
  transform: rotateY(-10deg) rotateX(4deg);
}
.workflow-step:nth-child(even) {
  transform: rotateY(9deg) rotateX(4deg) translateX(10px);
  background: linear-gradient(135deg, #fefefe 0%, #ccfbf1 100%);
  border-color: #91d8cd;
  box-shadow: -12px 14px 0 rgba(20, 184, 166, 0.18), 0 18px 28px rgba(15, 23, 42, 0.12);
}
.workflow-number {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  background: #2563eb;
  color: #ffffff;
  font-weight: 800;
}
.workflow-text {
  font-weight: 700;
  color: #172033 !important;
  text-shadow: none;
}
.workflow-card *,
.workflow-step *,
.workflow-text {
  color: #172033 !important;
}
.workflow-number,
.workflow-number * {
  color: #ffffff !important;
}
.result-card {
  margin-top: 18px;
  border: 1px solid #d9e5f3;
  border-radius: 14px;
  padding: 20px;
  background: #ffffff;
  color: #172033;
  box-shadow: 0 18px 45px rgba(37, 99, 235, 0.08);
}
.result-card h3 {
  margin: 4px 0 8px;
  color: #172033;
}
.result-card p {
  margin: 0 0 18px;
  color: #526070;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.metric-card {
  border: 1px solid #e0e9f5;
  border-radius: 12px;
  padding: 14px;
  background: #f8fbff;
}
.metric-value {
  font-size: 1.3rem;
  line-height: 1.2;
  font-weight: 800;
  color: #2563eb;
  overflow-wrap: anywhere;
}
.metric-label {
  margin-top: 4px;
  color: #526070;
  font-size: 0.92rem;
}
.placeholder {
  grid-column: 1 / -1;
  border: 1px dashed #9bb7d7;
  border-radius: 14px;
  padding: 56px 20px;
  text-align: center;
  background: #ffffff;
  color: #172033;
}
.placeholder-title { font-weight: 600; margin-bottom: 8px; }
.placeholder-copy { color: #526070; }
button.primary, .primary button {
  background: linear-gradient(90deg, #2563eb 0%, #14b8a6 100%) !important;
  border: 0 !important;
}
@media (max-width: 820px) {
  .demo-layout { grid-template-columns: 1fr; }
  .metric-grid { grid-template-columns: 1fr; }
  .workflow-scene { min-height: auto; }
  .workflow-step, .workflow-step:nth-child(even) { transform: none; }
}
"""


with gr.Blocks(title="Usecase ML", css=css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown(
        """
<div class="hero">

# Usecase ML

Choose a company inspired project, generate synthetic data, and watch the model output come to life.

</div>
"""
    )
    with gr.Column(elem_classes=["app-shell"]):
        with gr.Row():
            company = gr.Dropdown(choices=list(PROJECTS), value="Uber", label="Choose a project")
            rows = gr.Slider(500, 3000, value=1000, step=250, label="Synthetic examples to generate")
        intro = gr.Markdown(intro_markdown("Uber"))
        run_button = gr.Button(PROJECTS["Uber"]["button"], variant="primary")
        chart = gr.HTML(empty_chart())
        outputs = gr.HTML(
            """
            <div class="result-card">
              <div class="section-label">Model output</div>
              <h3>Ready when you are</h3>
              <p>Pick a company and click the button. The result summary will stay inside this box.</p>
            </div>
            """
        )

        company.change(update_project, inputs=company, outputs=[intro, run_button])
        run_button.click(generate, inputs=[company, rows], outputs=[intro, chart, outputs])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
