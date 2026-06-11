"""Streamlit UI for the X narrative pipeline.

Run with:  streamlit run render.py
Reads from cache/final_bundle.pkl produced by analyze.py.

Design language matches the mockup:
- Clean off-white surface with subtle warm tint (#FAF7F2) and amber accents
- Serif (Charter / Iowan / Georgia) for headlines and the bottom-line callout
- Sans (Inter / system) for everything else
- Stance colors: support = green ramp, mixed = warm gray, oppose = red ramp
- Outlier card accents: amber for engagement, pink for sentiment, amber for spike
- KG: node color = entity type, edge color = stance (critical/supportive/neutral)
"""
import pickle
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import CACHE_DIR

st.set_page_config(page_title="xnarrative", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# Design tokens — single source of truth
# ============================================================
PALETTE = {
    "page_bg":        "#FAF7F2",  # warm off-white
    "surface":        "#FFFFFF",
    "surface_alt":    "#F1EFE8",  # gray-50 from the design system
    "border":         "rgba(60, 52, 40, 0.12)",
    "border_strong":  "rgba(60, 52, 40, 0.22)",
    "text":           "#2C2C2A",  # gray-900
    "text_secondary": "#5F5E5A",  # gray-700
    "text_tertiary":  "#888780",  # gray-500
    "amber_fill":     "#FAEEDA",  # amber-50
    "amber_strong":   "#BA7517",  # amber-600
    "amber_text":     "#412402",  # amber-900
    "amber_warn_bg":  "#FAEEDA",  # warning surface
    # Stance ramp
    "support":        "#C0DD97",  # green-100
    "support_text":   "#173404",  # green-900
    "mixed":          "#D3D1C7",  # gray-100
    "mixed_text":     "#2C2C2A",
    "oppose":         "#F09595",  # red-200
    "oppose_text":    "#501313",  # red-900
    # Outlier accents
    "engagement":     "#BA7517",  # amber-600
    "sentiment":      "#D4537E",  # pink-400
    "spike":          "#BA7517",
    # KG entity types
    "person":         {"fill": "#EEEDFE", "border": "#534AB7", "text": "#26215C", "subtext": "#3C3489"},
    "country":        {"fill": "#FAECE7", "border": "#993C1D", "text": "#4A1B0C", "subtext": "#712B13"},
    "org":            {"fill": "#E6F1FB", "border": "#185FA5", "text": "#042C53", "subtext": "#0C447C"},
    "event":          {"fill": "#FAEEDA", "border": "#854F0B", "text": "#412402", "subtext": "#633806"},
    "place":          {"fill": "#EAF3DE", "border": "#3B6D11", "text": "#173404", "subtext": "#27500A"},
    "concept":        {"fill": "#FCEBEB", "border": "#A32D2D", "text": "#501313", "subtext": "#791F1F"},
    "movement":       {"fill": "#FBEAF0", "border": "#993556", "text": "#4B1528", "subtext": "#72243E"},
    # Edge stance
    "edge_critical":   "#A32D2D",
    "edge_supportive": "#3B6D11",
    "edge_neutral":    "#888780",
}

CSS = f"""
<style>
    /* ---- Page chrome ---- */
    .stApp {{ background: {PALETTE['page_bg']}; }}
    .main .block-container {{
        padding-top: 1.5rem; padding-bottom: 3rem;
        max-width: 1280px;
    }}
    /* Hide the default streamlit header bar */
    [data-testid="stHeader"] {{ background: transparent; }}

    /* ---- Typography ---- */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif;
        color: {PALETTE['text']};
    }}
    .serif {{
        font-family: "Charter", "Iowan Old Style", "Source Serif Pro", Georgia, serif;
        letter-spacing: -0.01em;
    }}
    h1, h2, h3, h4 {{
        font-family: "Charter", "Iowan Old Style", Georgia, serif !important;
        letter-spacing: -0.01em;
        font-weight: 500 !important;
        color: {PALETTE['text']};
    }}
    h1 {{ font-size: 1.75rem !important; margin: 0 !important; }}
    h2 {{ font-size: 1.1rem !important; margin: 0 0 0.6rem !important; }}
    h3 {{ font-size: 0.95rem !important; margin: 0 0 0.5rem !important; color: {PALETTE['text_secondary']} !important; }}

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 0.5px solid {PALETTE['border']};
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: auto;
        padding: 8px 14px;
        background: transparent !important;
        color: {PALETTE['text_secondary']};
        font-size: 13px;
        border-radius: 0;
    }}
    .stTabs [aria-selected="true"] {{
        color: {PALETTE['text']} !important;
        font-weight: 500;
        border-bottom: 2px solid {PALETTE['text']};
    }}

    /* ---- Tiny meta line at top ---- */
    .meta-line {{
        font-size: 12px; color: {PALETTE['text_secondary']};
        margin: 4px 0 1rem;
        padding-bottom: 1rem;
        border-bottom: 0.5px solid {PALETTE['border']};
    }}
    .meta-line .v {{ color: {PALETTE['text']}; }}
    .run-ts {{
        font-size: 11px; color: {PALETTE['text_tertiary']};
        letter-spacing: 0.05em; float: right;
    }}

    /* ---- Stat row ---- */
    .stat-block {{
        background: {PALETTE['surface_alt']};
        border-radius: 8px;
        padding: 12px 14px;
    }}
    .stat-num {{ font-size: 22px; font-weight: 500; line-height: 1.1; }}
    .stat-label {{
        font-size: 10.5px; color: {PALETTE['text_secondary']};
        text-transform: uppercase; letter-spacing: 0.05em;
        margin-top: 4px;
    }}

    /* ---- Bottom line callout ---- */
    .bottom-line {{
        background: {PALETTE['amber_warn_bg']};
        border-left: 3px solid {PALETTE['amber_strong']};
        padding: 14px 18px;
        margin: 0 0 1.25rem;
        border-radius: 0 8px 8px 0;
    }}
    .bottom-line-label {{
        font-size: 11px; font-weight: 500;
        color: {PALETTE['amber_text']};
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }}
    .bottom-line-text {{
        font-family: "Charter", "Iowan Old Style", Georgia, serif;
        font-size: 15px; line-height: 1.5;
        color: {PALETTE['amber_text']};
    }}

    /* ---- Tweet cards ---- */
    .tweet-card {{
        background: {PALETTE['surface']};
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
        font-size: 12.5px;
        line-height: 1.55;
        color: {PALETTE['text']};
    }}
    .tweet-card.engagement {{ border-left: 3px solid {PALETTE['engagement']}; }}
    .tweet-card.sentiment   {{ border-left: 3px solid {PALETTE['sentiment']}; }}
    .tweet-card.spike       {{ border-left: 3px solid {PALETTE['spike']}; }}
    .tweet-meta {{
        font-size: 10.5px; color: {PALETTE['text_secondary']};
        margin-top: 6px;
    }}
    .tweet-meta a {{ color: #185FA5; text-decoration: none; }}
    .tweet-meta a:hover {{ text-decoration: underline; }}
    .tweet-meta .score-amber {{ color: {PALETTE['amber_strong']}; }}
    .tweet-meta .score-pink  {{ color: {PALETTE['sentiment']}; }}

    /* ---- Narrative prose ---- */
    .narrative-prose {{
        font-size: 13.5px; line-height: 1.65;
        color: {PALETTE['text']};
    }}
    .narrative-prose p {{ margin: 0 0 12px; }}

    /* ---- Spike card ---- */
    .spike-card {{
        background: {PALETTE['surface']};
        border-left: 3px solid {PALETTE['amber_strong']};
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }}
    .spike-card-header {{
        display: flex; align-items: baseline; justify-content: space-between;
        margin-bottom: 6px;
    }}
    .spike-card-ts {{ font-weight: 500; font-size: 13px; }}
    .spike-card-meta {{ font-size: 11px; color: {PALETTE['text_secondary']}; }}
    .spike-card-body {{ font-size: 13px; line-height: 1.55; }}

    /* ---- Generic surface card (cadence, tables) ---- */
    .surface-card {{
        background: {PALETTE['surface']};
        border: 0.5px solid {PALETTE['border']};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 13px;
    }}
    .surface-card-title {{ font-weight: 500; margin-bottom: 8px; }}

    /* ---- Sentiment bar ---- */
    .sentiment-row {{ margin-bottom: 16px; }}
    .sentiment-row-label {{
        font-size: 12px; margin-bottom: 4px;
        display: flex; justify-content: space-between;
    }}
    .sentiment-row-n {{ color: {PALETTE['text_tertiary']}; font-size: 11px; }}
    .sentiment-bar {{
        display: flex; height: 18px; border-radius: 3px; overflow: hidden;
        font-size: 10px; font-weight: 500;
    }}
    .sentiment-bar > div {{
        display: flex; align-items: center; justify-content: center;
    }}
    .sentiment-bar .seg-support {{ background: {PALETTE['support']}; color: {PALETTE['support_text']}; }}
    .sentiment-bar .seg-mixed   {{ background: {PALETTE['mixed']};   color: {PALETTE['mixed_text']}; }}
    .sentiment-bar .seg-oppose  {{ background: {PALETTE['oppose']};  color: {PALETTE['oppose_text']}; }}
    .sentiment-legend {{
        display: flex; gap: 12px; font-size: 10px;
        color: {PALETTE['text_secondary']}; margin: -4px 0 18px;
    }}
    .sentiment-legend span {{ display: flex; align-items: center; gap: 4px; }}
    .swatch {{ width: 9px; height: 9px; border-radius: 2px; display: inline-block; }}

    /* ---- Theme bars ---- */
    .theme-row {{ margin-bottom: 5px; }}
    .theme-bar {{
        height: 16px; border-radius: 2px;
        display: flex; align-items: center;
        padding-left: 7px;
        font-size: 10.5px; font-weight: 500;
        color: {PALETTE['amber_text']};
    }}

    /* ---- Dividers between sections ---- */
    .section-rule {{
        border: none;
        border-top: 0.5px solid {PALETTE['border']};
        margin: 1.5rem 0;
    }}

    /* ---- Streamlit dataframe tweaks ---- */
    [data-testid="stDataFrame"] {{
        border: 0.5px solid {PALETTE['border']};
        border-radius: 8px;
    }}

    /* ---- Hide empty containers that streamlit leaves around ---- */
    .stMarkdown:empty {{ display: none; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# Sidebar — run a new analysis
# ============================================================
with st.sidebar:
    st.markdown("### Run analysis")
    query_input = st.text_area(
        "Query",
        placeholder="e.g. sentiment in Tehran on the US and the new administration",
        height=100,
    )
    lang_options = ["en", "fa", "ar", "es", "fr", "de", "zh", "ru", "tr", "pt"]
    languages_input = st.multiselect("Languages", lang_options, default=["en"])
    hours_input = st.slider("Time window (hours)", 6, 168, 24, step=6)
    run_clicked = st.button("Run", type="primary", use_container_width=True)
    st.markdown(
        "<div style='font-size:11px;color:#888;margin-top:8px'>"
        "Takes 3–7 min. One run at a time — don't click twice."
        "</div>",
        unsafe_allow_html=True,
    )

if run_clicked:
    if not query_input.strip():
        st.sidebar.error("Enter a query first.")
    elif not languages_input:
        st.sidebar.error("Select at least one language.")
    else:
        from analyze import run_pipeline, _invalidate_from
        _invalidate_from("scrape")
        try:
            with st.spinner("Running pipeline — this takes 3–7 minutes…"):
                run_pipeline(query_input.strip(), languages_input, hours_input)
            st.cache_data.clear()
            st.rerun()
        except SystemExit:
            st.error("No tweets found for that query. Try broadening the search or changing the time window.")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")


# ============================================================
# Load
# ============================================================
@st.cache_data
def load_bundle():
    p = CACHE_DIR / "final_bundle.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


bundle = load_bundle()
if bundle is None:
      _, col, _ = st.columns([1, 2, 1])
      with col:
          st.markdown("<h2 class='serif' style='margin-bottom:1rem'>Run an analysis</h2>", unsafe_allow_html=True)
          q = st.text_area("Query", placeholder="e.g. sentiment in Tehran on the
  US and the new administration", height=100, key="main_query")
          langs = st.multiselect("Languages", ["en", "fa", "ar", "es", "fr",
  "de", "zh", "ru", "tr", "pt"], default=["en"], key="main_langs")
          hrs = st.slider("Time window (hours)", 6, 168, 24, step=6,
  key="main_hours")
          if st.button("Run", type="primary", use_container_width=True,
  key="main_run"):
              if not q.strip():
                  st.error("Enter a query first.")
              elif not langs:
                  st.error("Select at least one language.")
              else:
                  from analyze import run_pipeline, _invalidate_from
                  _invalidate_from("scrape")
                  try:
                      with st.spinner("Running pipeline — this takes 3–7 minutes…"):
                          run_pipeline(q.strip(), langs, hrs)
                      st.cache_data.clear()
                      st.rerun()
                  except SystemExit:
                      st.error("No tweets found for that query. Try broadening the search or changing the time window.")
                  except Exception as e:
                      st.error(f"Pipeline failed: {e}")
          st.markdown(
              "<div style='font-size:11px;color:#888;margin-top:8px;text-align:center'>"
              "Takes 3–7 min. One run at a time — don't click twice.</div>",
              unsafe_allow_html=True,
          )
      st.stop()

df = bundle["df"]
narrative = bundle["narrative"]
outliers = bundle["outliers"]
temporal = bundle["temporal"]
spike_summaries = bundle["spike_summaries"]
kg = bundle["knowledge_graph"]
target_entities = bundle["localized"]["target_entities"]
entity_names = [e["name"] for e in target_entities]


# ============================================================
# Header
# ============================================================
import datetime as _dt
run_ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

st.markdown(
    f"<h1 class='serif'>xnarrative<span class='run-ts'>RUN {run_ts}</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='meta-line'>"
    f"Query: <span class='v'>\"{bundle['query']}\"</span> · "
    f"Languages: <span class='v'>{', '.join(bundle['languages'])}</span> · "
    f"Window: <span class='v'>last {bundle['hours']}h</span> · "
    f"n = <span class='v'>{len(df)} tweets</span>"
    f"</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Stat row
# ============================================================
def stat(label: str, value: str | int, accent: str | None = None) -> str:
    color_style = f"color: {accent};" if accent else ""
    return (
        f"<div class='stat-block'>"
        f"<div class='stat-num' style='{color_style}'>{value}</div>"
        f"<div class='stat-label'>{label}</div>"
        f"</div>"
    )


sc1, sc2, sc3, sc4, sc5 = st.columns(5)
with sc1:
    st.markdown(stat("Tweets", len(df)), unsafe_allow_html=True)
with sc2:
    st.markdown(stat("Languages", df["lang"].nunique()), unsafe_allow_html=True)
with sc3:
    st.markdown(stat("Authors", df["author_id"].nunique()), unsafe_allow_html=True)
with sc4:
    n_eng = len(outliers["engagement_outliers"])
    st.markdown(
        stat("Eng. outliers", n_eng, accent=PALETTE["amber_strong"] if n_eng else None),
        unsafe_allow_html=True,
    )
with sc5:
    n_spike = len(temporal["spikes"])
    st.markdown(
        stat("Temporal spikes", n_spike, accent="#A32D2D" if n_spike else None),
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 1.25rem'></div>", unsafe_allow_html=True)


# ============================================================
# Tabs
# ============================================================
tab_overview, tab_timeline, tab_outliers, tab_perlang, tab_kg, tab_raw = st.tabs([
    "Overview", "Timeline", "Outliers", "Per-language", "Knowledge graph", "Raw tweets",
])


# ============================================================
# Helpers used across tabs
# ============================================================
def render_sentiment_bar(entity: str, df_subset: pd.DataFrame) -> str:
    """Return HTML for a single per-entity sentiment bar (support/mixed/oppose %)."""
    stances = [d.get(entity) for d in df_subset["stance_by_entity"] if isinstance(d, dict)]
    on_topic = [s for s in stances if s and s != "off-topic"]
    if not on_topic:
        return (
            f"<div class='sentiment-row'>"
            f"<div class='sentiment-row-label'>"
            f"<span>{entity}</span><span class='sentiment-row-n'>n=0</span>"
            f"</div>"
            f"<div style='font-size:11px;color:{PALETTE['text_tertiary']}'>No on-topic tweets</div>"
            f"</div>"
        )
    total = len(on_topic)
    s = on_topic.count("support")
    m = on_topic.count("mixed")
    o = on_topic.count("oppose")
    s_pct = round(100 * s / total)
    m_pct = round(100 * m / total)
    o_pct = 100 - s_pct - m_pct  # ensure sums to 100

    def seg(cls: str, pct: int) -> str:
        if pct <= 0:
            return ""
        label = f"{pct}%" if pct >= 8 else ""  # hide label on thin slivers
        return f"<div class='{cls}' style='width:{pct}%'>{label}</div>"

    return (
        f"<div class='sentiment-row'>"
        f"<div class='sentiment-row-label'>"
        f"<span>{entity}</span><span class='sentiment-row-n'>n={total}</span>"
        f"</div>"
        f"<div class='sentiment-bar'>"
        f"{seg('seg-support', s_pct)}{seg('seg-mixed', m_pct)}{seg('seg-oppose', o_pct)}"
        f"</div>"
        f"</div>"
    )


def render_tweet_card(
    row: dict,
    accent: str = "engagement",
    score_text: str = "",
    score_class: str = "score-amber",
) -> None:
    text = row.get("translated_text") or row.get("text") or ""
    author = row.get("author_id", "unknown")
    followers = int(row.get("author_followers", 0) or 0)
    likes = int(row.get("like_count", 0) or 0)
    rts = int(row.get("retweet_count", 0) or 0)
    url = row.get("raw_url", "#")
    lang = row.get("lang", "?")
    score_html = (
        f" · <span class='{score_class}'>{score_text}</span>" if score_text else ""
    )
    st.markdown(
        f"<div class='tweet-card {accent}'>"
        f"{text[:500]}"
        f"<div class='tweet-meta'>"
        f"<a href='{url}' target='_blank'>@{author}</a> · "
        f"{followers:,} followers · {likes:,} ♥ · {rts:,} ↻ · {lang}"
        f"{score_html}"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    # Bottom line callout
    bottom_line = narrative.get("bottom_line", "").strip()
    if bottom_line:
        st.markdown(
            f"<div class='bottom-line'>"
            f"<div class='bottom-line-label'>BOTTOM LINE</div>"
            f"<div class='bottom-line-text'>{bottom_line}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    col_main, col_side = st.columns([2, 1], gap="large")

    # --- Main column: narrative + cross-language divergence ---
    with col_main:
        st.markdown("<h2 class='serif'>Narrative</h2>", unsafe_allow_html=True)
        agg = narrative.get("aggregated_narrative", "")
        # Convert blank-line-separated paragraphs into <p> for our prose styling
        paragraphs = [p.strip() for p in agg.split("\n\n") if p.strip()]
        prose_html = "".join(f"<p>{p}</p>" for p in paragraphs)
        st.markdown(f"<div class='narrative-prose'>{prose_html}</div>", unsafe_allow_html=True)

        divergence = narrative.get("cross_language_divergence", "").strip()
        if divergence:
            st.markdown("<h2 class='serif' style='margin-top:1.25rem'>Cross-language divergence</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='narrative-prose'>{divergence}</div>", unsafe_allow_html=True)

        caveats = narrative.get("confidence_caveats", "").strip()
        if caveats:
            with st.expander("Confidence caveats"):
                st.markdown(caveats)

    # --- Side column: sentiment bars + top themes ---
    with col_side:
        st.markdown("<h3>Sentiment by entity</h3>", unsafe_allow_html=True)
        sentiment_html = "".join(render_sentiment_bar(ent, df) for ent in entity_names)
        st.markdown(sentiment_html, unsafe_allow_html=True)
        st.markdown(
            f"<div class='sentiment-legend'>"
            f"<span><span class='swatch' style='background:{PALETTE['support']}'></span>Support</span>"
            f"<span><span class='swatch' style='background:{PALETTE['mixed']}'></span>Mixed</span>"
            f"<span><span class='swatch' style='background:{PALETTE['oppose']}'></span>Oppose</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<h3>Top themes</h3>", unsafe_allow_html=True)
        all_themes = []
        for themes in df["themes"]:
            if isinstance(themes, list):
                all_themes.extend(themes)
        theme_counts = Counter(all_themes).most_common(10)
        if theme_counts:
            max_count = theme_counts[0][1]
            # Three intensity bands for color
            for i, (theme, count) in enumerate(theme_counts):
                pct = max(20, int(100 * count / max_count))  # min width so labels stay readable
                if i < 3:
                    bar_color = "#BA7517"  # amber-600
                elif i < 6:
                    bar_color = "#EF9F27"  # amber-400
                else:
                    bar_color = "#FAC775"  # amber-200
                st.markdown(
                    f"<div class='theme-row'>"
                    f"<div class='theme-bar' style='width:{pct}%; background:{bar_color}'>"
                    f"{theme} {count}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ============================================================
# TIMELINE
# ============================================================
with tab_timeline:
    st.markdown("<h2 class='serif'>Tweet volume over time</h2>", unsafe_allow_html=True)

    binned = temporal["binned"]
    spikes = temporal["spikes"]

    if not binned.empty:
        fig = go.Figure()
        # Volume area
        fig.add_trace(go.Scatter(
            x=binned["bucket_start"], y=binned["count"],
            mode="lines",
            line=dict(color=PALETTE["amber_strong"], width=2, shape="spline", smoothing=0.6),
            fill="tozeroy",
            fillcolor="rgba(186, 117, 23, 0.10)",
            name="Volume",
            hovertemplate="%{x|%H:%M}<br>%{y} tweets<extra></extra>",
        ))
        # Baseline (if we have it inline; otherwise compute a quick rolling median)
        if "count" in binned.columns and len(binned) >= 5:
            baseline = binned["count"].rolling(window=max(5, len(binned)//6), min_periods=1, center=True).median()
            fig.add_trace(go.Scatter(
                x=binned["bucket_start"], y=baseline,
                mode="lines",
                line=dict(color="#888780", width=1, dash="dot"),
                name="Baseline",
                hoverinfo="skip",
            ))
        # Spike markers
        if not spikes.empty:
            fig.add_trace(go.Scatter(
                x=spikes["bucket_start"], y=spikes["count"],
                mode="markers",
                marker=dict(size=12, color="#A32D2D", symbol="diamond",
                            line=dict(width=1, color="#FFFFFF")),
                name="Spike",
                hovertemplate="%{x|%H:%M}<br>%{y} tweets (spike)<extra></extra>",
            ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color=PALETTE["text_secondary"],
            font_family="-apple-system, Inter, sans-serif",
            font_size=11,
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                gridcolor="rgba(60,52,40,0.08)", showline=False,
                tickfont=dict(color=PALETTE["text_tertiary"]),
            ),
            yaxis=dict(
                gridcolor="rgba(60,52,40,0.08)", showline=False,
                title="tweets / bin",
                title_font=dict(size=10, color=PALETTE["text_tertiary"]),
                tickfont=dict(color=PALETTE["text_tertiary"]),
            ),
            showlegend=True,
            legend=dict(
                orientation="h", y=-0.18, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Spike summaries
    if spike_summaries:
        st.markdown("<h2 class='serif' style='margin-top:1.5rem'>Spike summaries</h2>", unsafe_allow_html=True)
        for s in spike_summaries:
            ts = s.get("start", "")
            try:
                ts_fmt = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                ts_fmt = ts
            st.markdown(
                f"<div class='spike-card'>"
                f"<div class='spike-card-header'>"
                f"<span class='spike-card-ts'>{ts_fmt}</span>"
                f"<span class='spike-card-meta'>"
                f"{s.get('tweet_count', '?')} tweets · baseline {s.get('baseline', '?')} · z = {s.get('zscore', 0):.1f}"
                f"</span>"
                f"</div>"
                f"<div class='spike-card-body'>{s.get('summary', '')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Cadence
    cadence = temporal.get("cadence", {})
    if cadence.get("corpus_autocorr_peak_minutes") or cadence.get("suspicious_authors"):
        st.markdown("<h2 class='serif' style='margin-top:1rem'>Posting cadence</h2>", unsafe_allow_html=True)
        if cadence.get("corpus_autocorr_peak_minutes"):
            mins = cadence["corpus_autocorr_peak_minutes"]
            st.markdown(
                f"<div class='surface-card'>"
                f"Corpus-wide volume shows a periodic peak at "
                f"<strong style='font-weight:500'>{mins} min</strong> intervals — "
                f"moderate evidence of coordinated amplification timing."
                f"</div>",
                unsafe_allow_html=True,
            )
        if cadence.get("suspicious_authors"):
            sus_df = pd.DataFrame(cadence["suspicious_authors"])
            # Reorder + rename columns for display
            display_cols = [c for c in ["author_id", "interval_minutes", "n_posts", "cv"] if c in sus_df.columns]
            sus_display = sus_df[display_cols].copy()
            sus_display.columns = ["Author", "Interval (min)", "N posts", "CV"][: len(display_cols)]
            if "Interval (min)" in sus_display.columns:
                sus_display["Interval (min)"] = sus_display["Interval (min)"].round(1)
            if "CV" in sus_display.columns:
                sus_display["CV"] = sus_display["CV"].round(2)
            st.markdown(
                "<div class='surface-card'>"
                "<div class='surface-card-title'>Authors with suspiciously regular posting</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(sus_display, use_container_width=True, hide_index=True)


# ============================================================
# OUTLIERS
# ============================================================
with tab_outliers:
    col_eng, col_sent = st.columns(2, gap="large")

    with col_eng:
        st.markdown("<h2 class='serif'>Engagement outliers</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:11.5px; color:{PALETTE['text_secondary']}; margin-bottom:12px'>"
            f"Disproportionate reach given follower count"
            f"</div>",
            unsafe_allow_html=True,
        )
        eng_outliers = outliers["engagement_outliers"][:10]
        if eng_outliers:
            for o in eng_outliers:
                z = o.get("engagement_zscore", 0)
                render_tweet_card(
                    o, accent="engagement",
                    score_text=f"z = {z:.1f}",
                    score_class="score-amber",
                )
        else:
            st.markdown(
                f"<div style='color:{PALETTE['text_tertiary']}; font-size:12px'>None detected.</div>",
                unsafe_allow_html=True,
            )

    with col_sent:
        st.markdown("<h2 class='serif'>Sentiment outliers</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:11.5px; color:{PALETTE['text_secondary']}; margin-bottom:12px'>"
            f"Cuts against the dominant narrative"
            f"</div>",
            unsafe_allow_html=True,
        )
        sent_outliers = outliers["sentiment_outliers"][:10]
        if sent_outliers:
            for o in sent_outliers:
                score = o.get("outlier_score", 0)
                sent = o.get("sentiment_score_avg", 0)
                render_tweet_card(
                    o, accent="sentiment",
                    score_text=f"score = {score:.2f}, sent = {sent:+.1f}",
                    score_class="score-pink",
                )
        else:
            st.markdown(
                f"<div style='color:{PALETTE['text_tertiary']}; font-size:12px'>None detected.</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# PER-LANGUAGE
# ============================================================
with tab_perlang:
    per_lang = narrative.get("per_language_summaries", {})
    languages = sorted(df["lang"].unique().tolist())
    if not languages:
        st.markdown("No language data.")
    else:
        sub_tabs = st.tabs([f"{l} (n={len(df[df['lang']==l])})" for l in languages])
        for tab, lang in zip(sub_tabs, languages):
            with tab:
                sub = df[df["lang"] == lang]
                lang_summary = per_lang.get(lang, "").strip()
                if lang_summary:
                    st.markdown(
                        f"<div class='narrative-prose' style='margin-bottom:1.25rem'>{lang_summary}</div>",
                        unsafe_allow_html=True,
                    )

                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    st.markdown("<h3>Themes</h3>", unsafe_allow_html=True)
                    themes = []
                    for t in sub["themes"]:
                        if isinstance(t, list):
                            themes.extend(t)
                    if themes:
                        top = Counter(themes).most_common(10)
                        max_c = top[0][1]
                        for i, (theme, c) in enumerate(top):
                            pct = max(20, int(100 * c / max_c))
                            bar_color = "#BA7517" if i < 3 else ("#EF9F27" if i < 6 else "#FAC775")
                            st.markdown(
                                f"<div class='theme-row'>"
                                f"<div class='theme-bar' style='width:{pct}%; background:{bar_color}'>"
                                f"{theme} {c}"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )
                with col_b:
                    st.markdown("<h3>Sentiment by entity (this language)</h3>", unsafe_allow_html=True)
                    if not sub.empty:
                        sentiment_html = "".join(render_sentiment_bar(ent, sub) for ent in entity_names)
                        st.markdown(sentiment_html, unsafe_allow_html=True)


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================
with tab_kg:
    entities = kg.get("entities", [])
    relations = kg.get("relations", [])
    st.markdown(
        f"<div style='display:flex; align-items:baseline; justify-content:space-between; margin-bottom:4px'>"
        f"<h2 class='serif' style='margin:0'>Knowledge graph</h2>"
        f"<span style='font-size:11px; color:{PALETTE['text_secondary']}'>"
        f"{len(entities)} entities · {len(relations)} relations</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if entities and relations:
        try:
            from pyvis.network import Network
            net = Network(
                height="500px", width="100%",
                bgcolor=PALETTE["surface"], font_color=PALETTE["text"],
                directed=True,
            )
            for e in entities:
                etype = e.get("type", "concept")
                colors = PALETTE.get(etype, PALETTE["concept"])
                aliases = e.get("aliases", []) or []
                title = f"{etype}: {e['name']}"
                if aliases:
                    title += f"\nAliases: {', '.join(aliases)}"
                net.add_node(
                    e["id"],
                    label=e["name"],
                    title=title,
                    color={
                        "background": colors["fill"],
                        "border": colors["border"],
                        "highlight": {"background": colors["fill"], "border": colors["border"]},
                    },
                    font={"color": colors["text"], "size": 14},
                    borderWidth=1,
                    shape="dot" if etype == "person" else "box",
                )
            for r in relations:
                stance = r.get("stance", "neutral")
                edge_color = {
                    "supportive": PALETTE["edge_supportive"],
                    "critical":   PALETTE["edge_critical"],
                    "neutral":    PALETTE["edge_neutral"],
                    "ambiguous":  PALETTE["edge_neutral"],
                }.get(stance, PALETTE["edge_neutral"])
                evidence_n = len(r.get("evidence_tweet_ids", []))
                net.add_edge(
                    r["source_id"], r["target_id"],
                    label=r["relation"],
                    color=edge_color,
                    title=f"{r['relation']} ({stance})\nEvidence: {evidence_n} tweet(s)",
                    arrows="to",
                    font={"size": 11, "color": PALETTE["text_secondary"], "strokeWidth": 0},
                )
            net.set_options("""
            {
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -10000,
                  "springLength": 160,
                  "springConstant": 0.04,
                  "damping": 0.35
                },
                "stabilization": { "iterations": 200 }
              },
              "edges": {
                "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
                "smooth": { "type": "continuous" }
              },
              "interaction": { "hover": true }
            }
            """)
            html = net.generate_html(notebook=False)
            st.components.v1.html(html, height=520, scrolling=False)
        except Exception as e:
            st.error(f"KG render failed: {e}")
            st.json({"entities": entities, "relations": relations})

        # Legend
        legend_parts = []
        type_label_map = {
            "person": "Person", "country": "Country", "org": "Org / policy",
            "event": "Event", "place": "Place", "concept": "Concept", "movement": "Movement",
        }
        # Only show legend entries for types actually present
        types_present = sorted({e.get("type", "concept") for e in entities})
        for t in types_present:
            colors = PALETTE.get(t, PALETTE["concept"])
            label = type_label_map.get(t, t.title())
            shape_style = "border-radius:50%;" if t == "person" else ""
            legend_parts.append(
                f"<span style='display:flex; align-items:center; gap:5px'>"
                f"<span style='width:14px; height:14px; background:{colors['fill']}; "
                f"border:0.5px solid {colors['border']}; display:inline-block; {shape_style}'></span>"
                f"{label}"
                f"</span>"
            )
        stance_parts = [
            (PALETTE["edge_critical"], "Critical"),
            (PALETTE["edge_supportive"], "Supportive"),
            (PALETTE["edge_neutral"], "Neutral / factual"),
        ]
        for color, label in stance_parts:
            legend_parts.append(
                f"<span style='display:flex; align-items:center; gap:5px'>"
                f"<span style='width:18px; height:2px; background:{color}; display:inline-block'></span>"
                f"{label}"
                f"</span>"
            )
        st.markdown(
            f"<div style='display:flex; flex-wrap:wrap; gap:14px; font-size:11px; "
            f"color:{PALETTE['text_secondary']}; margin:12px 0 1rem'>"
            f"{''.join(legend_parts)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Relations table
        ent_map = {e["id"]: e["name"] for e in entities}
        rel_rows = []
        for r in relations:
            rel_rows.append({
                "Source":   ent_map.get(r["source_id"], r["source_id"]),
                "Relation": r["relation"],
                "Target":   ent_map.get(r["target_id"], r["target_id"]),
                "Stance":   r.get("stance", ""),
                "Evidence": len(r.get("evidence_tweet_ids", [])),
            })
        if rel_rows:
            rel_df = pd.DataFrame(rel_rows).sort_values("Evidence", ascending=False)
            st.markdown(
                "<div class='surface-card'>"
                "<div class='surface-card-title'>Relations · sorted by evidence strength</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(rel_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            f"<div style='color:{PALETTE['text_tertiary']}; font-size:12px'>"
            f"No entities or relations extracted.</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# RAW
# ============================================================
with tab_raw:
    st.markdown("<h2 class='serif'>Raw tweet corpus</h2>", unsafe_allow_html=True)
    display_cols = [
        "created_at", "lang", "author_id", "author_followers",
        "like_count", "retweet_count", "translated_text", "themes",
        "rhetorical_mode", "sentiment_score_avg", "raw_url",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True, hide_index=True, height=600)
