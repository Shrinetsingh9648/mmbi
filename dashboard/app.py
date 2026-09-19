import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
import json
import os
import sys
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mmbi.dashboard.data_bridge import read_bridge

st.set_page_config(
    page_title="MMBI Engine Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<style>"
    ".metric-card{background:#1e2130;border-radius:12px;padding:20px;"
    "text-align:center;border:1px solid #2d3250;margin-bottom:8px}"
    ".metric-value{font-size:2.2em;font-weight:bold;margin:0}"
    ".metric-label{font-size:0.85em;color:#888;margin:0}"
    ".status-box{background:#1e2130;border-radius:8px;padding:12px;"
    "border-left:4px solid #4ade80;margin:6px 0}"
    ".conflict-box{background:#2a1a1a;border-radius:8px;padding:12px;"
    "border-left:4px solid #f87171;margin:6px 0}"
    ".driver-box{background:#1a1e2a;border-radius:8px;padding:10px;"
    "border-left:4px solid #60a5fa;margin:4px 0;font-size:0.85em}"
    ".au-pill{display:inline-block;background:#2d4a2d;color:#4ade80;"
    "border-radius:20px;padding:3px 12px;margin:3px;font-size:0.8em}"
    "</style>",
    unsafe_allow_html=True
)


def eng_color(score):
    if score > 0.65: return "#4ade80"
    if score > 0.45: return "#facc15"
    return "#f87171"


def stress_color(score):
    if score < 0.20: return "#4ade80"
    if score < 0.45: return "#facc15"
    if score < 0.70: return "#fb923c"
    return "#f87171"


def emotion_color(emotion):
    return {
        'happiness':  "#4ade80",
        'surprise':   "#60a5fa",
        'disgust':    "#f87171",
        'repression': "#c084fc",
        'others':     "#94a3b8",
        'neutral':    "#94a3b8"
    }.get(emotion, "#94a3b8")


def trend_arrow(trend):
    return {'RISING': '▲', 'FALLING': '▼', 'STABLE': '→'}.get(trend, '→')


def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 14, 'color': '#ccc'}},
        number={'suffix': '%', 'font': {'size': 28, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#555',
                     'tickfont': {'color': '#555'}},
            'bar': {'color': color},
            'bgcolor': '#1e2130',
            'bordercolor': '#2d3250',
            'steps': [
                {'range': [0,  35], 'color': '#1a1e2e'},
                {'range': [35, 65], 'color': '#1e2438'},
                {'range': [65, 100], 'color': '#222844'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': value * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=10),
        height=200
    )
    return fig


def make_line_chart(history):
    if not history or len(history.get('timestamps', [])) < 2:
        return None

    t0   = history['timestamps'][0]
    secs = [round(t - t0, 1) for t in history['timestamps']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=secs, y=history['engagement'],
        name='Engagement',
        line=dict(color='#4ade80', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=secs, y=history['stress'],
        name='Stress',
        line=dict(color='#f87171', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=secs, y=history['eye_contact'],
        name='Eye Contact',
        line=dict(color='#60a5fa', width=1.5, dash='dot')
    ))
    fig.update_layout(
        title=dict(text='Live Signals', font=dict(color='#ccc', size=13)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0e1117',
        font=dict(color='#888'),
        xaxis=dict(title='Seconds', gridcolor='#1e2130', color='#555'),
        yaxis=dict(range=[0, 1.05], gridcolor='#1e2130', color='#555'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#aaa', size=11)),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280
    )
    return fig


def make_micro_chart(history, micro):
    """Plots the instance-wise micro-expression probability stream
    (Phase 12): raw + smoothed probability, ON/OFF hysteresis threshold
    lines, and start/apex/end markers for every tracked event."""
    if not history or len(history.get('timestamps', [])) < 2:
        return None
    if 'micro_prob' not in history:
        return None

    t0   = history['timestamps'][0]
    secs = [round(t - t0, 1) for t in history['timestamps']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=secs, y=history.get('micro_prob', []),
        name='ME probability (raw)',
        line=dict(color='#475569', width=1, dash='dot'),
    ))
    fig.add_trace(go.Scatter(
        x=secs, y=history.get('micro_smoothed', []),
        name='ME probability (smoothed)',
        line=dict(color='#f87171', width=2.5),
        fill='tozeroy', fillcolor='rgba(248,113,113,0.08)',
    ))

    threshold_on  = (micro or {}).get('threshold_on', 0.5)
    threshold_off = (micro or {}).get('threshold_off', 0.3)
    if threshold_on is not None:
        fig.add_hline(y=threshold_on, line=dict(color='#f87171', width=1, dash='dash'),
                      annotation_text='ON', annotation_font_color='#f87171')
    if threshold_off is not None:
        fig.add_hline(y=threshold_off, line=dict(color='#60a5fa', width=1, dash='dash'),
                      annotation_text='OFF', annotation_font_color='#60a5fa')

    for ev in (micro or {}).get('events', [])[-20:]:
        apex_t = ev.get('apex_time')
        if apex_t is None:
            continue
        fig.add_trace(go.Scatter(
            x=[round(apex_t - t0, 1)], y=[ev.get('confidence') or 1.0],
            mode='markers+text',
            marker=dict(color='#c084fc', size=10, symbol='diamond'),
            text=[ev.get('class', '?')],
            textposition='top center',
            textfont=dict(color='#c084fc', size=10),
            name=f"Event {ev.get('event_id')}",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text='Micro-expression probability (instance-wise)',
                   font=dict(color='#ccc', size=13)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0e1117',
        font=dict(color='#888'),
        xaxis=dict(title='Seconds', gridcolor='#1e2130', color='#555'),
        yaxis=dict(title='Probability', range=[0, 1.05], gridcolor='#1e2130', color='#555'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#aaa', size=10)),
        margin=dict(l=40, r=20, t=40, b=40),
        height=280
    )
    return fig


def make_blink_chart(history):
    if not history or len(history.get('timestamps', [])) < 2:
        return None

    t0   = history['timestamps'][0]
    secs = [round(t - t0, 1) for t in history['timestamps']]

    fig = go.Figure(go.Scatter(
        x=secs, y=history['blink_rate'],
        fill='tozeroy',
        line=dict(color='#c084fc', width=2),
        fillcolor='rgba(192,132,252,0.15)',
        name='Blink Rate'
    ))
    fig.add_hrect(
        y0=12, y1=20,
        fillcolor='rgba(74,222,128,0.06)',
        line_width=0,
        annotation_text="Normal",
        annotation_font_color="#4ade80",
        annotation_font_size=10
    )
    fig.update_layout(
        title=dict(text='Blink Rate / min', font=dict(color='#ccc', size=13)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0e1117',
        font=dict(color='#888'),
        xaxis=dict(title='Seconds', gridcolor='#1e2130', color='#555'),
        yaxis=dict(title='Blinks/min', gridcolor='#1e2130', color='#555'),
        margin=dict(l=40, r=20, t=40, b=40),
        height=200
    )
    return fig


def make_emotion_pie(emotions):
    if not emotions:
        return None

    counts = Counter(emotions)
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [emotion_color(e) for e in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color='#0e1117', width=2)),
        textfont=dict(color='white', size=11),
        hole=0.45
    ))
    fig.update_layout(
        title=dict(text='Emotion Distribution',
                   font=dict(color='#ccc', size=13)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#888'),
        legend=dict(font=dict(color='#aaa', size=10),
                    bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=220
    )
    return fig


def main():

    with st.sidebar:
        st.markdown("## MMBI Engine")
        st.markdown("**Multimodal Behaviour Intelligence**")
        st.markdown("---")
        refresh_rate = st.slider("Refresh rate (seconds)", 0.5, 3.0, 1.0, 0.5)
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("1. Run engine in terminal 1:")
        st.code("python -m mmbi.engine")
        st.markdown("2. This dashboard auto-updates")
        st.markdown("---")
        st.markdown("### Engine keys")
        st.markdown("- 1 = Engaged")
        st.markdown("- 2 = Neutral")
        st.markdown("- 3 = Disengaged")
        st.markdown("- 4 = Stressed")
        st.markdown("- 5 = Calm")
        st.markdown("- W = Save session")
        st.markdown("- Q = Quit")
        st.markdown("---")
        st.markdown("### Signal Guide")
        st.markdown("🟢 Engagement > 0.65 = HIGH")
        st.markdown("🟡 0.45 to 0.65 = NEUTRAL")
        st.markdown("🔴 Below 0.45 = LOW")
        st.markdown("---")
        st.markdown("🟢 Stress < 0.20 = CALM")
        st.markdown("🟡 0.20 to 0.45 = MILD")
        st.markdown("🟠 0.45 to 0.70 = MODERATE")
        st.markdown("🔴 Above 0.70 = HIGH")

    st.markdown("# 🧠 MMBI Live Dashboard")
    st.markdown("*Real-time facial behaviour intelligence*")
    st.markdown(f"*{datetime.now().strftime('%H:%M:%S')}*")
    st.markdown("---")

    data = read_bridge()

    if data is None:
        st.markdown("## ⏳ Waiting for engine...")
        st.markdown("Run the engine in a separate terminal:")
        st.code("python -m mmbi.engine")
        st.markdown("Dashboard updates automatically when engine starts.")
        time.sleep(refresh_rate)
        st.rerun()
        return

    eng    = data.get('engagement_score', 0)
    stress = data.get('stress_index', 0)
    hist   = data.get('history', {})

    # ROW 1 — Gauges
    col1, col2, col3, col4, col5 = st.columns([2, 2, 1.5, 1.5, 1.5])

    with col1:
        st.plotly_chart(
            make_gauge(eng, "ENGAGEMENT", eng_color(eng)),
            use_container_width=True, key="g_eng"
        )

    with col2:
        st.plotly_chart(
            make_gauge(stress, "STRESS", stress_color(stress)),
            use_container_width=True, key="g_str"
        )

    with col3:
        ec = data.get('eye_contact_score', 0)
        st.markdown(
            f"<div class='metric-card'>"
            f"<p class='metric-value' style='color:{eng_color(ec)}'>{ec:.2f}</p>"
            f"<p class='metric-label'>Eye Contact</p>"
            f"<p style='color:#555;font-size:0.75em'>{data.get('attention_zone','?')}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        br = data.get('blink_rate', 0)
        bc = "#4ade80" if 12 <= br <= 20 else "#f87171"
        st.markdown(
            f"<div class='metric-card'>"
            f"<p class='metric-value' style='color:{bc}'>{br}</p>"
            f"<p class='metric-label'>Blinks / min</p>"
            f"<p style='color:#555;font-size:0.75em'>{data.get('blink_stress','?')}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col4:
        em   = data.get('dominant_emotion', 'neutral')
        emc  = emotion_color(em)
        emcf = data.get('emotion_confidence', 0)
        st.markdown(
            f"<div class='metric-card'>"
            f"<p class='metric-value' style='color:{emc};font-size:1.5em'>{em.upper()}</p>"
            f"<p class='metric-label'>Emotion</p>"
            f"<p style='color:#555;font-size:0.75em'>"
            f"conf={emcf:.2f} [{data.get('emotion_source','?')}]</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        lean  = data.get('lean_signal', 'NEUTRAL')
        leanc = "#4ade80" if lean == 'FORWARD' else \
                "#f87171" if lean == 'AWAY'    else "#94a3b8"
        st.markdown(
            f"<div class='metric-card'>"
            f"<p class='metric-value' style='color:{leanc};font-size:1.5em'>{lean}</p>"
            f"<p class='metric-label'>Head Lean</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col5:
        el  = data.get('engagement_label', '?')
        et  = data.get('eng_trend', 'STABLE')
        sl  = data.get('stress_label', '?')
        st_ = data.get('str_trend', 'STABLE')
        st.markdown(
            f"<div class='metric-card'>"
            f"<p class='metric-value' style='color:{eng_color(eng)};font-size:1.4em'>{el}</p>"
            f"<p class='metric-label'>Engagement {trend_arrow(et)}</p>"
            f"<p style='color:#555;font-size:0.75em'>avg={data.get('session_eng_avg',0):.2f}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='metric-card'>"
            f"<p class='metric-value' style='color:{stress_color(stress)};font-size:1.4em'>{sl}</p>"
            f"<p class='metric-label'>Stress {trend_arrow(st_)}</p>"
            f"<p style='color:#555;font-size:0.75em'>avg={data.get('session_str_avg',0):.2f}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ROW 1.5 — Micro-expression graph (Phase 12)
    micro = data.get('micro', {})
    untrained = micro.get('untrained_heads', [])
    if untrained:
        st.markdown(
            f"<div class='conflict-box'>"
            f"<p style='color:#fca5a5;margin:0;font-size:0.85em'>"
            f"⚠ MicroLSTM heads with NO trained weights: <b>{', '.join(untrained)}</b>. "
            f"Their outputs (shown greyed out below) are random until "
            f"models/casme_trainer.py has been run and the checkpoint regenerated."
            f"</p></div>",
            unsafe_allow_html=True
        )
    fig_micro = make_micro_chart(hist, micro)
    if fig_micro:
        st.plotly_chart(fig_micro, use_container_width=True, key="micro_chart")
    else:
        st.info("Collecting micro-expression data... graph appears after a few seconds.")

    events = micro.get('events', [])
    if events:
        st.markdown("#### Detected micro-expression events (most recent 10)")
        ev_df = pd.DataFrame(events[-10:])
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ROW 2 — Charts
    col_line, col_right = st.columns([3, 1])

    with col_line:
        fig = make_line_chart(hist)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="line")
        else:
            st.info("Collecting data... graphs appear after a few seconds.")

    with col_right:
        fig2 = make_blink_chart(hist)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True, key="blink")

        fig3 = make_emotion_pie(hist.get('emotions', []))
        if fig3:
            st.plotly_chart(fig3, use_container_width=True, key="pie")

    st.markdown("---")

    # ROW 3 — Text info
    col_nl, col_au, col_drv = st.columns([2, 1.5, 1.5])

    with col_nl:
        st.markdown("#### What the engine sees")
        nl = data.get('natural_language', '')
        if nl:
            st.markdown(
                f"<div class='status-box'>"
                f"<p style='color:#e2e8f0;margin:0'>{nl}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        sustained = data.get('sustained_stress', 0)
        if sustained > 5:
            st.markdown(
                f"<div class='conflict-box'>"
                f"Stress sustained for <b>{sustained:.0f} seconds</b>"
                f"</div>",
                unsafe_allow_html=True
            )
        for f in data.get('conflict_flags', [])[:2]:
            st.markdown(
                f"<div class='conflict-box'>"
                f"<p style='color:#fca5a5;margin:0;font-size:0.82em'>{f}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

    with col_au:
        st.markdown("#### Active Action Units *(heuristic geometric proxy)*")
        st.markdown(
            "<p style='color:#666;font-size:0.72em;margin-top:-8px'>"
            "Computed from landmark distances, not a trained/FACS-validated "
            "AU detector — see technical audit.</p>",
            unsafe_allow_html=True
        )
        active_aus = data.get('active_aus', [])
        au_desc = {
            'AU1': 'Inner brow raise',
            'AU2': 'Outer brow raise',
            'AU4': 'Brow furrow',
            'AU5': 'Upper lid raise',
            'AU6': 'Cheek raiser',
            'AU12': 'Smile',
            'AU15': 'Lip corner down',
            'AU17': 'Chin raise',
            'AU25': 'Lips part',
            'AU26': 'Jaw drop'
        }
        if active_aus:
            pills = "".join(
                [f"<span class='au-pill'>{a}</span>" for a in active_aus])
            st.markdown(pills, unsafe_allow_html=True)
            for a in active_aus:
                d = au_desc.get(a, '')
                if d:
                    st.markdown(
                        f"<p style='color:#94a3b8;font-size:0.78em;margin:2px 0'>"
                        f"• {a}: {d}</p>",
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                "<p style='color:#555'>No significant AUs active</p>",
                unsafe_allow_html=True
            )

    with col_drv:
        st.markdown("#### Stress Drivers")
        drivers = data.get('stress_drivers', [])
        if drivers:
            for d in drivers[:4]:
                st.markdown(
                    f"<div class='driver-box'>"
                    f"<p style='color:#93c5fd;margin:0'>{d}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<p style='color:#555'>No stress drivers detected</p>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        "<p style='color:#333;text-align:center;font-size:0.75em'>"
        "MMBI Engine — Days 1-20 | MediaPipe + PyTorch + OpenCV + Streamlit"
        "</p>",
        unsafe_allow_html=True
    )

    time.sleep(refresh_rate)
    st.rerun()


if __name__ == '__main__':
    main()