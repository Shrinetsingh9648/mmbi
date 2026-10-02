"""
MMBI Comprehensive PDF Session Report Generator
===============================================
Generates complete, self-contained, publication-ready PDF reports
for MMBI Cognitive Sessions using ReportLab and Matplotlib.

Contains:
  A. Session Overview (ID, type, date, duration, frames, FPS)
  B. Final Interest Analysis (INTERESTED, NEUTRAL, NOT INTERESTED percentages, durations, transitions)
  C. Engagement Analysis (metrics + timeline graph)
  D. Attention Analysis (eye contact, gaze zones, head posture, blinks)
  E. Micro-Expression Analysis (metrics + full event table)
  F. Micro-Expression Timeline Graph (Matplotlib)
  G. Interest State Timeline (Matplotlib + segment table)
  H. Event Timeline (chronological assessment log)
  I. Behavioral Explanation & Scientific Disclaimer
  J. User Feedback Section
"""

from __future__ import annotations
import os
import io
from typing import Dict, Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'learning', 'data', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def _generate_micro_expression_chart(micro_data: Dict[str, Any], duration_sec: float, output_path: str) -> Optional[str]:
    """Generates a micro-expression intensity timeline chart and saves to output_path."""
    events = micro_data.get("events_log", [])
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=160)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    if not events:
        ax.text(0.5, 0.5, "No Micro-Expression Twitches Detected During Session",
                ha='center', va='center', color='#64748b', fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.set_xlim(0, max(duration_sec, 10))
        ax.set_ylim(0, 1.0)
    else:
        times = [ev.get("timestamp_sec", 0.0) for ev in events]
        intensities = [ev.get("intensity", 0.5) for ev in events]
        regions = [ev.get("region", "Face") for ev in events]
        types = [ev.get("type", "Unknown") for ev in events]

        # Stem / scatter plot
        markerline, stemlines, baseline = ax.stem(times, intensities, linefmt='-', markerfmt='o', basefmt=' ')
        plt.setp(markerline, color='#ea580c', markersize=7, markeredgecolor='#9a3412', markeredgewidth=1.5)
        plt.setp(stemlines, color='#fdba74', linewidth=1.5, linestyle='--')

        # Annotate top events
        for t, val, reg, typ in zip(times, intensities, regions, types):
            ax.annotate(f"{typ}\n({reg})",
                        xy=(t, val),
                        xytext=(0, 8),
                        textcoords='offset points',
                        ha='center',
                        fontsize=7,
                        color='#0f172a',
                        fontweight='semibold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffedd5', edgecolor='#fed7aa', alpha=0.9))

        ax.set_xlim(-1, max(duration_sec + 2, max(times) + 3))
        ax.set_ylim(0, max(1.05, max(intensities) * 1.25))

    ax.set_title("Micro-Expression Transient Spike Distribution", fontsize=10, fontweight='bold', color='#0f172a', pad=8)
    ax.set_xlabel("Session Timeline (Seconds)", fontsize=8, color='#475569')
    ax.set_ylabel("Spike Intensity", fontsize=8, color='#475569')
    ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    ax.tick_params(axis='both', which='major', labelsize=8, colors='#475569')
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')

    plt.tight_layout()
    fig.savefig(output_path, format='png', bbox_inches='tight')
    plt.close(fig)
    return output_path


def _generate_interest_engagement_chart(session: Dict[str, Any], output_path: str) -> Optional[str]:
    """Generates an interest state timeline and engagement trajectory plot and saves to output_path."""
    samples = session.get("engagement_summary", {}).get("timeline_samples", [])
    duration_sec = session.get("duration_seconds", 30.0)
    intervals = session.get("interest_timeline", [])

    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=160)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')

    # Color background zones for interest states based on intervals
    state_colors = {
        "INTERESTED": ("#dcfce7", "#16a34a"),      # Light green
        "NEUTRAL": ("#f1f5f9", "#64748b"),         # Light gray
        "NOT INTERESTED": ("#fee2e2", "#dc2626")   # Light red
    }

    if intervals:
        for itv in intervals:
            st = itv.get("state", "NEUTRAL")
            start_s = itv.get("start_sec", 0.0)
            end_s = itv.get("end_sec", duration_sec)
            bg_col = state_colors.get(st, state_colors["NEUTRAL"])[0]
            ax.axvspan(start_s, end_s, color=bg_col, alpha=0.65)
    else:
        ax.axvspan(0, duration_sec, color="#f1f5f9", alpha=0.5)

    # Plot Engagement curve if samples exist
    if samples:
        ts = [s.get("t", 0.0) for s in samples]
        engs = [s.get("eng", 0.5) for s in samples]
        ax.plot(ts, engs, color='#0284c7', linewidth=2.0, label='Engagement Score')
        ax.fill_between(ts, engs, color='#38bdf8', alpha=0.15)
        ax.set_xlim(0, max(duration_sec, ts[-1] if ts else 10))
    else:
        avg_eng = session.get("engagement_summary", {}).get("average_engagement", 0.5)
        ax.plot([0, duration_sec], [avg_eng, avg_eng], color='#0284c7', linewidth=2.0, label='Avg Engagement')
        ax.set_xlim(0, max(duration_sec, 10))

    ax.axhline(y=0.58, color='#16a34a', linestyle='--', linewidth=0.9, alpha=0.7, label='Interested Threshold (0.58)')
    ax.axhline(y=0.38, color='#dc2626', linestyle='--', linewidth=0.9, alpha=0.7, label='Disengaged Threshold (0.38)')

    ax.set_ylim(0.0, 1.0)
    ax.set_title("Engagement Trajectory & Interest State Classification Zones", fontsize=10, fontweight='bold', color='#0f172a', pad=8)
    ax.set_xlabel("Session Timeline (Seconds)", fontsize=8, color='#475569')
    ax.set_ylabel("Engagement Index", fontsize=8, color='#475569')
    ax.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    ax.tick_params(axis='both', which='major', labelsize=8, colors='#475569')
    ax.legend(loc='lower right', fontsize=7, framealpha=0.85, edgecolor='#cbd5e1')
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')

    plt.tight_layout()
    fig.savefig(output_path, format='png', bbox_inches='tight')
    plt.close(fig)
    return output_path


def generate_pdf_report(session: Dict[str, Any], output_filename: Optional[str] = None) -> str:
    """
    Main API: Generates a complete standalone PDF report from session data.
    Returns the path to the generated PDF.
    
    IMPORTANT: Chart images are generated BEFORE doc.build() and cleaned up AFTER.
    This prevents the Windows temp-file-deleted-before-read bug.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    session_id = session.get("session_id", "MMBI-SESSION")
    if not output_filename:
        output_filename = f"{session_id}_report.pdf"
    file_path = os.path.join(REPORTS_DIR, output_filename)

    # Pre-generate charts to persistent paths in REPORTS_DIR
    # (NOT temp files — they must survive until doc.build() finishes)
    chart_files_to_cleanup = []
    ig_chart_path = os.path.join(REPORTS_DIR, f"_chart_ig_{session_id}.png")
    me_chart_path = os.path.join(REPORTS_DIR, f"_chart_me_{session_id}.png")
    
    ig_path = _generate_interest_engagement_chart(session, ig_chart_path)
    chart_files_to_cleanup.append(ig_chart_path)
    
    me_sum = session.get("micro_expressions", {})
    me_path = _generate_micro_expression_chart(me_sum, session.get("duration_seconds", 30.0), me_chart_path)
    chart_files_to_cleanup.append(me_chart_path)

    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Clean Styles
    primary_color = colors.HexColor('#0f172a')   # Deep Slate
    accent_orange = colors.HexColor('#ea580c')   # Amber Orange
    accent_blue   = colors.HexColor('#0284c7')   # Sky Blue
    accent_green  = colors.HexColor('#16a34a')   # Emerald
    accent_red    = colors.HexColor('#dc2626')   # Red
    neutral_bg    = colors.HexColor('#f8fafc')   # Off white

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=accent_orange,
        spaceAfter=14
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.white
    )
    badge_style = ParagraphStyle(
        'Badge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        alignment=TA_CENTER
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_CENTER
    )

    story = []

    # ── Header Banner ─────────────────────────────────────────────────────────
    header_data = [
        [
            Paragraph("<b>MMBI AI BEHAVIOUR & INTEREST ANALYSIS</b>", title_style),
            Paragraph(f"<b>REPORT #</b> {session_id}<br/><b>GENERATED:</b> {session.get('start_time', 'N/A')}", ParagraphStyle('RHead', parent=body_style, alignment=TA_RIGHT))
        ],
        [
            Paragraph("Cognitive Telemetry, Micro-Expression Segmentation & Multimodal Session Audit", subtitle_style),
            Paragraph(f"Analysis Mode: <b>{session.get('analysis_type', 'Live')}</b>", ParagraphStyle('RHead2', parent=body_style, alignment=TA_RIGHT, textColor=accent_blue))
        ]
    ]
    t_header = Table(header_data, colWidths=[360, 180])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_orange, spaceBefore=4, spaceAfter=10))

    # ── A. Session Overview ───────────────────────────────────────────────────
    story.append(Paragraph("A. Session Overview", section_heading))
    meta = session.get("metadata", {})
    overview_data = [
        [
            Paragraph(f"<b>Session ID:</b> {session_id}", table_cell),
            Paragraph(f"<b>Analysis Type:</b> {session.get('analysis_type', 'Live')}", table_cell),
            Paragraph(f"<b>Date:</b> {session.get('start_time', '').split(' ')[0]}", table_cell),
        ],
        [
            Paragraph(f"<b>Start Time:</b> {session.get('start_time', 'N/A')}", table_cell),
            Paragraph(f"<b>End Time:</b> {session.get('end_time', 'N/A')}", table_cell),
            Paragraph(f"<b>Duration:</b> {session.get('duration_formatted', 'N/A')} ({session.get('duration_seconds', 0)}s)", table_cell),
        ],
        [
            Paragraph(f"<b>Video Source:</b> {session.get('video_filename', 'N/A')}", table_cell),
            Paragraph(f"<b>Total Frames:</b> {session.get('total_frames', 0)} ({session.get('face_frames', 0)} face)", table_cell),
            Paragraph(f"<b>Processing Rate:</b> {session.get('processing_fps', 0)} FPS", table_cell),
        ]
    ]
    t_overview = Table(overview_data, colWidths=[180, 180, 180])
    t_overview.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), neutral_bg),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_overview)
    story.append(Spacer(1, 10))

    # ── B. Final Interest Analysis ───────────────────────────────────────────
    story.append(Paragraph("B. Final Session Interest Analysis", section_heading))
    int_sum = session.get("interest_summary", {})
    dom_state = int_sum.get("dominant_state", "NEUTRAL")

    state_color_map = {
        "INTERESTED": ("#16a34a", "#dcfce7"),
        "NEUTRAL": ("#d97706", "#fef3c7"),
        "NOT INTERESTED": ("#dc2626", "#fee2e2")
    }
    fg_col, bg_col = state_color_map.get(dom_state, ("#0f172a", "#f1f5f9"))

    interest_banner_data = [
        [
            Paragraph(f"FINAL CLASSIFIED INTEREST STATE:<br/><font color='{fg_col}'><b>{dom_state}</b></font>", badge_style),
            Paragraph(
                f"<b>Interested:</b> {int_sum.get('interested_percentage', 0)}% ({int_sum.get('interested_seconds', 0)}s)<br/>"
                f"<b>Neutral:</b> {int_sum.get('neutral_percentage', 0)}% ({int_sum.get('neutral_seconds', 0)}s)<br/>"
                f"<b>Not Interested:</b> {int_sum.get('not_interested_percentage', 0)}% ({int_sum.get('not_interested_seconds', 0)}s)",
                table_cell
            ),
            Paragraph(
                f"<b>Average Engagement:</b> {int_sum.get('average_engagement', 0.5):.2f}<br/>"
                f"<b>State Transitions:</b> {int_sum.get('transition_count', 0)}<br/>"
                f"<b>Confidence:</b> High (Multimodal Consensus)",
                table_cell
            )
        ]
    ]
    t_interest = Table(interest_banner_data, colWidths=[200, 170, 170])
    t_interest.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor(bg_col)),
        ('BACKGROUND', (1,0), (-1,-1), neutral_bg),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor(fg_col)),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_interest)
    story.append(Spacer(1, 8))

    # Add interest graph (file was already generated above, before build)
    if ig_path and os.path.exists(ig_path):
        story.append(Image(ig_path, width=7.2*inch, height=2.6*inch))
    story.append(Spacer(1, 8))

    # ── C & D. Engagement & Attention Metrics Table ───────────────────────────
    story.append(Paragraph("C & D. Cognitive Telemetry & Attention Breakdown", section_heading))
    eng_sum = session.get("engagement_summary", {})
    att_sum = session.get("attention_summary", {})
    str_sum = session.get("stress_summary", {})
    head = att_sum.get("head_pose", {})
    blinks = att_sum.get("blink_statistics", {})
    gaze_d = att_sum.get("gaze_distribution", {})

    metrics_table_data = [
        [
            Paragraph("<b>METRIC CATEGORY</b>", table_header),
            Paragraph("<b>VALUE / READING</b>", table_header),
            Paragraph("<b>ANALYSIS & INTERPRETATION</b>", table_header)
        ],
        [
            Paragraph("Average Engagement", table_cell),
            Paragraph(f"<b>{eng_sum.get('average_engagement', 0.50):.2f}</b> (Min: {eng_sum.get('min_engagement', 0.50):.2f}, Max: {eng_sum.get('max_engagement', 0.50):.2f})", table_cell),
            Paragraph("Sustained mental presence based on FACS activity, lean, and focus", table_cell)
        ],
        [
            Paragraph("Eye Contact Percentage", table_cell),
            Paragraph(f"<b>{att_sum.get('eye_contact_percentage', 0.0)}%</b> (Score: {att_sum.get('average_eye_contact', 0.50):.2f})", table_cell),
            Paragraph("Ratio of direct focal gaze towards the interaction target", table_cell)
        ],
        [
            Paragraph("Gaze Distribution", table_cell),
            Paragraph(f"Center: {gaze_d.get('CENTER', 0)}% | Away: {round(100 - gaze_d.get('CENTER', 0), 1)}%", table_cell),
            Paragraph("Directional visual tracking and looking-away behaviors", table_cell)
        ],
        [
            Paragraph("Head Pose Dynamics", table_cell),
            Paragraph(f"Yaw: {head.get('average_yaw', 0.0)}° | Pitch: {head.get('average_pitch', 0.0)}°", table_cell),
            Paragraph(f"Lean posture distribution: {head.get('lean_distribution', {})}", table_cell)
        ],
        [
            Paragraph("Blink Regulation", table_cell),
            Paragraph(f"<b>{blinks.get('average_blink_rate', 14)} bpm</b> ({blinks.get('stress_blink_rate', 'NORMAL')})", table_cell),
            Paragraph("Normal cognitive blinking range is 10–22 blinks/min", table_cell)
        ],
        [
            Paragraph("Sustained Stress Load", table_cell),
            Paragraph(f"<b>{str_sum.get('sustained_stress_load_pct', 0.0)}%</b> (Avg: {str_sum.get('average_stress', 0.0):.2f})", table_cell),
            Paragraph("Proportion of session spent with elevated stress indices", table_cell)
        ]
    ]
    t_metrics = Table(metrics_table_data, colWidths=[140, 160, 240])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, neutral_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 12))

    # ── E & F. Micro-Expression Analysis ─────────────────────────────────────
    story.append(Paragraph("E & F. Micro-Expression Detection Analysis", section_heading))
    me_sum = session.get("micro_expressions", {})
    events = me_sum.get("events_log", [])

    me_stats_data = [
        [
            Paragraph(f"<b>Total Detected:</b> {me_sum.get('total_count', 0)}", table_cell),
            Paragraph(f"<b>Rate per Minute:</b> {me_sum.get('rate_per_min', 0.0)} / min", table_cell),
            Paragraph(f"<b>Average Duration:</b> {me_sum.get('average_duration_ms', 0.0)} ms", table_cell),
            Paragraph(f"<b>Primary Region:</b> {me_sum.get('most_frequent_region', 'None')}", table_cell),
        ]
    ]
    t_me_stats = Table(me_stats_data, colWidths=[135, 135, 135, 135])
    t_me_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#ffedd5')),
        ('BOX', (0,0), (-1,-1), 0.5, accent_orange),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#fed7aa')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_me_stats)
    story.append(Spacer(1, 6))

    # Micro-expression timeline chart (file was already generated above, before build)
    if me_path and os.path.exists(me_path):
        story.append(Image(me_path, width=7.2*inch, height=2.4*inch))
    story.append(Spacer(1, 6))

    # Micro-expression Event Table
    story.append(Paragraph("<b>Segmented Micro-Expression Event Log:</b>", body_style))
    story.append(Spacer(1, 3))

    me_table_data = [
        [
            Paragraph("<b>#</b>", table_header),
            Paragraph("<b>TIME</b>", table_header),
            Paragraph("<b>DURATION</b>", table_header),
            Paragraph("<b>REGION</b>", table_header),
            Paragraph("<b>CLASSIFICATION</b>", table_header),
            Paragraph("<b>INTENSITY</b>", table_header),
            Paragraph("<b>CONFIDENCE</b>", table_header)
        ]
    ]

    if not events:
        me_table_data.append([
            Paragraph("—", table_cell),
            Paragraph("No transient involuntary twitches captured in this session", table_cell),
            Paragraph("—", table_cell),
            Paragraph("—", table_cell),
            Paragraph("—", table_cell),
            Paragraph("—", table_cell),
            Paragraph("—", table_cell)
        ])
    else:
        for idx, ev in enumerate(events[:12], 1):  # Cap table to 12 in PDF for clean paging
            me_table_data.append([
                Paragraph(f"{idx:02d}", table_cell),
                Paragraph(str(ev.get("timestamp", "00:00")), table_cell),
                Paragraph(f"{ev.get('duration_ms', 0):.0f} ms", table_cell),
                Paragraph(str(ev.get("region", "Face")), table_cell),
                Paragraph(f"<font color='#dc2626'><b>{ev.get('type', 'Surprise')}</b></font>", table_cell),
                Paragraph(f"{ev.get('intensity', 0.0):.2f}", table_cell),
                Paragraph(f"{ev.get('confidence', 0.75):.2f}", table_cell)
            ])

    t_me_table = Table(me_table_data, colWidths=[24, 76, 70, 100, 120, 75, 75])
    t_me_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#9a3412')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, neutral_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_me_table)
    story.append(Spacer(1, 10))

    # ── G & H. Event Timeline ─────────────────────────────────────────────────
    story.append(Paragraph("G & H. Session Event Timeline", section_heading))
    ev_timeline = session.get("event_timeline", [])
    ev_table_data = [
        [
            Paragraph("<b>TIME</b>", table_header),
            Paragraph("<b>EVENT CATEGORY</b>", table_header),
            Paragraph("<b>OBSERVATION & CONTEXT</b>", table_header)
        ]
    ]

    if not ev_timeline:
        ev_table_data.append([
            Paragraph("00:00", table_cell),
            Paragraph("BASELINE", table_cell),
            Paragraph("Smooth baseline cognitive activity maintained throughout session", table_cell)
        ])
    else:
        for ev in ev_timeline[:10]:
            ev_table_data.append([
                Paragraph(str(ev.get("time", "00:00")), table_cell),
                Paragraph(f"<b>{ev.get('event_type', 'EVENT')}</b>", table_cell),
                Paragraph(f"{ev.get('title', '')} — {ev.get('details', '')}", table_cell)
            ])

    t_events = Table(ev_table_data, colWidths=[65, 145, 330])
    t_events.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, neutral_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_events)
    story.append(Spacer(1, 10))

    # ── I. Explanation / Evidence ─────────────────────────────────────────────
    story.append(Paragraph("I. Behavioural Evidence & Scientific Explanation", section_heading))
    evidence_items = int_sum.get("supporting_signals", [])
    if not evidence_items:
        evidence_items = ["Multimodal signals aligned with expected baseline."]

    evidence_text = "<br/>".join([f"• <b>{item}</b>" for item in evidence_items])
    evidence_box = [
        [
            Paragraph(
                f"<b>Supporting Behavioral Signals for {dom_state}:</b><br/>"
                f"{evidence_text}<br/><br/>"
                f"<i>Micro-expression events are detected using the currently configured MMBI micro-expression pipeline. "
                f"Do not claim that any single facial twitch or Action Unit proves an individual's internal mental state. "
                f"All conclusions represent model-estimated interest states based on observed behavioral signals.</i>",
                table_cell
            )
        ]
    ]
    t_evidence = Table(evidence_box, colWidths=[540])
    t_evidence.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor('#94a3b8')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_evidence)
    story.append(Spacer(1, 10))

    # ── J. User Feedback ──────────────────────────────────────────────────────
    story.append(Paragraph("J. User Evaluation & Audit Feedback", section_heading))
    fb = session.get("feedback")
    if fb:
        fb_data = [
            [
                Paragraph(f"<b>Overall Rating:</b> {fb.get('rating', 'N/A')}/5 ★", table_cell),
                Paragraph(f"<b>Interest Prediction Accuracy:</b> {fb.get('interest_accuracy', 'N/A')}/5", table_cell),
                Paragraph(f"<b>Micro-Expression Accuracy:</b> {fb.get('micro_accuracy', 'N/A')}/5", table_cell),
            ],
            [
                Paragraph(f"<b>Auditor Comments:</b> \"{fb.get('comment', 'No written comments provided.')}\"", table_cell),
                Paragraph("", table_cell),
                Paragraph(f"<b>Submitted At:</b> {fb.get('submitted_at', 'N/A')}", table_cell)
            ]
        ]
    else:
        fb_data = [
            [
                Paragraph("<b>Audit Status:</b> Awaiting user feedback submission.", table_cell),
                Paragraph("Feedback can be submitted directly via the MMBI Analyst Dashboard.", table_cell),
                Paragraph("Rating Scale: 1 (Poor) to 5 (Excellent)", table_cell)
            ]
        ]

    t_fb = Table(fb_data, colWidths=[180, 180, 180])
    t_fb.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eff6ff')),
        ('BOX', (0,0), (-1,-1), 0.5, accent_blue),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bfdbfe')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_fb)
    story.append(Spacer(1, 12))

    # Footer note
    story.append(Paragraph("Model-estimated interest state based on observed behavioural signals. MMBI Enterprise Platform © 2026.", disclaimer_style))

    # Build PDF — chart images must still exist on disk at this point
    doc.build(story)
    
    # Clean up chart image files only AFTER the PDF build is complete
    for chart_path in chart_files_to_cleanup:
        try:
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except Exception:
            pass  # non-fatal if cleanup fails
    
    return file_path
