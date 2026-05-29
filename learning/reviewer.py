# mmbi/learning/reviewer.py
"""
SOTA Session Review Player & Cognitive Telemetry Analyzer
==========================================================
Provides offline command-line analysis of saved participant assessment sessions.
Parses timeline CSV logs and JSON reports to print structured report cards, 
timeline anomalies, attention dropouts, and stress spikes.

Usage:
  python mmbi/learning/reviewer.py <path_to_session_file>
"""

import os
import sys
import json
import numpy as np

def print_banner(title: str, color_code: str = "\033[93m"):
    print(f"\n{color_code}" + "=" * 70)
    print(f" {title.center(68)} ")
    print("=" * 70 + "\033[0m")

def draw_bar(label: str, pct: float, color: str = "\033[96m"):
    bar_len = int(pct / 4)
    bar = "#" * bar_len + "-" * (25 - bar_len)
    print(f"  {label:<12} {color}[{bar}] {pct:>5.1f}%\033[0m")

def analyze_session(file_path: str):
    if not os.path.exists(file_path):
        print(f"\033[91mERROR: File not found -> {file_path}\033[0m")
        sys.exit(1)

    # Resolve base paths
    base, _ = os.path.splitext(file_path)
    json_path = base + ".json"
    csv_path = base + ".csv"

    print_banner("MMBI COGNITIVE REVIEW ANALYST - POST-SESSION REPORT", "\033[95m")

    # 1. Load JSON report if exists
    report_data = None
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
        except Exception as e:
            print(f"\033[91mWarning: Failed to load JSON report: {e}\033[0m")

    if report_data:
        meta = report_data.get('metadata', {})
        cog = report_data.get('cognitive_summary', {})
        macro = report_data.get('macro_facial_expressions', {}).get('dominant_macro_breakdown_pct', {})
        micro = report_data.get('micro_expressions', {})

        # Display metadata
        print(f"\033[92m[*] SESSION PROFILE METADATA\033[0m")
        print(f"  Session ID:    {meta.get('timestamp', 'unknown')}")
        print(f"  Start Time:    {meta.get('session_start_time', 'unknown')}")
        print(f"  Duration:      {meta.get('duration_seconds', 0.0)} seconds")
        print(f"  Total Frames:  {meta.get('total_frames', 0)} frames")
        print("-" * 70)

        # Display cognitive summary
        print(f"\033[92m[*] COGNITIVE TELEMETRY AVERAGES\033[0m")
        print(f"  Average Engagement Score:  {cog.get('average_engagement', 0.5):.3f}")
        print(f"  Average Stress Index:      {cog.get('average_stress_index', 0.0):.3f}")
        print(f"  Average Eye Contact Score: {cog.get('average_eye_contact_score', 0.5):.3f}")
        print(f"  High Stress Load Duration: {cog.get('sustained_stress_load_pct', 0.0)}% of session")
        print("-" * 70)

        # Display macro expression distribution
        print(f"\033[92m[*] MACRO FACIAL STATE DISTRIBUTION\033[0m")
        for k, v in sorted(macro.items(), key=lambda x: x[1], reverse=True):
            draw_bar(k.upper(), v)
        print("-" * 70)

        # Display micro twitches
        print(f"\033[92m[*] CAPTURED INVOLUNTARY TRANSIENT TWITCHES ({micro.get('total_events_detected', 0)} detected)\033[0m")
        events = micro.get('events_log', [])
        if not events:
            print("  No micro-expression events logged.")
        else:
            print(f"  {'#':<3} {'REGION':<15} {'EMOTION':<12} {'DURATION':<10} {'TIME':<8}")
            for idx, ev in enumerate(events):
                reg = ev.get('region', 'face').replace('_', ' ').upper()
                emo = ev.get('micro_emotion', 'neutral').upper()
                dur = f"{ev.get('duration_ms', 0.0):.0f}ms"
                print(f"  {idx+1:<3} {reg:<15} {emo:<12} {dur:<10}")
        print("-" * 70)

        # Suppressed emotion analysis
        hidden_emo = micro.get('expected_hidden_emotion', 'NONE DETECTED')
        h_color = "\033[92m" if hidden_emo == "NONE DETECTED" else "\033[91m"
        print(f"\033[92m[*] INTELLIGENT REPRESSED EMOTION ANALYSIS\033[0m")
        print(f"  Expected Hidden Emotion: {h_color}[ {hidden_emo} ]\033[0m")
        print(f"  Diagnosis: {micro.get('suppressed_emotion_analysis', '')}")
        print("=" * 70)
    else:
        print("\033[93mWarning: JSON Report Card file not found. Skipping static profile metadata.\033[0m")

    # 2. Parse CSV timeline for chronological deep timeline scans
    if os.path.exists(csv_path):
        import csv
        records = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    records.append(r)
        except Exception as e:
            print(f"\033[91mError: Failed to read CSV timeline file: {e}\033[0m")
            return

        if not records:
            return

        print_banner("DEEP CHRONOLOGICAL TIMELINE SCANS (ANOMALIES)", "\033[94m")
        
        # Chronological anomaly detection parameters
        t_start = float(records[0]['timestamp'])
        
        # Scan A: Attention Dropouts (Gaze away from center/low eye contact for >2 seconds)
        dropout_events = []
        in_dropout = False
        dropout_start = 0.0
        
        for r in records:
            zone = r.get('attention_zone', 'CENTER')
            ec = float(r.get('eye_contact_score', 0.5))
            t_curr = float(r['timestamp']) - t_start
            
            if zone != 'CENTER' or ec < 0.35:
                if not in_dropout:
                    in_dropout = True
                    dropout_start = t_curr
            else:
                if in_dropout:
                    in_dropout = False
                    dur = t_curr - dropout_start
                    if dur >= 2.0:
                        dropout_events.append((dropout_start, t_curr, dur))
                        
        if in_dropout:
            dur = (float(records[-1]['timestamp']) - t_start) - dropout_start
            if dur >= 2.0:
                dropout_events.append((dropout_start, (float(records[-1]['timestamp']) - t_start), dur))

        print(f"\033[93m[!] ATTENTION DROPOUTS DETECTED ({len(dropout_events)} focus losses >2s)\033[0m")
        if not dropout_events:
            print("  No attention loss dropouts detected. Subject maintained outstanding eye contact.")
        else:
            for idx, (s, e, d) in enumerate(dropout_events):
                print(f"  Focus Loss #{idx+1}:  Start: {s:>5.1f}s | End: {e:>5.1f}s | Duration: {d:>4.1f}s (Looking away)")
        print("-" * 70)

        # Scan B: Cognitive/Stress Spikes (Stress index >0.50)
        stress_events = []
        in_stress = False
        stress_start = 0.0
        peak_stress = 0.0
        
        for r in records:
            st = float(r.get('stress_index', 0.0))
            t_curr = float(r['timestamp']) - t_start
            
            if st > 0.50:
                if not in_stress:
                    in_stress = True
                    stress_start = t_curr
                    peak_stress = st
                else:
                    peak_stress = max(peak_stress, st)
            else:
                if in_stress:
                    in_stress = False
                    dur = t_curr - stress_start
                    stress_events.append((stress_start, t_curr, dur, peak_stress))
                    
        if in_stress:
            dur = (float(records[-1]['timestamp']) - t_start) - stress_start
            stress_events.append((stress_start, (float(records[-1]['timestamp']) - t_start), dur, peak_stress))

        print(f"\033[93m[!] COGNITIVE STRESS SPIKES DETECTED ({len(stress_events)} events >0.50)\033[0m")
        if not stress_events:
            print("  No stress spikes detected. Subject remained completely calm throughout the assessment.")
        else:
            for idx, (s, e, d, p) in enumerate(stress_events):
                print(f"  Stress Spike #{idx+1}:  Start: {s:>5.1f}s | End: {e:>5.1f}s | Duration: {d:>4.1f}s | Peak: \033[91m{p:.3f}\033[0m")
        print("=" * 70 + "\n")

    else:
        print("\033[93mWarning: CSV Timeline file not found. Skipping chronological timeline scans.\033[0m")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\033[93mUsage: python mmbi/learning/reviewer.py <path_to_session_file>\033[0m")
        # Try to auto-locate most recent session file
        session_dir = os.path.join(os.path.dirname(__file__), 'data', 'sessions')
        if os.path.exists(session_dir):
            files = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.endswith('.json')]
            if files:
                newest = max(files, key=os.path.getmtime)
                print(f"\nAuto-loading most recent session: {os.path.basename(newest)}")
                analyze_session(newest)
                sys.exit(0)
        sys.exit(1)
        
    analyze_session(sys.argv[1])
