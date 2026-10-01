# importing libraries
import customtkinter
from customtkinter import CTkImage
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from tkinter import messagebox
import tkinter as tk
import json
import os
import traceback
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set appearance mode
customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("green")

import os
print("WORKING DIR:", os.getcwd())

# ============== ASPECT / RECOMMENDATION DATA ==============
aspects = {
    'account-management.account-access': "Managing Account Access",
    'company-brand.general-satisfaction': "Brand Satisfaction",
    'company-brand.competitor': "Brand Competitor Comparison",
    'company-brand.reviews': "Customer Reviews",
    'logistics-rides.speed': "Ride Speed",
    'online-experience.app-website': "App & Website Experience",
    'purchase-booking-experience.ease-of-use': "Ease of Purchasing/Booking Experience",
    'staff-support.attitude-of-staff': "Staff Attitude",
    'staff-support.phone': "Call Support",
    'staff-support.email': "Email Support",
    'value.price-value-for-money': "Prices",
    'value.discounts-promotion': "Discounts & Promotions"
}

recommendations = {
    "App & Website Experience": "Review and enhance the application's design and user interface.",
    "Ride Speed": "Improve system responsiveness and overall performance.",
    "Ease of Purchasing/Booking Experience": "Maintain the current user-friendly booking experience.",
    "Prices": "Review pricing strategy and customer value perception.",
    "Staff Attitude": "Continue staff training and customer service initiatives.",
    "Call Support": "Improve support response time and customer communication.",
    "Email Support": "Improve email response quality and turnaround time.",
    "Managing Account Access": "Maintain account accessibility and authentication reliability.",
    "Brand Satisfaction": "Continue maintaining overall customer satisfaction levels.",
    "Discounts & Promotions": "Review promotional offerings and customer incentives.",
    "Customer Reviews": "Monitor customer feedback trends and address recurring concerns.",
    "Brand Competitor Comparison": "Benchmark offerings against competitors regularly."
}

aspects_plain = list(aspects.keys())
sentiments = [-1, 0, 1]
total_labels = []
for aspect in aspects_plain:
    for sent in sentiments:
        total_labels.append(f"{aspect}.{sent}")

# Adjustable from the Settings page (Detection Sensitivity slider). Lowering
# this catches more aspects but raises the risk of false positives; it does
# NOT help with wording that doesn't closely match any trained aspect at all
# (e.g. a generic "services" mention with no matching label in `aspects`).
CONFIDENCE_THRESHOLD = 0.25

tokenizer = AutoTokenizer.from_pretrained("deberta_absa_model")
prediction_model = AutoModelForSequenceClassification.from_pretrained("deberta_absa_model")
prediction_model.eval()

# ============== COLOR PALETTE (beige & dark green, restored) ==============
COLORS = {
    # Sidebar (dark green, kept from layout v2, tuned to match beige/gold family)
    "sidebar_bg": "#2A4A3C",          # Dark green header/sidebar
    "sidebar_active": "#3D6650",      # Active nav item
    "sidebar_hover": "#35594A",       # Hover nav item
    "sidebar_text": "#D9D2BE",        # Beige-tinted light text on sidebar
    "sidebar_icon": "#C9A45C",        # Gold accent

    # Top bar / main background (beige family)
    "header_bg": "#F4F0E6",
    "header_border": "#D8D0BC",
    "main_bg": "#EFEAE0",             # Main background (beige)
    "card_bg": "#F4F0E6",             # Card background (lighter beige)
    "card_border": "#D8D0BC",
    "input_bg": "#FBF8F0",            # Input/inner field background (warm white)

    "text_primary": "#2E2A22",        # Near-black warm text
    "text_secondary": "#6B6354",      # Muted brown-gray
    "text_muted": "#8C8576",          # Lighter muted text
    "text_on_dark": "#F4F0E6",        # Beige text on dark green

    "accent_green": "#C9A45C",        # Gold/tan accent (primary buttons, to match v1)
    "accent_green_hover": "#D4B274",  # Lighter gold for hover

    "positive": "#3F7A5A",            # Green
    "positive_bg": "#E1EBE3",
    "negative": "#A6443A",            # Muted red
    "negative_bg": "#F3E0DC",
    "neutral": "#8C8576",
    "neutral_bg": "#EAE6DA",
    "mixed": "#B6914A",               # Amber/gold
    "mixed_bg": "#F0E6CE",
    "conf_bar_bg": "#D8D0BC",
}

FONT_SERIF = "Georgia"
FONT_SANS = "Segoe UI"
FONT_BOLD = "Segoe UI"

# Emoji faces for overall sentiment
SENTIMENT_EMOJI = {
    "POSITIVE": "😊",
    "NEGATIVE": "😞",
    "NEUTRAL":  "😐",
    "MIXED":    "😕",
}

SENTIMENT_COLOR = {
    "POSITIVE": COLORS["positive"],
    "NEGATIVE": COLORS["negative"],
    "NEUTRAL":  COLORS["neutral"],
    "MIXED":    COLORS["mixed"],
}

# ============== HISTORY PERSISTENCE ==============
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "absa_history.json")


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _json_safe_default(o):
    """Fallback converter passed to json.dump(). Catches numpy scalar/array
    types (float32, int64, ndarray, etc.) wherever they end up inside
    history_records and converts them to plain Python types, instead of
    json.dump() raising 'TypeError: Object of type float32 is not JSON
    serializable' and aborting the save (which previously also aborted
    get_predictions() before render_results() could run - that's why the
    UI got stuck on "Analyzing feedback...")."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_records, f, indent=2, default=_json_safe_default)
    except Exception as e:
        print(f"Could not save history: {e}")
        traceback.print_exc()


history_records = load_history()

# ============== STATE ==============
current_page = "dashboard"


# ============== PREDICTION ==============
def get_predictions():
    text = input_textbox.get("0.0", "end").strip()
    inputs = tokenizer(text, truncation=True, return_tensors="pt", padding=True, max_length=128)

    with torch.no_grad():
        outputs = prediction_model(**inputs)

    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    
    predictions, prediction_probs = [], []
    for label, prob in zip(total_labels, probs):
        if prob >= CONFIDENCE_THRESHOLD:
            predictions.append(label)
            # prob is numpy.float32 - cast to plain float so it stays JSON
            # serializable all the way into the saved history record.
            prediction_probs.append(float(prob))

    highest_confidence = max(prediction_probs) if prediction_probs else 0

    aspect_results = {}
    for prediction, prob in zip(predictions, prediction_probs):
        aspect, sentiment = prediction.rsplit(".", 1)
        if aspect not in aspect_results:
            aspect_results[aspect] = {}
        aspect_results[aspect][int(sentiment)] = float(prob)

    positive_count = negative_count = neutral_count = mixed_count = 0
    positive_aspects = []
    negative_aspects = []
    mixed_aspects = []
    neutral_aspects = []
    aspect_rows = []

    for aspect, sentiment_scores in aspect_results.items():
        display_aspect = aspects.get(aspect, aspect)
        pos = sentiment_scores.get(1, 0)
        neu = sentiment_scores.get(0, 0)
        neg = sentiment_scores.get(-1, 0)

        if pos > CONFIDENCE_THRESHOLD and neg > CONFIDENCE_THRESHOLD:
            mixed_count += 1
            mixed_aspects.append(display_aspect)
            aspect_rows.append({
                "name": display_aspect, "sentiment": "Mixed",
                "confidence": max(pos, neg), "kind": "mixed",
                "positive_pct": round(pos * 100, 1),
                "negative_pct": round(neg * 100, 1),
            })
        elif pos >= max(neu, neg):
            positive_count += 1
            positive_aspects.append(display_aspect)
            aspect_rows.append({
                "name": display_aspect, "sentiment": "Positive",
                "confidence": pos, "kind": "positive"
            })
        elif neg >= max(pos, neu):
            negative_count += 1
            negative_aspects.append(display_aspect)
            aspect_rows.append({
                "name": display_aspect, "sentiment": "Negative",
                "confidence": neg, "kind": "negative"
            })
        else:
            neutral_count += 1
            neutral_aspects.append(display_aspect)
            aspect_rows.append({
                "name": display_aspect, "sentiment": "Neutral",
                "confidence": neu, "kind": "neutral"
            })

    if positive_count > negative_count and mixed_count == 0:
        overall_sentiment = "POSITIVE"
    elif negative_count > positive_count and mixed_count == 0:
        overall_sentiment = "NEGATIVE"
    elif mixed_count > 0:
        overall_sentiment = "MIXED"
    else:
        overall_sentiment = "NEUTRAL"

    overall_confidence = round(highest_confidence, 2)

    # Build executive summary text
    parts = []
    if positive_aspects:
        parts.append(f"The customer expressed satisfaction with: {', '.join(positive_aspects)}.")
    if negative_aspects:
        parts.append(f"Areas of dissatisfaction include: {', '.join(negative_aspects)}.")
    if mixed_aspects:
        parts.append(f"Mixed feedback was noted for: {', '.join(mixed_aspects)}.")
    if neutral_aspects:
        parts.append(f"Neutral remarks were made about: {', '.join(neutral_aspects)}.")
    executive_summary = " ".join(parts) if parts else "No significant aspects were detected in the provided feedback."

    # Top recommendation
    top_rec = ""
    for aspect in (negative_aspects + mixed_aspects + positive_aspects):
        if aspect in recommendations:
            top_rec = recommendations[aspect]
            break

    # ---- Save this analysis into history ----
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "overall_sentiment": overall_sentiment,
        "overall_confidence": overall_confidence,
        "aspect_rows": aspect_rows,
    }
    history_records.insert(0, record)  # newest first
    save_history()

    render_results(
        overall_sentiment=overall_sentiment,
        overall_confidence=overall_confidence,
        aspect_rows=aspect_rows,
        executive_summary=executive_summary,
        top_recommendation=top_rec,
        low_confidence_note=highest_confidence < 0.60
    )

    refresh_dashboard_if_visible()
    refresh_history_if_visible()

    return predictions


# ============== SHARED UI HELPERS ==============

def make_card(parent, **kwargs):
    defaults = dict(
        fg_color=COLORS["card_bg"], corner_radius=14,
        border_width=1, border_color=COLORS["card_border"]
    )
    defaults.update(kwargs)
    return customtkinter.CTkFrame(parent, **defaults)


def make_confidence_bar(parent, color, fraction, width=90, height=8):
    outer = tk.Frame(parent, bg=COLORS["conf_bar_bg"], height=height, width=width)
    inner = tk.Frame(outer, bg=color, height=height, width=int(width * min(max(fraction, 0), 1.0)))
    inner.place(x=0, y=0)
    return outer


def sentiment_visuals(kind):
    if kind == "positive":
        return COLORS["positive"], COLORS["positive_bg"], "😊"
    elif kind == "negative":
        return COLORS["negative"], COLORS["negative_bg"], "😞"
    elif kind == "mixed":
        return COLORS["mixed"], COLORS["mixed_bg"], "😕"
    else:
        return COLORS["neutral"], COLORS["neutral_bg"], "😐"


def render_aspect_row(parent, row, index):
    kind = row["kind"]
    sent_label_color, sent_tag_bg, sent_icon = sentiment_visuals(kind)

    row_bg = "#EAE5D8" if index % 2 == 0 else COLORS["card_bg"]
    row_frame = customtkinter.CTkFrame(parent, fg_color=row_bg, corner_radius=4)
    row_frame.pack(fill="x", padx=20, pady=1)
    row_frame.columnconfigure(0, weight=5)
    row_frame.columnconfigure(1, weight=3)
    row_frame.columnconfigure(2, weight=3)

    # Aspect name
    name_cell = customtkinter.CTkFrame(row_frame, fg_color="transparent")
    name_cell.grid(row=0, column=0, sticky="w", padx=10, pady=8)
    customtkinter.CTkLabel(
        name_cell, text=row["name"],
        font=(FONT_SANS, 13), text_color=COLORS["text_primary"],
        anchor="w"
    ).pack(fill="x", anchor="w")

    # Sentiment tag
    sent_tag = customtkinter.CTkFrame(row_frame, fg_color=sent_tag_bg, corner_radius=12, height=26)
    sent_tag.grid(row=0, column=1, sticky="w", padx=6, pady=6)
    customtkinter.CTkLabel(
        sent_tag,
        text=f"{sent_icon}  {row['sentiment']}",
        font=(FONT_BOLD, 12, "bold"),
        text_color=sent_label_color,
        padx=10, pady=3
    ).pack()

    # Confidence cell: for Mixed, show both positive% and negative%.
    # For everything else, show the single confidence number + bar.
    conf_cell = customtkinter.CTkFrame(row_frame, fg_color="transparent")
    conf_cell.grid(row=0, column=2, sticky="w", padx=6, pady=6)

    if kind == "mixed":
        customtkinter.CTkLabel(
            conf_cell,
            text=f"↑ Positive: {row['positive_pct']}%   ↓ Negative: {row['negative_pct']}%",
            font=(FONT_SANS, 12), text_color=COLORS["text_secondary"]
        ).pack(side="left")
    else:
        conf_val = row["confidence"]
        customtkinter.CTkLabel(
            conf_cell, text=f"{conf_val:.2f}",
            font=(FONT_SANS, 12), text_color=COLORS["text_secondary"]
        ).pack(side="left", padx=(0, 6))
        bar = make_confidence_bar(conf_cell, sent_label_color, conf_val)
        bar.pack(side="left")


def render_aspect_confidence_chart(parent, aspect_rows):
    """Horizontal bar chart - one bar per detected aspect, colored by
    sentiment, height = confidence (or the stronger side for Mixed rows).
    Returns the embedded canvas widget, or None if there's nothing to draw."""
    if not aspect_rows:
        return None

    names, values, colors = [], [], []
    for r in aspect_rows:
        names.append(r["name"])
        if r["kind"] == "mixed":
            values.append(max(r["positive_pct"], r["negative_pct"]) / 100)
        else:
            values.append(r["confidence"])
        color, _, _ = sentiment_visuals(r["kind"])
        colors.append(color)

    fig = Figure(figsize=(6.4, max(1.6, 0.5 * len(names))), dpi=100)
    fig.patch.set_facecolor(COLORS["card_bg"])
    ax = fig.add_subplot(111)
    ax.set_facecolor(COLORS["card_bg"])

    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9, color=COLORS["text_primary"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence", fontsize=9, color=COLORS["text_secondary"])
    ax.invert_yaxis()
    ax.tick_params(colors=COLORS["text_secondary"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.configure(bg=COLORS["card_bg"])
    return widget


def render_sentiment_distribution_chart(parent, pos_count, neg_count, mixed_count, neu_count):
    """Donut chart summarizing overall sentiment across all saved history.
    Returns the embedded canvas widget, or None if there's no data yet."""
    labels_all = ["Positive", "Negative", "Mixed", "Neutral"]
    counts_all = [pos_count, neg_count, mixed_count, neu_count]
    colors_all = [COLORS["positive"], COLORS["negative"], COLORS["mixed"], COLORS["neutral"]]

    data = [(l, c, col) for l, c, col in zip(labels_all, counts_all, colors_all) if c > 0]
    if not data:
        return None
    labels_f, counts_f, colors_f = zip(*data)

    fig = Figure(figsize=(4.4, 3.6), dpi=100)
    fig.patch.set_facecolor(COLORS["card_bg"])
    ax = fig.add_subplot(111)
    wedges, texts, autotexts = ax.pie(
        counts_f, labels=labels_f, colors=colors_f, autopct="%1.0f%%",
        startangle=90, pctdistance=0.78, labeldistance=1.12,
        wedgeprops=dict(width=0.45, edgecolor=COLORS["card_bg"])
    )
    for t in texts:
        t.set_color(COLORS["text_primary"])
        t.set_fontsize(10)
    for at in autotexts:
        at.set_color("#FFFFFF")
        at.set_fontsize(9)
        at.set_fontweight("bold")
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.configure(bg=COLORS["card_bg"])
    return widget


# ============== ANALYZE REVIEW PAGE RENDERING ==============

def clear_results_area():
    for w in results_scroll.winfo_children():
        w.destroy()


def render_placeholder_results(msg):
    clear_results_area()
    lbl = customtkinter.CTkLabel(
        results_scroll, text=msg,
        font=(FONT_SANS, 14), text_color=COLORS["text_muted"], anchor="center"
    )
    lbl.pack(expand=True, pady=60)


def render_results(overall_sentiment, overall_confidence, aspect_rows,
                   executive_summary, top_recommendation, low_confidence_note):
    clear_results_area()

    emoji = SENTIMENT_EMOJI.get(overall_sentiment, "😐")
    sent_color = SENTIMENT_COLOR.get(overall_sentiment, COLORS["neutral"])

    # ---- Top row: Overall Sentiment (left) + Detected Aspects Table (right) ----
    top_row = customtkinter.CTkFrame(results_scroll, fg_color="transparent")
    top_row.pack(fill="x", padx=0, pady=(0, 16))
    top_row.columnconfigure(0, weight=1)
    top_row.columnconfigure(1, weight=3)

    overall_card = make_card(top_row)
    overall_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

    customtkinter.CTkLabel(
        overall_card, text="Overall Sentiment",
        font=(FONT_BOLD, 13, "bold"), text_color=COLORS["text_secondary"]
    ).pack(pady=(18, 4))

    customtkinter.CTkLabel(
        overall_card, text=emoji,
        font=("Segoe UI Emoji", 54), text_color=sent_color
    ).pack(pady=(8, 4))

    customtkinter.CTkLabel(
        overall_card, text=overall_sentiment.capitalize(),
        font=(FONT_BOLD, 18, "bold"), text_color=sent_color
    ).pack()

    customtkinter.CTkLabel(
        overall_card, text=f"Confidence: {overall_confidence:.2f}",
        font=(FONT_SANS, 12), text_color=COLORS["text_muted"]
    ).pack(pady=(2, 6))

    conf_bar_outer = make_confidence_bar(overall_card, sent_color, overall_confidence, width=120, height=6)
    conf_bar_outer.pack(pady=(2, 18))

    if low_confidence_note:
        customtkinter.CTkLabel(
            overall_card, text="ℹ️ Low confidence",
            font=(FONT_SANS, 11, "italic"), text_color=COLORS["text_muted"]
        ).pack(pady=(0, 10))

    # Detected Aspects & Sentiments table card
    aspects_card = make_card(top_row)
    aspects_card.grid(row=0, column=1, sticky="nsew")

    customtkinter.CTkLabel(
        aspects_card, text="Detected Aspects & Sentiments",
        font=(FONT_BOLD, 13, "bold"), text_color=COLORS["text_primary"], anchor="w"
    ).pack(fill="x", padx=20, pady=(18, 10))

    header_row = customtkinter.CTkFrame(aspects_card, fg_color=COLORS["neutral_bg"], corner_radius=6)
    header_row.pack(fill="x", padx=20, pady=(0, 6))
    header_row.columnconfigure(0, weight=5)
    header_row.columnconfigure(1, weight=3)
    header_row.columnconfigure(2, weight=3)

    for col_idx, col_text in enumerate(["Aspect", "Sentiment", "Confidence"]):
        customtkinter.CTkLabel(
            header_row, text=col_text,
            font=(FONT_BOLD, 12, "bold"), text_color=COLORS["text_secondary"],
            anchor="w"
        ).grid(row=0, column=col_idx, sticky="w", padx=10, pady=6)

    if not aspect_rows:
        customtkinter.CTkLabel(
            aspects_card, text="No aspects detected.",
            font=(FONT_SANS, 13), text_color=COLORS["text_muted"]
        ).pack(pady=20)
    else:
        for i, row in enumerate(aspect_rows):
            render_aspect_row(aspects_card, row, i)
        customtkinter.CTkLabel(aspects_card, text="", height=10).pack()

    # ---- Aspect confidence chart ----
    chart_card = make_card(results_scroll)
    chart_card.pack(fill="x", pady=(0, 16))

    customtkinter.CTkLabel(
        chart_card, text="📈  Aspect Confidence Overview",
        font=(FONT_BOLD, 14, "bold"), text_color=COLORS["text_primary"], anchor="w"
    ).pack(fill="x", padx=20, pady=(18, 6))

    chart_widget = render_aspect_confidence_chart(chart_card, aspect_rows)
    if chart_widget:
        chart_widget.pack(padx=20, pady=(0, 18))
    else:
        customtkinter.CTkLabel(
            chart_card, text="No aspects detected to chart.",
            font=(FONT_SANS, 13), text_color=COLORS["text_muted"]
        ).pack(padx=20, pady=(0, 18), anchor="w")

    # ---- Bottom row: Executive Summary (left) + Recommendation (right) ----
    bottom_row = customtkinter.CTkFrame(results_scroll, fg_color="transparent")
    bottom_row.pack(fill="x", padx=0, pady=(0, 20))
    bottom_row.columnconfigure(0, weight=1)
    bottom_row.columnconfigure(1, weight=1)

    summary_card = make_card(bottom_row)
    summary_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

    customtkinter.CTkLabel(
        summary_card, text="Executive Summary",
        font=(FONT_BOLD, 15, "bold"), text_color=COLORS["text_primary"], anchor="w"
    ).pack(fill="x", padx=20, pady=(18, 8))

    customtkinter.CTkLabel(
        summary_card, text=executive_summary,
        font=(FONT_SANS, 16), text_color=COLORS["text_secondary"],
        anchor="w", justify="left", wraplength=420
    ).pack(fill="x", padx=20, pady=(0, 20))

    rec_card = make_card(bottom_row)
    rec_card.grid(row=0, column=1, sticky="nsew")

    customtkinter.CTkLabel(
        rec_card, text="Recommendation",
        font=(FONT_BOLD, 15, "bold"), text_color=COLORS["text_primary"], anchor="w"
    ).pack(fill="x", padx=20, pady=(18, 8))

    customtkinter.CTkLabel(
        rec_card,
        text=top_recommendation if top_recommendation else "No specific recommendation at this time.",
        font=(FONT_SANS, 16), text_color=COLORS["text_secondary"],
        anchor="w", justify="left", wraplength=420
    ).pack(fill="x", padx=20, pady=(0, 20))

    customtkinter.CTkLabel(
        rec_card, text="🎯",
        font=("Segoe UI Emoji", 28)
    ).pack(anchor="e", padx=20, pady=(0, 16))


# ============== DASHBOARD PAGE ==============

def clear_dashboard_area():
    for w in dashboard_scroll.winfo_children():
        w.destroy()


def render_dashboard():
    clear_dashboard_area()

    total = len(history_records)
    pos_count = sum(1 for r in history_records if r["overall_sentiment"] == "POSITIVE")
    neg_count = sum(1 for r in history_records if r["overall_sentiment"] == "NEGATIVE")
    mixed_count = sum(1 for r in history_records if r["overall_sentiment"] == "MIXED")
    neu_count = sum(1 for r in history_records if r["overall_sentiment"] == "NEUTRAL")

    # ---- Stat cards row ----
    stats_row = customtkinter.CTkFrame(dashboard_scroll, fg_color="transparent")
    stats_row.pack(fill="x", pady=(0, 16))
    for i in range(5):
        stats_row.columnconfigure(i, weight=1)

    stat_defs = [
        ("Total Reviews", total, COLORS["text_primary"], "📊"),
        ("Positive", pos_count, COLORS["positive"], "😊"),
        ("Negative", neg_count, COLORS["negative"], "😞"),
        ("Mixed", mixed_count, COLORS["mixed"], "😕"),
        ("Neutral", neu_count, COLORS["neutral"], "😐"),
    ]

    for i, (label, value, color, icon) in enumerate(stat_defs):
        card = make_card(stats_row)
        card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
        customtkinter.CTkLabel(
            card, text=icon, font=("Segoe UI Emoji", 22)
        ).pack(pady=(16, 2))
        customtkinter.CTkLabel(
            card, text=str(value), font=(FONT_BOLD, 22, "bold"), text_color=color
        ).pack()
        customtkinter.CTkLabel(
            card, text=label, font=(FONT_SANS, 12), text_color=COLORS["text_secondary"]
        ).pack(pady=(0, 16))

    # ---- Sentiment distribution chart ----
    chart_card = make_card(dashboard_scroll)
    chart_card.pack(fill="x", pady=(0, 16))

    customtkinter.CTkLabel(
        chart_card, text="🍩  Sentiment Distribution",
        font=(FONT_BOLD, 14, "bold"), text_color=COLORS["text_primary"], anchor="w"
    ).pack(fill="x", padx=20, pady=(18, 6))

    pie_widget = render_sentiment_distribution_chart(chart_card, pos_count, neg_count, mixed_count, neu_count)
    if pie_widget:
        pie_widget.pack(padx=20, pady=(0, 18))
    else:
        customtkinter.CTkLabel(
            chart_card, text="No reviews analyzed yet - chart will appear once you have some history.",
            font=(FONT_SANS, 13), text_color=COLORS["text_muted"]
        ).pack(padx=20, pady=(0, 18), anchor="w")

    # ---- Recent activity card ----
    recent_card = make_card(dashboard_scroll)
    recent_card.pack(fill="both", expand=True, pady=(0, 10))

    customtkinter.CTkLabel(
        recent_card, text="Recent Activity",
        font=(FONT_BOLD, 14, "bold"), text_color=COLORS["text_primary"], anchor="w"
    ).pack(fill="x", padx=20, pady=(18, 10))

    if not history_records:
        customtkinter.CTkLabel(
            recent_card,
            text="No reviews analyzed yet. Head to \"Analyze Review\" to get started.",
            font=(FONT_SANS, 13), text_color=COLORS["text_muted"]
        ).pack(padx=20, pady=(0, 20), anchor="w")
    else:
        for i, record in enumerate(history_records[:6]):
            sent = record["overall_sentiment"]
            color = SENTIMENT_COLOR.get(sent, COLORS["neutral"])
            emoji = SENTIMENT_EMOJI.get(sent, "😐")
            row_bg = "#EAE5D8" if i % 2 == 0 else COLORS["card_bg"]
            row_frame = customtkinter.CTkFrame(recent_card, fg_color=row_bg, corner_radius=6)
            row_frame.pack(fill="x", padx=20, pady=3)

            snippet = record["text"][:70] + ("…" if len(record["text"]) > 70 else "")
            customtkinter.CTkLabel(
                row_frame, text=f"{emoji}  {snippet}",
                font=(FONT_SANS, 13), text_color=COLORS["text_primary"],
                anchor="w"
            ).pack(side="left", fill="x", expand=True, padx=12, pady=10)

            customtkinter.CTkLabel(
                row_frame, text=record["timestamp"],
                font=(FONT_SANS, 11), text_color=COLORS["text_muted"]
            ).pack(side="right", padx=12)

        customtkinter.CTkLabel(recent_card, text="", height=8).pack()


def refresh_dashboard_if_visible():
    if current_page == "dashboard":
        render_dashboard()


# ============== HISTORY PAGE ==============

def clear_history_area():
    for w in history_scroll.winfo_children():
        w.destroy()


def render_history():
    clear_history_area()

    if not history_records:
        customtkinter.CTkLabel(
            history_scroll,
            text="No feedback history yet. Analyzed reviews will appear here.",
            font=(FONT_SANS, 14), text_color=COLORS["text_muted"]
        ).pack(expand=True, pady=60)
        return

    for record in history_records:
        sent = record["overall_sentiment"]
        color = SENTIMENT_COLOR.get(sent, COLORS["neutral"])
        emoji = SENTIMENT_EMOJI.get(sent, "😐")

        card = make_card(history_scroll)
        card.pack(fill="x", pady=(0, 12))

        top_line = customtkinter.CTkFrame(card, fg_color="transparent")
        top_line.pack(fill="x", padx=18, pady=(14, 4))

        customtkinter.CTkLabel(
            top_line, text=f"{emoji}  {sent.capitalize()}",
            font=(FONT_BOLD, 14, "bold"), text_color=color, anchor="w"
        ).pack(side="left")

        customtkinter.CTkLabel(
            top_line, text=f"Confidence: {record['overall_confidence']:.2f}    •    {record['timestamp']}",
            font=(FONT_SANS, 12), text_color=COLORS["text_muted"]
        ).pack(side="right")

        customtkinter.CTkLabel(
            card, text=record["text"],
            font=(FONT_SANS, 13), text_color=COLORS["text_primary"],
            anchor="w", justify="left", wraplength=950
        ).pack(fill="x", padx=18, pady=(0, 10))

        aspects_found = [r["name"] for r in record.get("aspect_rows", [])]
        if aspects_found:
            customtkinter.CTkLabel(
                card, text="Aspects: " + ", ".join(aspects_found),
                font=(FONT_SANS, 12, "italic"), text_color=COLORS["text_secondary"],
                anchor="w", justify="left", wraplength=950
            ).pack(fill="x", padx=18, pady=(0, 16))
        else:
            customtkinter.CTkLabel(card, text="", height=6).pack()


def refresh_history_if_visible():
    if current_page == "history":
        render_history()


def clear_all_history():
    if not history_records:
        return
    confirm = messagebox.askyesno(
        "Clear History",
        "This will permanently delete all saved feedback history. Continue?"
    )
    if confirm:
        history_records.clear()
        save_history()
        render_history()
        refresh_dashboard_if_visible()


# ============== SETTINGS PAGE ==============

def render_settings():
    for w in settings_scroll.winfo_children():
        w.destroy()

    card = make_card(settings_scroll)
    card.pack(fill="x", pady=(0, 16))

    customtkinter.CTkLabel(
        card, text="Data & Storage",
        font=(FONT_BOLD, 15, "bold"), text_color=COLORS["text_primary"], anchor="w"
    ).pack(fill="x", padx=20, pady=(18, 6))

    customtkinter.CTkLabel(
        card, text=f"History is saved locally to:\n{HISTORY_FILE}",
        font=(FONT_SANS, 12), text_color=COLORS["text_secondary"],
        anchor="w", justify="left", wraplength=850
    ).pack(fill="x", padx=20, pady=(0, 14))

    customtkinter.CTkButton(
        card, text="🗑  Clear All History",
        font=(FONT_BOLD, 13, "bold"),
        fg_color=COLORS["negative"], hover_color="#8C3A31",
        text_color="#FFFFFF", corner_radius=8, height=38, width=180,
        command=clear_all_history
    ).pack(anchor="w", padx=20, pady=(0, 20))

    # ---- Detection sensitivity card ----
    sens_card = make_card(settings_scroll)
    sens_card.pack(fill="x", pady=(0, 16))

    customtkinter.CTkLabel(
        sens_card, text="Detection Sensitivity",
        font=(FONT_BOLD, 15, "bold"), text_color=COLORS["text_primary"], anchor="w"
    ).pack(fill="x", padx=20, pady=(18, 6))

    customtkinter.CTkLabel(
        sens_card,
        text=("Lower this to catch more aspects, at the cost of more false "
              "positives. Note: wording that doesn't closely match any of "
              "the model's trained aspects (e.g. a generic \"services\" "
              "mention with no direct equivalent below) may still go "
              "undetected even at a low threshold - that's a model "
              "coverage limit this slider can't fully fix."),
        font=(FONT_SANS, 12), text_color=COLORS["text_secondary"],
        anchor="w", justify="left", wraplength=850
    ).pack(fill="x", padx=20, pady=(0, 12))

    slider_row = customtkinter.CTkFrame(sens_card, fg_color="transparent")
    slider_row.pack(fill="x", padx=20, pady=(0, 20))

    threshold_value_label = customtkinter.CTkLabel(
        slider_row, text=f"{CONFIDENCE_THRESHOLD:.2f}",
        font=(FONT_BOLD, 13, "bold"), text_color=COLORS["accent_green"], width=50
    )
    threshold_value_label.pack(side="right")

    def on_threshold_change(val):
        global CONFIDENCE_THRESHOLD
        CONFIDENCE_THRESHOLD = round(float(val), 2)
        threshold_value_label.configure(text=f"{CONFIDENCE_THRESHOLD:.2f}")

    slider = customtkinter.CTkSlider(
        slider_row, from_=0.10, to=0.50, number_of_steps=40,
        button_color=COLORS["accent_green"], progress_color=COLORS["accent_green"],
        command=on_threshold_change
    )
    slider.set(CONFIDENCE_THRESHOLD)
    slider.pack(side="left", fill="x", expand=True, padx=(0, 12))

    card2 = make_card(settings_scroll)
    card2.pack(fill="x")

    customtkinter.CTkLabel(
        card2, text="About this Build",
        font=(FONT_BOLD, 15, "bold"), text_color=COLORS["text_primary"], anchor="w"
    ).pack(fill="x", padx=20, pady=(18, 6))

    customtkinter.CTkLabel(
        card2,
        text="Model: DeBERTa (local) for aspect-based sentiment classification.\n"
             "Confidence threshold: adjustable above, in Detection Sensitivity.\n"
             "Theme: Beige & Dark Green.",
        font=(FONT_SANS, 12), text_color=COLORS["text_secondary"],
        anchor="w", justify="left"
    ).pack(fill="x", padx=20, pady=(0, 20))


# ============== ABOUT PAGE ==============

def render_about():
    for w in about_scroll.winfo_children():
        w.destroy()

    card = make_card(about_scroll)
    card.pack(fill="x")

    customtkinter.CTkLabel(
        card, text="🧠  ABSA Analyzer",
        font=(FONT_SERIF, 22, "bold"), text_color=COLORS["text_primary"]
    ).pack(padx=20, pady=(24, 4), anchor="w")

    customtkinter.CTkLabel(
        card,
        text="Aspect-Based Sentiment Analysis Platform, powered by a local DeBERTa model.",
        font=(FONT_SANS, 14), text_color=COLORS["text_secondary"],
        anchor="w", justify="left", wraplength=850
    ).pack(padx=20, pady=(0, 16), anchor="w")

    customtkinter.CTkLabel(
        card,
        text=(
            "This tool analyzes customer feedback text and identifies which specific "
            "aspects of the product or service are being discussed (e.g. staff attitude, "
            "pricing, app experience), along with the sentiment expressed toward each one. "
            "Results, including mixed sentiment where both praise and criticism are present "
            "for the same aspect, are saved to History for later review."
        ),
        font=(FONT_SANS, 13), text_color=COLORS["text_secondary"],
        anchor="w", justify="left", wraplength=850
    ).pack(padx=20, pady=(0, 20), anchor="w")

    customtkinter.CTkLabel(
        card, text="DeBERTa ABSA  •  Multi-label Classification  •  v2.2",
        font=(FONT_SANS, 11), text_color=COLORS["text_muted"]
    ).pack(padx=20, pady=(0, 24), anchor="w")


# ============== BUTTON CALLBACKS ==============

def button_callback():
    text_input = input_textbox.get("0.0", "end").strip()
    if text_input == "":
        render_placeholder_results("⚠️   Please enter customer feedback to analyze.")
        return
    if len(text_input.split()) < 4:
        messagebox.showwarning(
            "Insufficient Feedback",
            "Your review is too short for reliable analysis.\n\nPlease provide more detailed feedback."
        )
        return
    render_placeholder_results("⏳   Analyzing feedback...")
    app.update()
    try:
        get_predictions()
    except Exception as e:
        # Defensive net: if anything ever goes wrong inside get_predictions
        # again, show it instead of leaving the UI stuck on "Analyzing...".
        traceback.print_exc()
        render_placeholder_results(f"⚠️   Something went wrong while analyzing: {e}")


def new_analysis():
    """Clears the input box and resets the results panel, then takes the
    user to the Analyze Review page so they can start a fresh review.
    Nothing is lost - any prior analysis is already saved in History."""
    input_textbox.delete("0.0", "end")
    render_placeholder_results("Results will appear here once you analyze your feedback.")
    nav_click("analyze")


# ============== PAGE SWITCHING ==============

PAGE_FRAMES = {}  # populated after frames are created


def show_page(page_id):
    for key, frame in PAGE_FRAMES.items():
        if key == page_id:
            frame.grid(row=0, column=0, sticky="nsew")
        else:
            frame.grid_remove()

    if page_id == "dashboard":
        render_dashboard()
    elif page_id == "history":
        render_history()
    elif page_id == "settings":
        render_settings()
    elif page_id == "about":
        render_about()
    # "analyze" page keeps whatever is currently in results_scroll


def nav_click(page):
    global current_page
    current_page = page
    for key, btn in nav_buttons.items():
        if key == page:
            btn.configure(fg_color=COLORS["sidebar_active"], text_color="#FFFFFF")
        else:
            btn.configure(fg_color="transparent", text_color=COLORS["sidebar_text"])
    show_page(page)


# ============== MAIN APP ==============
app = customtkinter.CTk()
app.configure(fg_color=COLORS["main_bg"])
app.geometry("1200x820")
app.title("ABSA Analyzer")
app.minsize(960, 700)

# ============== ROOT LAYOUT: sidebar + main ==============
root_frame = customtkinter.CTkFrame(app, fg_color="transparent")
root_frame.pack(fill="both", expand=True)
root_frame.columnconfigure(1, weight=1)
root_frame.rowconfigure(0, weight=1)

# ============== SIDEBAR ==============
sidebar = customtkinter.CTkFrame(root_frame, fg_color=COLORS["sidebar_bg"], corner_radius=0, width=220)
sidebar.grid(row=0, column=0, sticky="nsew")
sidebar.pack_propagate(False)

brand_frame = customtkinter.CTkFrame(sidebar, fg_color="transparent")
brand_frame.pack(fill="x", padx=20, pady=(28, 24))

customtkinter.CTkLabel(
    brand_frame, text="🧠  ABSA",
    font=(FONT_SERIF, 20, "bold"), text_color="#FFFFFF", anchor="w"
).pack(fill="x")
customtkinter.CTkLabel(
    brand_frame, text="Analyzer",
    font=(FONT_SANS, 13), text_color=COLORS["sidebar_text"], anchor="w"
).pack(fill="x")

sep = tk.Frame(sidebar, bg="#3D6650", height=1)
sep.pack(fill="x", padx=16, pady=(0, 16))

nav_items = [
    ("dashboard", "📊  Dashboard"),
    ("analyze",   "🔍  Analyze Review"),
    ("history",   "📋  History"),
    ("settings",  "⚙️   Settings"),
    ("about",     "ℹ️    About"),
]

nav_buttons = {}
for page_id, label in nav_items:
    is_active = page_id == current_page
    btn = customtkinter.CTkButton(
        sidebar, text=label,
        font=(FONT_SANS, 14),
        fg_color=COLORS["sidebar_active"] if is_active else "transparent",
        text_color="#FFFFFF" if is_active else COLORS["sidebar_text"],
        hover_color=COLORS["sidebar_hover"],
        anchor="w", corner_radius=8, height=42,
        command=lambda p=page_id: nav_click_wrapper(p)
    )
    btn.pack(fill="x", padx=12, pady=2)
    nav_buttons[page_id] = btn

# ============== MAIN PANEL ==============
main_panel = customtkinter.CTkFrame(root_frame, fg_color=COLORS["main_bg"], corner_radius=0)
main_panel.grid(row=0, column=1, sticky="nsew")
main_panel.columnconfigure(0, weight=1)
main_panel.rowconfigure(1, weight=1)

# ---- Top bar ----
topbar = customtkinter.CTkFrame(
    main_panel, fg_color=COLORS["header_bg"], corner_radius=0, height=64,
    border_width=0
)
topbar.grid(row=0, column=0, sticky="ew")
topbar.pack_propagate(False)

topbar_inner = customtkinter.CTkFrame(topbar, fg_color="transparent")
topbar_inner.pack(fill="both", expand=True, padx=28)

title_row = customtkinter.CTkFrame(topbar_inner, fg_color="transparent")
title_row.pack(side="left", fill="y")

PAGE_TITLES = {
    "dashboard": "📊  DASHBOARD",
    "analyze": "🔍  ANALYZE REVIEW",
    "history": "📋  HISTORY",
    "settings": "⚙️  SETTINGS",
    "about": "ℹ️  ABOUT",
}

page_title_label = customtkinter.CTkLabel(
    title_row, text=PAGE_TITLES["dashboard"],
    font=(FONT_SERIF, 18, "bold"), text_color=COLORS["text_primary"]
)
page_title_label.pack(side="left", pady=18)

new_analysis_btn = customtkinter.CTkButton(
    topbar_inner, text="New Analysis",
    font=(FONT_BOLD, 13, "bold"),
    fg_color=COLORS["accent_green"],
    hover_color=COLORS["accent_green_hover"],
    text_color=COLORS["text_on_dark"],
    corner_radius=8, height=36, width=130,
    command=lambda: None  # replaced below with new_analysis_wrapper
)
new_analysis_btn.pack(side="right", pady=14)


def nav_click_wrapper(page):
    """Wraps nav_click to also update the topbar title."""
    nav_click(page)
    page_title_label.configure(text=PAGE_TITLES.get(page, ""))


def new_analysis_wrapper():
    new_analysis()
    page_title_label.configure(text=PAGE_TITLES["analyze"])


new_analysis_btn.configure(command=new_analysis_wrapper)

# ---- Content container that holds all pages stacked via grid ----
content_container = customtkinter.CTkFrame(main_panel, fg_color="transparent")
content_container.grid(row=1, column=0, sticky="nsew", padx=24, pady=20)
content_container.columnconfigure(0, weight=1)
content_container.rowconfigure(0, weight=1)

# ============== PAGE: DASHBOARD ==============
dashboard_frame = customtkinter.CTkFrame(content_container, fg_color="transparent")
dashboard_scroll = customtkinter.CTkScrollableFrame(dashboard_frame, fg_color="transparent")
dashboard_scroll.pack(fill="both", expand=True)
PAGE_FRAMES["dashboard"] = dashboard_frame

# ============== PAGE: ANALYZE REVIEW ==============
analyze_frame = customtkinter.CTkFrame(content_container, fg_color="transparent")
analyze_frame.columnconfigure(0, weight=1)
analyze_frame.rowconfigure(1, weight=1)
PAGE_FRAMES["analyze"] = analyze_frame

input_card = make_card(analyze_frame)
input_card.grid(row=0, column=0, sticky="ew", pady=(0, 16))

input_header = customtkinter.CTkFrame(input_card, fg_color="transparent")
input_header.pack(fill="x", padx=20, pady=(16, 8))

customtkinter.CTkLabel(
    input_header, text="📝  Enter Customer Feedback",
    font=(FONT_SERIF, 14, "bold"), text_color=COLORS["text_primary"], anchor="w"
).pack(side="left")

customtkinter.CTkLabel(
    input_header, text="Paste or type a review",
    font=(FONT_SANS, 12), text_color=COLORS["text_muted"], anchor="e"
).pack(side="right")

input_textbox = customtkinter.CTkTextbox(
    input_card, height=100, corner_radius=10,
    font=(FONT_SANS, 14), fg_color=COLORS["input_bg"],
    text_color=COLORS["text_primary"],
    border_width=1, border_color=COLORS["card_border"], wrap="word"
)
input_textbox.pack(fill="x", padx=20, pady=(0, 12))

btn_row = customtkinter.CTkFrame(input_card, fg_color="transparent")
btn_row.pack(fill="x", padx=20, pady=(0, 16))

analyze_btn = customtkinter.CTkButton(
    btn_row, text="⚡  Analyze Sentiment",
    font=(FONT_SERIF, 14, "bold"),
    fg_color=COLORS["accent_green"], hover_color=COLORS["accent_green_hover"],
    text_color=COLORS["text_on_dark"], corner_radius=8, height=42, width=200,
    command=button_callback
)
analyze_btn.pack(side="left")

clear_btn = customtkinter.CTkButton(
    btn_row, text="Clear",
    font=(FONT_SANS, 13),
    fg_color="transparent", hover_color=COLORS["input_bg"],
    text_color=COLORS["text_secondary"],
    border_width=1, border_color=COLORS["card_border"],
    corner_radius=8, height=42, width=90,
    command=lambda: new_analysis_wrapper()
)
clear_btn.pack(side="left", padx=(12, 0))

results_outer = make_card(analyze_frame)
results_outer.grid(row=1, column=0, sticky="nsew")

results_header_row = customtkinter.CTkFrame(results_outer, fg_color="transparent")
results_header_row.pack(fill="x", padx=20, pady=(16, 4))

customtkinter.CTkLabel(
    results_header_row, text="Analysis Results",
    font=(FONT_SERIF, 15, "bold"), text_color=COLORS["text_primary"], anchor="w"
).pack(side="left")

results_scroll = customtkinter.CTkScrollableFrame(
    results_outer, fg_color="transparent", corner_radius=0
)
results_scroll.pack(fill="both", expand=True, padx=16, pady=(4, 16))

render_placeholder_results("Results will appear here once you analyze your feedback.")

# ============== PAGE: HISTORY ==============
history_frame = customtkinter.CTkFrame(content_container, fg_color="transparent")
history_frame.columnconfigure(0, weight=1)
history_frame.rowconfigure(0, weight=1)
PAGE_FRAMES["history"] = history_frame

history_scroll = customtkinter.CTkScrollableFrame(history_frame, fg_color="transparent")
history_scroll.grid(row=0, column=0, sticky="nsew")

# ============== PAGE: SETTINGS ==============
settings_frame = customtkinter.CTkFrame(content_container, fg_color="transparent")
settings_frame.columnconfigure(0, weight=1)
settings_frame.rowconfigure(0, weight=1)
PAGE_FRAMES["settings"] = settings_frame

settings_scroll = customtkinter.CTkScrollableFrame(settings_frame, fg_color="transparent")
settings_scroll.grid(row=0, column=0, sticky="nsew")

# ============== PAGE: ABOUT ==============
about_frame = customtkinter.CTkFrame(content_container, fg_color="transparent")
about_frame.columnconfigure(0, weight=1)
about_frame.rowconfigure(0, weight=1)
PAGE_FRAMES["about"] = about_frame

about_scroll = customtkinter.CTkScrollableFrame(about_frame, fg_color="transparent")
about_scroll.grid(row=0, column=0, sticky="nsew")

# Show the initial page
show_page(current_page)

# ============== FOOTER ==============
footer = customtkinter.CTkFrame(main_panel, fg_color="transparent", height=36)
footer.grid(row=2, column=0, sticky="ew")
customtkinter.CTkLabel(
    footer, text="DeBERTa ABSA  •  Multi-label Classification  •  v2.2",
    font=(FONT_SANS, 11), text_color=COLORS["text_muted"]
).pack()

app.mainloop()
