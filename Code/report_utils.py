from __future__ import annotations

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

import pandas as pd

from config import get_plot_path, PLOT_NAMES

PAGE_W = A4[0] - 32 * mm   # usable width with 16 mm margins each side


# ── Table helpers ─────────────────────────────────────────────────────────────

def _make_table(data, col_widths=None, header_fill="#111827"):
    table = Table(data, colWidths=col_widths, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1,  0), colors.HexColor(header_fill)),
        ("TEXTCOLOR",    (0, 0), (-1,  0), colors.white),
        ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("LEADING",      (0, 0), (-1, -1), 11),
        ("GRID",         (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    return table


def _info_card(body_style, label: str, value: str, value_size: int = 13):
    table = Table(
        [[
            Paragraph(f"<font color='#64748B'><b>{label}</b></font>", body_style),
            Paragraph(f"<font size='{value_size}' color='#111827'><b>{value}</b></font>", body_style),
        ]],
        colWidths=[PAGE_W / 3 * 0.35, PAGE_W / 3 * 0.65],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.white),
        ("BOX",           (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _embed_plot(path: str, caption: str, max_width: float = None) -> list:
    """Return a list of flowables: image + caption, or a warning if missing."""
    if not os.path.exists(path):
        return [Paragraph(f"<i>[Plot not found: {os.path.basename(path)}]</i>",
                          getSampleStyleSheet()["Normal"])]
    img = Image(path)
    if max_width is None:
        max_width = PAGE_W
    # Scale proportionally to fit max width
    scale = max_width / img.imageWidth
    img.drawWidth  = img.imageWidth  * scale
    img.drawHeight = img.imageHeight * scale

    cap_style = ParagraphStyle(
        "Caption", fontName="Helvetica-Oblique", fontSize=8,
        textColor=colors.HexColor("#6B7280"), alignment=TA_CENTER, spaceAfter=6,
    )
    return [img, Paragraph(caption, cap_style), Spacer(1, 3 * mm)]


# ── Main report builder ───────────────────────────────────────────────────────

def save_pdf_report(
    csv_path: str,
    args,
    input_shape,
    class_names,
    binary_res: dict,
    multi_res: dict,
    elapsed_seconds: float,
    outputs: list,
    out_path: str,
):
    dataset_name = os.path.basename(csv_path)
    styles       = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        alignment=TA_CENTER, textColor=colors.white, spaceAfter=0,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10, leading=13,
        alignment=TA_CENTER, textColor=colors.white, spaceAfter=0,
    )
    section_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=13, leading=16,
        textColor=colors.HexColor("#111827"), spaceBefore=10, spaceAfter=6,
    )
    note_style = ParagraphStyle(
        "NoteText", parent=styles["BodyText"],
        fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
        textColor=colors.HexColor("#64748B"), alignment=TA_CENTER,
    )
    body_style = ParagraphStyle(
        "BodyTextClean", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=9.5, leading=12,
        textColor=colors.HexColor("#1F2937"),
    )

    accent       = colors.HexColor("#0F4C81")
    accent_light = colors.HexColor("#EAF2FB")
    slate        = colors.HexColor("#334155")
    paper        = colors.HexColor("#F8FAFC")

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=16 * mm,   bottomMargin=16 * mm,
        title="Predictive Fault Detection in Robots Using Deep Learning",
        author="Asjal Abdullah",
    )

    story = []

    # ── Banner ────────────────────────────────────────────────────────────────
    banner = Table(
        [[
            Paragraph("Predictive Fault Detection Report", title_style),
            Paragraph("LSTM-based anomaly detection and failure classification", subtitle_style),
            Paragraph(f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}", note_style),
        ]],
        colWidths=[PAGE_W * 0.47, PAGE_W * 0.33, PAGE_W * 0.20],
        hAlign="CENTER",
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), accent),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.white),
        ("BOX",           (0, 0), (-1, -1), 0, accent),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(banner)
    story.append(Spacer(1, 4 * mm))

    # ── Snapshot cards ────────────────────────────────────────────────────────
    snapshot = Table(
        [[
            _info_card(body_style, "Dataset",     dataset_name),
            _info_card(body_style, "Input Shape", str(input_shape)),
            _info_card(body_style, "Classes",     str(len(class_names))),
        ]],
        colWidths=[PAGE_W / 3, PAGE_W / 3, PAGE_W / 3],
        hAlign="CENTER",
    )
    snapshot.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), paper)]))
    story.append(snapshot)
    story.append(Spacer(1, 3 * mm))

    # ── Run Overview ──────────────────────────────────────────────────────────
    story.append(Paragraph("Run Overview", section_style))
    story.append(_make_table(
        [["Field", "Value"],
         ["Generated",    datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
         ["Dataset",      dataset_name],
         ["Input Shape",  str(input_shape)],
         ["Classes",      ", ".join(class_names)]],
        col_widths=[40 * mm, PAGE_W - 40 * mm],
    ))

    # ── Run Settings ──────────────────────────────────────────────────────────
    story.append(Paragraph("Run Settings", section_style))
    story.append(_make_table(
        [["Setting",          "Value"],
         ["Epochs",           args.epochs],
         ["Batch Size",       args.batch],
         ["Time Steps",       args.timesteps],
         ["Simulator",        "Disabled" if args.no_sim else "Enabled"],
         ["Simulation Steps", args.sim_steps]],
        col_widths=[50 * mm, PAGE_W - 50 * mm],
    ))

    # ── Model Results ─────────────────────────────────────────────────────────
    story.append(Paragraph("Model Results", section_style))
    results_table = _make_table(
        [["Model",            "Accuracy",  "Precision", "Recall",  "F1",      "ROC-AUC"],
         ["Binary LSTM",
          f"{binary_res['accuracy']:.4f}",
          f"{binary_res['precision']:.4f}",
          f"{binary_res['recall']:.4f}",
          f"{binary_res['f1']:.4f}",
          f"{binary_res['roc_auc']:.4f}"],
         ["Multiclass LSTM",
          f"{multi_res['accuracy']:.4f}",
          f"{multi_res['precision']:.4f}",
          f"{multi_res['recall']:.4f}",
          f"{multi_res['f1']:.4f}",
          "N/A"]],
        col_widths=[PAGE_W * 0.24, PAGE_W * 0.14, PAGE_W * 0.15, PAGE_W * 0.14, PAGE_W * 0.14, PAGE_W * 0.19],
    )
    results_table.hAlign = "CENTER"
    results_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), accent),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, accent_light]),
    ]))
    story.append(results_table)

    metric_strip = Table(
        [[
            _info_card(body_style, "Binary Accuracy",     f"{binary_res['accuracy']:.3f}", 14),
            _info_card(body_style, "Multiclass Accuracy", f"{multi_res['accuracy']:.3f}",  14),
            _info_card(body_style, "Elapsed",             f"{elapsed_seconds / 60:.1f} min", 14),
        ]],
        colWidths=[PAGE_W / 3, PAGE_W / 3, PAGE_W / 3],
        hAlign="CENTER",
    )
    metric_strip.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.white)]))
    story.append(Spacer(1, 3 * mm))
    story.append(metric_strip)

    # ── Working Summary ──────────────────────────────────────────────────────
    story.append(Paragraph("How the System Works", section_style))
    story.append(Paragraph(
        "The pipeline first cleans the industrial sensor data, converts it into time-series windows, "
        "balances the training folds safely after sequencing, and then trains multiple architectures. "
        "The binary model decides whether the machine is in a fault or no-fault state, the multiclass model "
        "identifies the specific failure type, and the RUL model estimates how many steps remain before failure.",
        body_style,
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "During evaluation, the project reports accuracy, F1, ROC-AUC, PR-AUC, MCC, calibration, and a model "
        "comparison table. The simulator then uses the saved models to show live risk, the predicted failure type, "
        "and the estimated remaining useful life.",
        body_style,
    ))

    # ── Fault Identification Summary ────────────────────────────────────────
    story.append(Paragraph("Fault vs No-Fault Identification", section_style))
    story.append(Paragraph(
        f"Binary classification is the first decision layer: when the model predicts No Fault, the machine is "
        f"treated as healthy; when it predicts Fault, the system escalates to the multiclass classifier to name the "
        f"specific failure type. This makes the output easier to interpret in a maintenance setting and gives a "
        f"direct link between sensor patterns and machine state.",
        body_style,
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"In the current run, the binary model achieved {binary_res['accuracy']:.3f} accuracy with an F1 of "
        f"{binary_res['f1']:.3f}, while the multiclass model achieved {multi_res['accuracy']:.3f} accuracy and "
        f"{multi_res['f1']:.3f} weighted F1. The comparison shows how each architecture balances detection quality "
        f"and calibration across the fault classes.",
        body_style,
    ))

    # ── Model Comparison ─────────────────────────────────────────────────────
    story.append(Paragraph("Model Comparison", section_style))
    if os.path.exists("results_summary.csv"):
        try:
            comparison_df = pd.read_csv("results_summary.csv")
            comparison_df = comparison_df[[
                "Task", "Architecture", "Accuracy", "F1", "ROC-AUC", "PR-AUC", "MCC", "Training Time (s)",
            ]]
            comparison_rows = [comparison_df.columns.tolist()] + comparison_df.astype(str).values.tolist()
            comparison_table = _make_table(
                comparison_rows,
                col_widths=[25 * mm, 28 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm, 16 * mm, 25 * mm],
            )
            comparison_table.hAlign = "CENTER"
            story.append(comparison_table)
        except Exception:
            story.append(Paragraph(
                "The model comparison table could not be loaded from results_summary.csv for this PDF build.",
                body_style,
            ))
    else:
        story.append(Paragraph(
            "results_summary.csv was not found, so the model comparison table could not be embedded.",
            body_style,
        ))

    # ── Visualizations — Binary ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Binary LSTM — Visualizations", section_style))

    for key, caption in [
        ("binary_history", "Figure 1 — Binary LSTM: Training Loss & Accuracy"),
        ("binary_cm",      "Figure 2 — Binary LSTM: Confusion Matrix"),
        ("binary_roc",     "Figure 3 — Binary LSTM: ROC Curve"),
        ("binary_f1",      "Figure 4 — Binary LSTM: F1 Score per Class"),
    ]:
        story.extend(_embed_plot(get_plot_path(key), caption))

    # ── Visualizations — Multiclass ───────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Multiclass LSTM — Visualizations", section_style))

    for key, caption in [
        ("multi_history", "Figure 5 — Multiclass LSTM: Training Loss & Accuracy"),
        ("multi_cm",      "Figure 6 — Multiclass LSTM: Confusion Matrix"),
        ("multi_roc",     "Figure 7 — Multiclass LSTM: One-vs-Rest ROC Curves"),
        ("multi_f1",      "Figure 8 — Multiclass LSTM: F1 Score per Class"),
    ]:
        story.extend(_embed_plot(get_plot_path(key), caption))

    # ── Simulator Timeline ────────────────────────────────────────────────────
    sim_path = get_plot_path("sim_timeline")
    if not args.no_sim and os.path.exists(sim_path):
        story.append(Paragraph("Simulator — Risk Timeline", section_style))
        story.extend(_embed_plot(sim_path, "Figure 9 — Simulator: Fault Risk % over Simulation Steps"))

    # ── Generated Files ───────────────────────────────────────────────────────
    story.append(Paragraph("Generated Files", section_style))
    file_rows = [["Status", "File", "Description"]]
    for path, desc in outputs:
        if path.endswith("simulation_log.csv") and args.no_sim:
            status = "SKIPPED"
        else:
            status = "OK" if os.path.exists(path) else "MISSING"
        file_rows.append([status, os.path.basename(path), desc])
    file_rows.append(["OK", os.path.basename(out_path), "Full PDF report with embedded plots"])
    files_table = _make_table(file_rows, col_widths=[24 * mm, 70 * mm, PAGE_W - 94 * mm])
    files_table.hAlign = "CENTER"
    files_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), slate),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(files_table)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"Total elapsed time: {elapsed_seconds / 60:.1f} min", note_style))

    # ── Build ─────────────────────────────────────────────────────────────────
    def add_page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawRightString(doc_obj.pagesize[0] - 16 * mm, 10 * mm, f"Page {doc_obj.page}")
        canvas.drawString(16 * mm, 10 * mm, "Predictive Fault Detection — Asjal Abdullah")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"\n  [Saved] {out_path}")
