"""
report_utils.py
===============
PDF report generation utilities for the predictive maintenance project.
"""

from __future__ import annotations

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _make_table(data, col_widths=None, header_fill="#111827"):
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_fill)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _info_card(body_style, label: str, value: str, value_size: int = 13):
    table = Table(
        [[
            Paragraph(f"<font color='#64748B'><b>{label}</b></font>", body_style),
            Paragraph(f"<font size='{value_size}' color='#111827'><b>{value}</b></font>", body_style),
        ]],
        colWidths=[28 * mm, 52 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


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

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.white,
        spaceAfter=0,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.white,
        spaceAfter=0,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#111827"),
        spaceBefore=10,
        spaceAfter=6,
    )
    note_style = ParagraphStyle(
        "NoteText",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        alignment=TA_CENTER,
    )
    body_style = ParagraphStyle(
        "BodyTextClean",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1F2937"),
    )

    accent = colors.HexColor("#0F4C81")
    accent_light = colors.HexColor("#EAF2FB")
    slate = colors.HexColor("#334155")
    paper = colors.HexColor("#F8FAFC")

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Predictive Fault Detection Report",
        author="GitHub Copilot",
    )

    story = []
    banner = Table(
        [[
            Paragraph("Predictive Fault Detection Report", title_style),
            Paragraph("Clean summary of the run, metrics, and generated artifacts", subtitle_style),
            Paragraph(f"Generated {datetime.now().strftime('%d %b %Y, %H:%M')}", note_style),
        ]],
        colWidths=[82 * mm, 60 * mm, 34 * mm],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), accent),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(banner)
    story.append(Spacer(1, 4 * mm))

    snapshot = Table(
        [[
            _info_card(body_style, "Dataset", dataset_name),
            _info_card(body_style, "Input Shape", str(input_shape)),
            _info_card(body_style, "Classes", str(len(class_names))),
        ]],
        colWidths=[54 * mm, 54 * mm, 54 * mm],
    )
    snapshot.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), paper)]))
    story.append(snapshot)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("A polished summary of the run, metrics, and generated artifacts.", note_style))
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("Run Overview", section_style))
    overview_rows = [
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Dataset", dataset_name],
        ["Input Shape", str(input_shape)],
        ["Classes", ", ".join(class_names)],
    ]
    story.append(_make_table([["Field", "Value"]] + overview_rows, col_widths=[35 * mm, 135 * mm]))

    story.append(Paragraph("Run Settings", section_style))
    settings_rows = [
        ["Epochs", args.epochs],
        ["Batch", args.batch],
        ["Timesteps", args.timesteps],
        ["Simulator", "Disabled" if args.no_sim else "Enabled"],
        ["Simulation Steps", args.sim_steps],
    ]
    story.append(_make_table([["Setting", "Value"]] + settings_rows, col_widths=[45 * mm, 125 * mm]))

    story.append(Paragraph("Model Results", section_style))
    results_data = [
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        [
            "Binary LSTM",
            f"{binary_res['accuracy']:.4f}",
            f"{binary_res['precision']:.4f}",
            f"{binary_res['recall']:.4f}",
            f"{binary_res['f1']:.4f}",
            f"{binary_res['roc_auc']:.4f}",
        ],
        [
            "Multiclass LSTM",
            f"{multi_res['accuracy']:.4f}",
            f"{multi_res['precision']:.4f}",
            f"{multi_res['recall']:.4f}",
            f"{multi_res['f1']:.4f}",
            "N/A",
        ],
    ]
    results_table = _make_table(results_data, col_widths=[38 * mm, 25 * mm, 25 * mm, 22 * mm, 18 * mm, 25 * mm])
    results_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), accent),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, accent_light]),
            ]
        )
    )
    story.append(results_table)

    metric_strip = Table(
        [[
            _info_card(body_style, "Binary Accuracy", f"{binary_res['accuracy']:.3f}", 14),
            _info_card(body_style, "Multiclass Accuracy", f"{multi_res['accuracy']:.3f}", 14),
            _info_card(body_style, "Elapsed", f"{elapsed_seconds / 60:.1f} min", 14),
        ]],
        colWidths=[54 * mm, 54 * mm, 54 * mm],
    )
    metric_strip.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.white)]))
    story.append(Spacer(1, 3 * mm))
    story.append(metric_strip)

    story.append(Paragraph("Generated Files", section_style))
    file_rows = [["Status", "File", "Description"]]
    for path, desc in outputs:
        if path.endswith("simulation_log.csv") and args.no_sim:
            status = "SKIPPED"
        else:
            status = "OK" if os.path.exists(path) else "MISSING"
        file_rows.append([status, os.path.basename(path), desc])
    file_rows.append(["OK", os.path.basename(out_path), "Clean PDF report"])
    files_table = _make_table(file_rows, col_widths=[24 * mm, 58 * mm, 88 * mm])
    files_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), slate),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    story.append(files_table)

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"Elapsed time: {elapsed_seconds / 60:.1f} min", note_style))

    def add_page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawRightString(doc_obj.pagesize[0] - 16 * mm, 10 * mm, f"Page {doc_obj.page}")
        canvas.drawString(16 * mm, 10 * mm, "Predictive Fault Detection Report")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"\n  Saved → {out_path}")
