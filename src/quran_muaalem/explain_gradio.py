from html import escape
from typing import Literal
import diff_match_patch as dmp

from .explain import expalin_sifat
from .muaalem_typing import TajweedRuleEvent
from .modeling.tajweed_rules import RULE_LEVEL_LABELS
from .modeling.vocab import SIFAT_ATTR_TO_ARABIC_WITHOUT_BRACKETS


_RULE_LEVEL_LABELS = {
    "noon_tanween": {
        "english": "Noon sakinah and tanween",
        "arabic": "أحكام النون الساكنة والتنوين",
    },
    "meem_sakinah": {
        "english": "Meem sakinah",
        "arabic": "أحكام الميم الساكنة",
    },
}
_RULE_NAMES = {
    "izhar": {"english": "Izhar", "arabic": "إظهار حلقي"},
    "idgham_bighunnah": {"english": "Idgham with ghunnah", "arabic": "إدغام بغنة"},
    "idgham_bilaghunnah": {
        "english": "Idgham without ghunnah",
        "arabic": "إدغام بغير غنة",
    },
    "iqlab": {"english": "Iqlab", "arabic": "إقلاب"},
    "ikhfa": {"english": "Ikhfa", "arabic": "إخفاء حقيقي"},
    "izhar_shafawi": {"english": "Izhar shafawi", "arabic": "إظهار شفوي"},
    "idgham_shafawi": {"english": "Idgham shafawi", "arabic": "إدغام شفوي"},
    "ikhfa_shafawi": {"english": "Ikhfa shafawi", "arabic": "إخفاء شفوي"},
}


def explain_for_gradio(
    phonemes: str,
    exp_phonemes: str,
    sifat: list,
    exp_sifat: list,
    lang: Literal["arabic", "english"] = "english",
    tajweed_rules: list[TajweedRuleEvent] | None = None,
) -> str:
    # Create diff-match-patch object
    dmp_obj = dmp.diff_match_patch()

    # Calculate differences using Google's diff-match-patch (same as terminal)
    diffs = dmp_obj.diff_main(exp_phonemes, phonemes)

    # Create HTML for phoneme differences
    phoneme_html = explain_phonemes_html(dmp_obj, diffs)

    # Create HTML for sifat table using your existing function
    rule_columns = format_tajweed_rule_columns(sifat, tajweed_rules or [], lang)
    sifat_table = expalin_sifat(
        sifat, exp_sifat, diffs, additional_columns=rule_columns
    )
    sifat_html = explain_sifat_html(sifat_table, lang)

    # Combine both sections
    html_output = f"""
    <div style="font-family: monospace; width: 100%;">
        <h3>مقارنة الحروف</h3>
        {phoneme_html}
        <h3>مقارنة صفات الحروف</h3>
        {sifat_html}
       <div class="color-legend">
    </div>
    """

    return html_output


def format_tajweed_rule_columns(
    sifat: list,
    tajweed_rules: list[TajweedRuleEvent],
    lang: Literal["arabic", "english"],
) -> dict[str, list[str]]:
    columns = {level: ["—"] * len(sifat) for level in RULE_LEVEL_LABELS}
    for event in tajweed_rules:
        if event.level not in columns or event.rule not in _RULE_NAMES:
            raise ValueError(f"Unknown Tajweed rule event: {event}")
        if event.phoneme_group_index is None or not (
            0 <= event.phoneme_group_index < len(sifat)
        ):
            raise ValueError(f"Tajweed event has no valid phoneme group: {event}")

        rule_name = _RULE_NAMES[event.rule][lang]
        value = f"{rule_name} ({event.prob:.1%})"
        row = event.phoneme_group_index
        columns[event.level][row] = (
            value
            if columns[event.level][row] == "—"
            else f"{columns[event.level][row]}<br>{value}"
        )
    return columns


def explain_phonemes_html(dmp_obj, diffs):
    html_output = '<div style="background-color: #000; padding: 10px; border-radius: 5px; margin-bottom: 20px; font-size: 30px;">'

    # Process each difference (same logic as terminal version)
    for op, data in diffs:
        if op == dmp_obj.DIFF_EQUAL:
            html_output += f'<span style="color: #ffffff;">{data}</span>'
        elif op == dmp_obj.DIFF_INSERT:
            html_output += f'<span style="color: #00ff00;">{data}</span>'
        elif op == dmp_obj.DIFF_DELETE:
            html_output += f'<span style="color: #ff0000; text-decoration: line-through;">{data}</span>'

    html_output += "</div>"
    return html_output


def explain_sifat_html(table, lang):
    if not table:
        return "<p>No sifat data available</p>"

    # Create HTML table with full width
    html_output = """
    <table style="width: 100%; border-collapse: collapse; background-color: #000; color: #fff; margin-bottom: 20px;">
        <thead>
            <tr>
    """

    # Get base columns (non-exp keys without 'tag')
    base_keys = [k for k in table[0].keys() if not k.startswith("exp_") and k != "tag"]
    rule_keys = [key for key in RULE_LEVEL_LABELS if key in base_keys]
    base_keys = [key for key in base_keys if key not in rule_keys] + rule_keys

    # Add columns
    for key in base_keys:
        column_name = _RULE_LEVEL_LABELS[key][lang] if key in rule_keys else key.replace("_", " ").title()
        html_output += f'<th style="border: 1px solid #444; padding: 8px; text-align: left;">{escape(column_name)}</th>'

    html_output += """
            </tr>
        </thead>
        <tbody>
    """

    # Add rows
    for row in table:
        tag = row["tag"]
        html_output += "<tr>"

        for key in base_keys:
            exp_key = f"exp_{key}"
            value = str(row[key])

            # Apply Arabic translation if needed
            if key != "phonemes" and lang == "arabic":
                value = SIFAT_ATTR_TO_ARABIC_WITHOUT_BRACKETS.get(value, value)

            # Apply styling based on tag and comparison
            if tag == "exact" and row.get(exp_key) != row[key]:
                html_output += f'<td style="border: 1px solid #444; padding: 8px; color: #ff0000;">{value}</td>'
            elif tag == "insert":
                html_output += f'<td style="border: 1px solid #444; padding: 8px; color: #ffff00;">{value}</td>'
            else:
                html_output += (
                    f'<td style="border: 1px solid #444; padding: 8px;">{value}</td>'
                )

        html_output += "</tr>"

    html_output += """
        </tbody>
    </table>
    """

    return f'<div style="overflow-x: auto;">{html_output}</div>'
