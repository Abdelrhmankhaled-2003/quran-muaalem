from types import SimpleNamespace

import torch
import pytest
from quran_transcript import alphabet

from quran_muaalem.modeling.tajweed_rules import (
    MEEM_SAKINAH_LABELS,
    NOON_TANWEEN_LABELS,
    decode_tajweed_rule_events,
    extract_tajweed_rule_labels,
    rule_labels_to_ids,
)
from quran_muaalem.explain_gradio import explain_for_gradio
from quran_muaalem.muaalem_typing import Sifa, TajweedRuleEvent


uth = alphabet.uthmani


def test_tanween_uses_the_next_word_and_skips_hamzat_wasl():
    text = (
        uth.baa
        + uth.tanween_fath
        + uth.tanween_idhaam_dterminer
        + uth.space
        + uth.hamzat_wasl
        + uth.lam
    )

    labels = extract_tajweed_rule_labels(text)

    assert labels["noon_tanween"] == ["idgham_bilaghunnah"]
    assert rule_labels_to_ids(labels)["noon_tanween"] == [3]


@pytest.mark.parametrize(
    ("next_letter", "expected"),
    [
        (uth.baa, "iqlab"),
        (uth.noon_ikhfaa_group[0], "ikhfa"),
        (uth.hamza_above_alif, "izhar"),
    ],
)
def test_noon_sakinah_classifies_by_following_letter(next_letter, expected):
    labels = extract_tajweed_rule_labels(
        uth.noon + uth.ras_haaa + uth.space + next_letter
    )

    assert labels["noon_tanween"] == [expected]


def test_vocalized_noon_is_not_labeled_as_noon_sakinah():
    labels = extract_tajweed_rule_labels(
        uth.noon + uth.fatha + uth.space + uth.baa
    )

    assert labels["noon_tanween"] == []


@pytest.mark.parametrize(
    ("next_letter", "expected"),
    [
        (uth.baa, "ikhfa_shafawi"),
        (uth.meem, "idgham_shafawi"),
        (uth.faa, "izhar_shafawi"),
    ],
)
def test_meem_sakinah_classifies_by_following_letter(next_letter, expected):
    labels = extract_tajweed_rule_labels(uth.meem + uth.ras_haaa + next_letter)

    assert labels["meem_sakinah"] == [expected]


def test_rule_targets_map_to_ctc_ids_with_zero_reserved_for_blank():
    labels = {
        "noon_tanween": [NOON_TANWEEN_LABELS[0], NOON_TANWEEN_LABELS[-1]],
        "meem_sakinah": [MEEM_SAKINAH_LABELS[0]],
    }

    assert rule_labels_to_ids(labels) == {
        "noon_tanween": [1, len(NOON_TANWEEN_LABELS)],
        "meem_sakinah": [1],
    }


def test_unknown_rule_label_is_rejected():
    with pytest.raises(ValueError, match="Unknown noon_tanween rule label"):
        rule_labels_to_ids(
            {"noon_tanween": ["unknown"], "meem_sakinah": []}
        )


def test_rule_event_decoder_collapses_repeats_but_keeps_blanks_separators():
    probabilities = torch.tensor(
        [
            [
                [0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0, 0.0, 0.1, 0.8],
            ]
        ]
    )

    events = decode_tajweed_rule_events(probabilities, [5], "noon_tanween")

    assert [event[0] for event in events[0]] == [
        "izhar",
        "izhar",
        "ikhfa",
    ]
    assert [event[2] for event in events[0]] == [0, 3, 4]


def test_tajweed_rules_are_columns_in_the_phoneme_sifat_table():
    ph = alphabet.phonetics
    phoneme_groups = [ph.baa, ph.taa]
    sifat = [
        Sifa(
            phonemes_group=group,
            hams_or_jahr=None,
            shidda_or_rakhawa=None,
            tafkheem_or_taqeeq=None,
            itbaq=None,
            safeer=None,
            qalqla=None,
            tikraar=None,
            tafashie=None,
            istitala=None,
            ghonna=None,
        )
        for group in phoneme_groups
    ]
    references = [
        SimpleNamespace(
            phonemes=group,
            hams_or_jahr=None,
            shidda_or_rakhawa=None,
            tafkheem_or_taqeeq=None,
            itbaq=None,
            safeer=None,
            qalqla=None,
            tikraar=None,
            tafashie=None,
            istitala=None,
            ghonna=None,
        )
        for group in phoneme_groups
    ]

    html = explain_for_gradio(
        phonemes="".join(phoneme_groups),
        exp_phonemes="".join(phoneme_groups),
        sifat=sifat,
        exp_sifat=references,
        lang="english",
        tajweed_rules=[
            TajweedRuleEvent(
                level="noon_tanween",
                rule="iqlab",
                prob=0.75,
                frame_index=12,
                phoneme_group_index=1,
            )
        ],
    )

    assert html.count("<table") == 1
    assert "Noon sakinah and tanween" in html
    assert "Meem sakinah" in html
    assert "Detected Tajweed rules" not in html
    second_row = html[html.rfind("<tr>", 0, html.index(f">{ph.taa}</td>")) :]
    second_row = second_row[: second_row.index("</tr>")]
    assert "Iqlab (75.0%)" in second_row
    assert "Iqlab (75.0%)" not in html[: html.index(f">{ph.taa}</td>")]
