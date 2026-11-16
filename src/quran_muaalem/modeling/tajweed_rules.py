import torch
from quran_transcript import alphabet


uth = alphabet.uthmani

NOON_TANWEEN_LABELS = (
    "izhar",
    "idgham_bighunnah",
    "idgham_bilaghunnah",
    "iqlab",
    "ikhfa",
)
MEEM_SAKINAH_LABELS = (
    "izhar_shafawi",
    "idgham_shafawi",
    "ikhfa_shafawi",
)
RULE_LEVEL_LABELS = {
    "noon_tanween": NOON_TANWEEN_LABELS,
    "meem_sakinah": MEEM_SAKINAH_LABELS,
}
RULE_BLANK_ID = 0

_LETTERS = set(uth.pure_letters_group) | {
    uth.alif,
    uth.alif_maksora,
    uth.hamzat_wasl,
    uth.hamza_above_alif,
    uth.hamza_below_alif,
    uth.hamza_above_waw,
    uth.hamza_above_yaa,
}
_MARKS = {
    char
    for name in (
        "fatha",
        "dama",
        "kasra",
        "tanween_fath",
        "tanween_dam",
        "tanween_kasr",
        "shadda",
        "ras_haaa",
        "madd",
        "alif_khnjaria",
        "small_seen_above",
        "small_seen_below",
        "small_waw",
        "small_yaa_sila",
        "small_yaa",
        "small_noon",
        "skoon_mostadeer",
        "skoon_mostateel",
        "meem_iqlab",
        "tanween_idhaam_dterminer",
        "imala_sign",
        "ishmam_sign",
        "tasheel_sign",
    )
    for char in getattr(uth, name)
}
_MARKS.add(uth.hamza_mamdoda)
_VOWELS = {
    uth.fatha,
    uth.dama,
    uth.kasra,
    uth.tanween_fath,
    uth.tanween_dam,
    uth.tanween_kasr,
}
_TANWEEN = {uth.tanween_fath, uth.tanween_dam, uth.tanween_kasr}
_IZHAR_LETTERS = {
    uth.hamza,
    uth.hamza_above_alif,
    uth.hamza_below_alif,
    uth.hamza_above_waw,
    uth.hamza_above_yaa,
    uth.haa,
    uth.haa_mohmala,
    uth.ayn,
    uth.khaa,
    uth.ghyn,
}
_IDGHAM_BIGHUNNA_LETTERS = {uth.yaa, uth.noon, uth.meem, uth.waw}
_IDGHAM_BILAGHUNNA_LETTERS = {uth.lam, uth.raa}
_IKHFA_LETTERS = set(uth.noon_ikhfaa_group)


def _next_letter(
    text: str, start: int, crossed_space: bool = False
) -> tuple[str, bool] | None:
    for char in text[start:]:
        if char in _LETTERS:
            if crossed_space and char == uth.hamzat_wasl:
                continue
            return char, crossed_space
        if char == uth.space:
            crossed_space = True
            continue
        if char in _MARKS or char == uth.kasheeda:
            continue
        return None
    return None


def _next_word_letter(text: str, start: int) -> str | None:
    word_end = text.find(uth.space, start)
    if word_end < 0:
        return None
    result = _next_letter(text, word_end + 1, crossed_space=True)
    return result[0] if result else None


def _has_vowel_mark(text: str, position: int) -> bool:
    for char in text[position + 1 :]:
        if char in _LETTERS or char == uth.space:
            return False
        if char in _VOWELS:
            return True
        if char not in _MARKS and char != uth.kasheeda:
            return False
    return False


def _classify_noon_tanween(next_letter: str | None) -> str | None:
    if next_letter == uth.baa:
        return "iqlab"
    if next_letter in _IZHAR_LETTERS:
        return "izhar"
    if next_letter in _IDGHAM_BIGHUNNA_LETTERS:
        return "idgham_bighunnah"
    if next_letter in _IDGHAM_BILAGHUNNA_LETTERS:
        return "idgham_bilaghunnah"
    if next_letter in _IKHFA_LETTERS:
        return "ikhfa"
    return None


def extract_tajweed_rule_labels(uthmani_text: str) -> dict[str, list[str]]:
    """Derive contextual rule-event targets from canonical Uthmani text."""
    if not isinstance(uthmani_text, str):
        raise TypeError("uthmani_text must be a string")

    labels = {level: [] for level in RULE_LEVEL_LABELS}
    for position, char in enumerate(uthmani_text):
        if char in _TANWEEN:
            rule = _classify_noon_tanween(
                _next_word_letter(uthmani_text, position + 1)
            )
            if rule is not None:
                labels["noon_tanween"].append(rule)
        elif char == uth.noon and not _has_vowel_mark(uthmani_text, position):
            next_letter = _next_letter(uthmani_text, position + 1)
            if next_letter is not None:
                rule = _classify_noon_tanween(next_letter[0])
                if (
                    not next_letter[1]
                    and next_letter[0] in {uth.yaa, uth.waw}
                ):
                    rule = "izhar"
                if rule is not None:
                    labels["noon_tanween"].append(rule)
        elif char == uth.meem and not _has_vowel_mark(uthmani_text, position):
            next_letter = _next_letter(uthmani_text, position + 1)
            if next_letter is None:
                continue
            if next_letter[0] == uth.baa:
                labels["meem_sakinah"].append("ikhfa_shafawi")
            elif next_letter[0] == uth.meem:
                labels["meem_sakinah"].append("idgham_shafawi")
            else:
                labels["meem_sakinah"].append("izhar_shafawi")
    return labels


def rule_labels_to_ids(labels: dict[str, list[str]]) -> dict[str, list[int]]:
    """Map named event targets to CTC IDs; zero is reserved for the blank."""
    result = {}
    for level, class_names in RULE_LEVEL_LABELS.items():
        if level not in labels:
            raise ValueError(f"Missing rule targets for level: {level}")
        class_to_id = {name: index + 1 for index, name in enumerate(class_names)}
        try:
            result[level] = [class_to_id[name] for name in labels[level]]
        except KeyError as exc:
            raise ValueError(
                f"Unknown {level} rule label: {exc.args[0]}"
            ) from exc
    return result


def decode_tajweed_rule_events(
    probabilities: torch.Tensor,
    sequence_lengths: list[int],
    level: str,
) -> list[list[tuple[str, float, int]]]:
    """Greedy CTC-decode rule events as (label, confidence, frame index)."""
    if level not in RULE_LEVEL_LABELS:
        raise ValueError(f"Unknown Tajweed rule level: {level}")
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (batch, time, classes)")
    if len(sequence_lengths) != probabilities.shape[0]:
        raise ValueError("sequence_lengths must contain one value per batch item")
    if probabilities.shape[2] != len(RULE_LEVEL_LABELS[level]) + 1:
        raise ValueError(f"Unexpected class count for Tajweed rule level: {level}")

    labels = RULE_LEVEL_LABELS[level]
    decoded = []
    for batch_idx, sequence_length in enumerate(sequence_lengths):
        if not 0 <= sequence_length <= probabilities.shape[1]:
            raise ValueError("sequence length is outside the probability tensor")
        frame_probabilities = probabilities[batch_idx, :sequence_length]
        frame_ids = frame_probabilities.argmax(dim=-1)
        frame_confidences = frame_probabilities.max(dim=-1).values
        events = []
        previous_id = RULE_BLANK_ID
        for frame_idx, token_id in enumerate(frame_ids.tolist()):
            if token_id != RULE_BLANK_ID and token_id != previous_id:
                events.append(
                    (
                        labels[token_id - 1],
                        float(frame_confidences[frame_idx]),
                        frame_idx,
                    )
                )
            previous_id = token_id
        decoded.append(events)
    return decoded
