import json
from dataclasses import asdict
from typing import get_origin, get_args, Literal

from quran_transcript import alphabet as alph
from quran_transcript import SifaOutput

SIFAT_ATTR_TO_ARABIC_WITHOUT_BRACKETS = {
    "hams": "همس",
    "jahr": "جهر",
    "shadeed": "شديد",
    "between": "بين الشدة والرخاوة",
    "rikhw": "رخو",
    "mofakham": "مفخم",
    "moraqaq": "مرقق",
    "low_mofakham": "أدنى المفخم",
    "monfateh": "منفتح",
    "motbaq": "مطبق",
    "safeer": "صفير",
    "no_safeer": "لا صفير",
    "moqalqal": "مقلقل",
    "not_moqalqal": "لا قلقلة",
    "mokarar": "مكرر",
    "not_mokarar": "لا تكرار",
    "motafashie": "متفشي",
    "not_motafashie": "لا تفشي",
    "mostateel": "مستطيل",
    "not_mostateel": "لا إستطالة",
    "maghnoon": "مغن",
    "not_maghnoon": "لا غنة",
    "ismat": "إصمات",
    "not_ismat": "لا إصمات",
    "idhlaq": "إذلاق",
    "not_idhlaq": "لا إذلاق",
}
SIFAT_ATTR_TO_ARABIC = {
    k: f"[{v}]" for k, v in SIFAT_ATTR_TO_ARABIC_WITHOUT_BRACKETS.items()
}
SIFAT_ATTR_TO_ENGLISH = {v: k for k, v in SIFAT_ATTR_TO_ARABIC.items()}

_PHONETIC_LETTERS = {
    alph.phonetics.hamza,
    alph.phonetics.baa,
    alph.phonetics.taa,
    alph.phonetics.thaa,
    alph.phonetics.jeem,
    alph.phonetics.haa_mohmala,
    alph.phonetics.khaa,
    alph.phonetics.daal,
    alph.phonetics.thaal,
    alph.phonetics.raa,
    alph.phonetics.zay,
    alph.phonetics.seen,
    alph.phonetics.sheen,
    alph.phonetics.saad,
    alph.phonetics.daad,
    alph.phonetics.taa_mofakhama,
    alph.phonetics.zaa_mofakhama,
    alph.phonetics.ayn,
    alph.phonetics.ghyn,
    alph.phonetics.faa,
    alph.phonetics.qaf,
    alph.phonetics.kaf,
    alph.phonetics.lam,
    alph.phonetics.meem,
    alph.phonetics.noon,
    alph.phonetics.haa,
    alph.phonetics.waw,
    alph.phonetics.yaa,
    alph.phonetics.alif,
    alph.phonetics.yaa_madd,
    alph.phonetics.waw_madd,
    alph.phonetics.alif_momala,
    alph.phonetics.hamza_mosahala,
    alph.phonetics.noon_mokhfah,
    alph.phonetics.meem_mokhfah,
}
_IDHLAQ_PHONETIC_LETTERS = {
    alph.phonetics.baa,
    alph.phonetics.faa,
    alph.phonetics.raa,
    alph.phonetics.meem,
    alph.phonetics.noon,
    alph.phonetics.lam,
    alph.phonetics.noon_mokhfah,
    alph.phonetics.meem_mokhfah,
}


def derive_ismat_idhlaq_labels(phonemes_group: str) -> dict[str, str] | None:
    """Derive both binary labels from the first phoneme in a letter group."""
    if not phonemes_group:
        return None

    phoneme = phonemes_group[0]
    if phoneme not in _PHONETIC_LETTERS:
        return None

    is_idhlaq = phoneme in _IDHLAQ_PHONETIC_LETTERS
    return {
        "ismat": "not_ismat" if is_idhlaq else "ismat",
        "idhlaq": "idhlaq" if is_idhlaq else "not_idhlaq",
    }


PAD_TOKEN = "[PAD]"
PAD_TOKEN_IDX = 0


def build_quran_phoneme_script_vocab(path: str):
    level_to_token_to_idx = {}
    # Phonemes level
    phonemes = list(asdict(alph.phonetics).values())
    level_to_token_to_idx["phonemes"] = {PAD_TOKEN: PAD_TOKEN_IDX}
    idx = 0
    for p in phonemes:
        if idx == PAD_TOKEN_IDX:
            idx += 1
        level_to_token_to_idx["phonemes"][p] = idx
        idx += 1

    for field_name, fieldinfo in SifaOutput.model_fields.items():
        if get_origin(fieldinfo.annotation) == Literal:
            level = field_name
            phonemes = get_args(fieldinfo.annotation)
            phonemes = [SIFAT_ATTR_TO_ARABIC[p] for p in phonemes]
            level_to_token_to_idx[level] = {PAD_TOKEN: PAD_TOKEN_IDX}
            idx = 0
            for p in phonemes:
                if idx == PAD_TOKEN_IDX:
                    idx += 1
                level_to_token_to_idx[level][p] = idx
                idx += 1

    with open(path, "w+", encoding="utf-8") as f:
        json.dump(level_to_token_to_idx, f, ensure_ascii=False, indent=2)
