# Outputs and Explanations

The inference output is a list of `MuaalemOutput` objects (`src/quran_muaalem/muaalem_typing.py`). Each item contains:

- `phonemes`: a `Unit` with decoded phoneme text, probabilities, and ids.
- `sifat`: a list of `Sifa` entries (one per phoneme group), each with optional phonetic attributes.

## Output schema (conceptual)

```text
MuaalemOutput
  phonemes: Unit
  sifat: list[Sifa]
  tajweed_rules: list[TajweedRuleEvent]

TajweedRuleEvent
  level: str
  rule: str
  prob: float
  frame_index: int

Unit
  text: str
  probs: Tensor | list[float]
  ids: Tensor | list[int]

Sifa
  phonemes_group: str
  hams_or_jahr: SingleUnit | None
  shidda_or_rakhawa: SingleUnit | None
  tafkheem_or_taqeeq: SingleUnit | None
  itbaq: SingleUnit | None
  safeer: SingleUnit | None
  qalqla: SingleUnit | None
  tikraar: SingleUnit | None
  tafashie: SingleUnit | None
  istitala: SingleUnit | None
  ghonna: SingleUnit | None
  ismat: SingleUnit | None
  idhlaq: SingleUnit | None

SingleUnit
  text: str
  prob: float
  idx: int
```

## Example (abridged)

```json
{
  "phonemes": {
    "text": "بِسْمِٱللَّهِ...",
    "probs": [0.98, 0.93, 0.87],
    "ids": [12, 7, 31]
  },
  "sifat": [
    {
      "phonemes_group": "بِ",
      "hams_or_jahr": {"text": "jahr", "prob": 0.99, "idx": 1},
      "shidda_or_rakhawa": {"text": "shadeed", "prob": 0.95, "idx": 2},
      "tafkheem_or_taqeeq": {"text": "moraqaq", "prob": 0.94, "idx": 1},
      "itbaq": {"text": "monfateh", "prob": 0.92, "idx": 1},
      "safeer": {"text": "no_safeer", "prob": 0.99, "idx": 0},
      "qalqla": {"text": "not_moqalqal", "prob": 0.99, "idx": 0},
      "tikraar": {"text": "not_mokarar", "prob": 0.99, "idx": 0},
      "tafashie": {"text": "not_motafashie", "prob": 0.99, "idx": 0},
      "istitala": {"text": "not_mostateel", "prob": 0.99, "idx": 0},
      "ghonna": {"text": "not_maghnoon", "prob": 0.99, "idx": 0},
      "ismat": {"text": "not_ismat", "prob": 1.0, "idx": 0},
      "idhlaq": {"text": "idhlaq", "prob": 1.0, "idx": 1}
    }
  ]
}
```

> Notes:
> - `probs` are derived from the CTC softmax; they are not calibrated.
> - `ismat` and `idhlaq` are derived from the recognized phoneme, not predicted by separate model heads. Their probability is `1.0` conditional on that phoneme prediction.
> - Idhlaq applies to `ب ر ف م ن ل`; other recognized letters are labeled Ismat.
> - Some `Sifa` fields may be `None` if alignment length mismatches occur.
> - `tajweed_rules` is empty for checkpoints without Tajweed rule heads. `frame_index` is a model-output frame, not a time in seconds, and its probability is not calibrated.

Rule-enabled checkpoints return events from the `noon_tanween` and `meem_sakinah` levels, such as `ikhfa`, `idgham_bighunnah`, and `ikhfa_shafawi`. These outputs recognize rule occurrence; they do not assess learner errors.

## Comparing predictions to reference

Two helper modules format the outputs for humans:

- `src/quran_muaalem/explain.py` renders a terminal table using `rich`.
  - `explain_for_terminal(...)` builds a diff between predicted phonemes and the reference, then prints a table.
- `src/quran_muaalem/explain_gradio.py` renders HTML for the Gradio UI.
  - `explain_for_gradio(...)` shows a colorized phoneme diff and an attribute table.
  - The table includes Noon Sakinah/Tanween and Meem Sakinah columns; detected rules and their confidence are shown alongside the aligned phoneme group.

Both use `diff-match-patch` to segment insertions, deletions, and partial matches between the predicted phonemes and the reference phoneme string.

## Field list (Sifa)

- `hams_or_jahr`
- `shidda_or_rakhawa`
- `tafkheem_or_taqeeq`
- `itbaq`
- `safeer`
- `qalqla`
- `tikraar`
- `tafashie`
- `istitala`
- `ghonna`
- `ismat`
- `idhlaq`
