# المخرجات والشرح

الاستدلال يُرجع قائمة من `MuaalemOutput` (`src/quran_muaalem/muaalem_typing.py`). كل عنصر يحتوي على:

- `phonemes`: كائن `Unit` مع نص الفونيمات، الاحتمالات، والمعرّفات.
- `sifat`: قائمة `Sifa` (عنصر لكل مجموعة فونيمات) مع خصائص اختيارية.

## مخطط المخرجات (مفهومي)

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

## مثال مبسّط

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

> ملاحظات:
> - `probs` ناتجة عن softmax خاص بـ CTC وليست مُعايرة بالضرورة.
> - `ismat` و`idhlaq` مشتقان من الفونيم المتعرّف عليه، ولا يتنبأ بهما رأسا نموذج منفصلان. احتمالهما `1.0` مشروط بصحة الفونيم المتعرّف عليه.
> - حروف الإذلاق هي `ب ر ف م ن ل`، وتُصنّف بقية الحروف المتعرّف عليها ضمن الإصمات.
> - قد تكون بعض حقول `Sifa` بقيمة `None` إذا حدث عدم تطابق في المحاذاة.
> - `tajweed_rules` تكون فارغة للنماذج التي لا تحتوي على رؤوس قواعد التجويد. `frame_index` يشير إلى إطار خرج النموذج وليس إلى زمن بالثواني، والاحتمال غير مُعاير.

تتضمن النماذج المدرّبة لرؤوس القواعد أحداثًا من المستويين `noon_tanween` و`meem_sakinah`، مثل `ikhfa` و`idgham_bighunnah` و`ikhfa_shafawi`. هذه التوقعات تتعرف على وقوع القاعدة، ولا تقيّم أخطاء المتعلّم.

## مقارنة التوقع مع المرجع

هناك وحدتان لعرض النتائج:

- `src/quran_muaalem/explain.py` يعرض جدولًا في الطرفية باستخدام `rich`.
  - `explain_for_terminal(...)` يبني فرقًا بين الفونيمات المتوقعة والمرجع ثم يطبع جدولًا.
- `src/quran_muaalem/explain_gradio.py` يولّد HTML لواجهة Gradio.
  - `explain_for_gradio(...)` يعرض فرقًا ملونًا للفونيمات وجدولًا للخصائص.
  - يتضمن الجدول عمودَي أحكام النون الساكنة والتنوين والميم الساكنة؛ وتظهر القاعدة المكتشفة وثقتها بجانب مجموعة الفونيمات المرتبطة بها.

كلاهما يستخدم `diff-match-patch` لتقسيم الإدراجات والحذف والاختلافات الجزئية بين الفونيمات.

## حقول `Sifa`

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
