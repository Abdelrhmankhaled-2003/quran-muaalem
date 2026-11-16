import logging
from dataclasses import asdict
import json
from pathlib import Path
from typing import Literal, Optional, Any, get_origin, get_args

import numpy as np
from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
from quran_transcript.utils import PartOfUthmaniWord
from quran_transcript.phonetics.moshaf_attributes import (
    get_arabic_attributes,
    get_arabic_name,
)
from librosa.core import load
from pydantic.fields import FieldInfo, PydanticUndefined
import torch
import gradio as gr

from quran_muaalem.inference import Muaalem
from quran_muaalem.muaalem_typing import MuaalemOutput
from quran_muaalem.explain import explain_for_terminal
from quran_muaalem.explain_gradio import explain_for_gradio
from quran_muaalem.audio_processing import (
    reduce_background_noise,
    trim_trailing_noise,
)

# Initialize components
REQUIRED_MOSHAF_FIELDS = [
    "rewaya",
    "takbeer",
    "madd_monfasel_len",
    "madd_mottasel_len",
    "madd_mottasel_waqf",
    "madd_aared_len",
    "madd_alleen_len",
    "ghonna_lam_and_raa",
    "meem_aal_imran",
    "madd_yaa_alayn_alharfy",
    "saken_before_hamz",
    "sakt_iwaja",
    "sakt_marqdena",
    "sakt_man_raq",
    "sakt_bal_ran",
    "sakt_maleeyah",
    "between_anfal_and_tawba",
    "noon_and_yaseen",
    "yaa_ataan",
    "start_with_ism",
    "yabsut",
    "bastah",
    "almusaytirun",
    "bimusaytir",
    "tasheel_or_madd",
    "yalhath_dhalik",
    "irkab_maana",
    "noon_tamnna",
    "harakat_daaf",
    "alif_salasila",
    "idgham_nakhluqkum",
    "raa_firq",
    "raa_alqitr",
    "raa_misr",
    "raa_nudhur",
    "raa_yasr",
    "meem_mokhfah",
]
tajweed_model_path = (
    Path(__file__).resolve().parents[2] / "models" / "tajweed-rules-v1"
)
model_id = (
    str(tajweed_model_path)
    if (tajweed_model_path / "config.json").is_file()
    else "obadx/muaalem-model-v3_2"
)
logging.basicConfig(level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"
muaalem = Muaalem(model_name_or_path=model_id, device=device)
sampling_rate = 16000

# Load Sura information
sura_idx_to_name = {}
sura_to_aya_count = {}
start_aya = Aya()
for sura_idx in range(1, 115):
    start_aya.set(sura_idx, 1)
    sura_idx_to_name[sura_idx] = start_aya.get().sura_name
    sura_to_aya_count[sura_idx] = start_aya.get().num_ayat_in_sura

# Default moshaf settings
default_moshaf = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=4,
)

# Current moshaf settings (will be updated from settings page)
current_moshaf = default_moshaf


def get_field_name(field_name: str, field_info: FieldInfo) -> str:
    """Return the Arabic name of the field if applicable else the field_name"""
    label = field_name
    arabic_name = get_arabic_name(field_info)
    if arabic_name:
        label = f"{arabic_name} ({field_name})"
    return label


def create_gradio_input_for_field(
    field_name: str,
    field_info: FieldInfo,
    default_value: Any = None,
    key_prefix="model_",
    help: str | None = None,
) -> Any:
    """Create a gradio input field given a pydantic field info"""
    # Extract Arabic name from field description if available
    label = get_field_name(field_name, field_info)

    if default_value is None:
        if field_info.default != PydanticUndefined:
            default_value = field_info.default

    if help is None:
        help = field_info.description

    # Handle Literal types
    if get_origin(field_info.annotation) is Literal:
        choices = list(get_args(field_info.annotation))
        arabic_attributes = get_arabic_attributes(field_info)

        # Create choice list with Arabic labels if available
        choice_list = []
        for choice in choices:
            if arabic_attributes and choice in arabic_attributes:
                choice_list.append((arabic_attributes[choice], choice))
            else:
                choice_list.append((str(choice), choice))

        return gr.Dropdown(
            choices=choice_list,
            value=default_value,
            label=label,
            info=help,
            interactive=True,
        )

    # Handle different field types
    if field_info.annotation in [str, Optional[str]]:
        return gr.Textbox(value=default_value or "", label=label, info=help)
    elif field_info.annotation in [int, Optional[int]]:
        return gr.Number(value=default_value or 0, label=label, info=help, precision=0)
    elif field_info.annotation in [float, Optional[float]]:
        return gr.Number(
            value=default_value or 0.0, label=label, info=help, precision=1
        )
    elif field_info.annotation in [bool, Optional[bool]]:
        return gr.Checkbox(value=default_value or False, label=label, info=help)

    raise ValueError(f"Unsupported field type for {label}: {field_info.annotation}")


def update_aya_dropdown(sura_idx):
    if not sura_idx:
        sura_idx = 1
    return gr.update(
        choices=list(range(1, sura_to_aya_count[int(sura_idx)] + 1)), value=1
    )


def update_num_words_for_ayah(sura_idx, aya_idx, start_idx, num_words):
    if any(value is None for value in (sura_idx, aya_idx, start_idx, num_words)):
        return gr.update()

    ayah_word_count = len(Aya(int(sura_idx), int(aya_idx)).get().uthmani.split())
    remaining_words = ayah_word_count - int(start_idx)
    if remaining_words < 1:
        return gr.update()

    return gr.update(value=min(int(num_words), remaining_words))


def update_uthmani_ref(sura_idx, aya_idx, start_idx, num_words):
    if not all([sura_idx, aya_idx, start_idx is not None, num_words is not None]):
        return ""
    try:
        uthmani_ref = (
            Aya(int(sura_idx), int(aya_idx))
            .get_by_imlaey_words(int(start_idx), int(num_words))
            .uthmani
        )
        return uthmani_ref
    except PartOfUthmaniWord as e:
        return f"⚠️ Warning: You've selected part of a Uthmani word. Please adjust the number of words to include complete words only.\n\nError details: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def process_audio(
    audio, sura_idx, aya_idx, start_idx, num_words, noise_cancellation=True
):
    global current_moshaf

    if audio is None:
        return None, "Please upload an audio file first"

    try:
        # Get Uthmani reference text
        uthmani_ref = (
            Aya(int(sura_idx), int(aya_idx))
            .get_by_imlaey_words(int(start_idx), int(num_words))
            .uthmani
        )
        phonetizer_out = quran_phonetizer(
            uthmani_ref, current_moshaf, remove_spaces=True
        )

        # Process audio
        wave, _ = load(audio, sr=sampling_rate, mono=True)
        if noise_cancellation:
            trimmed_length = trim_trailing_noise(wave, sampling_rate).size
            wave = reduce_background_noise(wave, sampling_rate)
            wave = wave[:trimmed_length]
        outs = muaalem(
            [wave],
            [phonetizer_out],
            sampling_rate=sampling_rate,
        )

        # # Prepare output
        # output_text = f"Phonemes: {outs[0].phonemes}\n\n"
        # for sifa in outs[0].sifat:
        #     output_text += json.dumps(asdict(sifa), indent=2, ensure_ascii=False) + "\n"
        #     output_text += "*" * 30 + "\n"
        # output_text += "-" * 40 + "\n\n"

        # Add explanation
        explanation_html = explain_for_gradio(
            outs[0].phonemes.text,
            phonetizer_out.phonemes,
            outs[0].sifat,
            phonetizer_out.sifat,
            lang="arabic",
            tajweed_rules=outs[0].tajweed_rules,
        )

        playback_wave = np.clip(
            np.rint(wave * 32768.0), -32768, 32767
        ).astype(np.int16)
        return (sampling_rate, playback_wave), explanation_html

    except PartOfUthmaniWord as e:
        return (
            None,
            f"⚠️ Error: The selected word range includes partial Uthmani words. Please adjust the number of words to include complete words only.\n\nError details: {str(e)}",
        )
    # except Exception as e:
    #     return f"Error processing audio: {str(e)}"


def update_moshaf_settings(*args):
    """Update the global moshaf settings with values from the settings page"""
    global current_moshaf, field_names

    try:
        # Create a dictionary from the field names and values
        settings_dict = dict(zip(field_names, args))

        # Create a new MoshafAttributes object with the updated values
        current_moshaf = MoshafAttributes(**settings_dict)
        return "✅ تم حفظ الإعدادات بنجاح - Settings saved successfully!"
    except Exception as e:
        return f"❌ خطأ في حفظ الإعدادات - Error saving settings: {str(e)}"


def reset_settings():
    """Reset all settings to default values"""
    global current_moshaf

    try:
        current_moshaf = default_moshaf
        # Return default values for all fields
        default_values = [
            getattr(default_moshaf, field_name) for field_name in field_names
        ]
        return default_values + [
            "✅ تم إعادة التعيين إلى الإعدادات الافتراضية - Reset to default settings successfully!"
        ]
    except Exception as e:
        return [getattr(current_moshaf, field_name) for field_name in field_names] + [
            f"❌ Error resetting settings: {str(e)}"
        ]


AI_ADVISORY_NOTICE = """
> **إشعار الذكاء الاصطناعي / AI Advisory Notice**
>
> **العربية:** هذا المعلّم القرآني نموذج ذكاء اصطناعي قد يخطئ؛ فهو للإعانة فقط ولا يُعتمد عليه، وليس بديلاً عن معلّمٍ مُتقِنٍ مُجاز؛ لأنّ القرآن لا يُؤخذ إلا بالمشافهة.
>
> **English:** This Quranic Teacher is an artificial-intelligence model that may make mistakes. It is intended for assistance only and must not be relied upon. It is not a replacement for a proficient, authorized teacher, because the Quran is only received through oral transmission (mushafahah).
"""


# Create the Gradio app
with gr.Blocks(title="المعلم القرآني") as app:
    gr.Markdown(AI_ADVISORY_NOTICE)

    # Store current moshaf settings in session state
    current_moshaf_state = gr.State(default_moshaf)

    # Initialize field names list
    field_names = []

    with gr.Tab("التحليل الرئيسي - Main Analysis"):
        gr.Markdown("# كشف أخطاء التلاوة والتجويد وصفات الحروف")
        gr.Markdown("اختر المقطع القرآني المراد تعلمه")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### التلاة المقارنة")

                # Create sura dropdown with both index and name
                sura_choices = [
                    (f"{idx} - {sura_idx_to_name[idx]}", idx) for idx in range(1, 115)
                ]
                sura_dropdown = gr.Dropdown(
                    choices=sura_choices,
                    label="السورة",
                    value=1,
                    elem_id="sura_dropdown",
                )

                aya_dropdown = gr.Dropdown(
                    choices=list(range(1, sura_to_aya_count[1] + 1)),
                    label="رقم الآية",
                    value=1,
                    elem_id="aya_dropdown",
                )
                start_idx = gr.Number(
                    value=0,
                    label="رقمة الكلمة بداية من صفر (Word Index)",
                    minimum=0,
                    step=1,
                    elem_id="start_idx",
                )
                num_words = gr.Number(
                    value=4,
                    label="عدد الكلمات",
                    minimum=1,
                    step=1,
                    info="Reduced to the words remaining in this ayah when you change the ayah or starting word. Increase it manually to include following ayat.",
                    elem_id="num_words",
                )
                uthmani_text = gr.Textbox(
                    label="الرسم العثماني",
                    interactive=False,
                    elem_id="uthmani_text",
                )

            with gr.Column(scale=2):
                gr.Markdown("### فحص التلاوة القرآنية")
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    label="Upload or Record Audio",
                    type="filepath",
                    elem_id="audio_input",
                )
                noise_cancellation = gr.Checkbox(
                    label="Noise cancellation",
                    value=True,
                    info="Estimate steady background noise from quieter parts and trim a long background-only tail. Leave a short quiet lead-in when possible; turn off if letters sound distorted.",
                )
                analyze_btn = gr.Button(
                    "افحص التلاوة", variant="primary", elem_id="analyze_btn"
                )
                cleaned_audio = gr.Audio(
                    label="Audio analyzed by the model (noise-reduced when enabled)",
                    interactive=False,
                )
                output_html = gr.HTML(
                    label="نتيجة الفحص",
                    elem_id="output_html",
                )

        # Initial update of uthmani text
        app.load(
            update_uthmani_ref,
            inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=uthmani_text,
        )

        # Update aya dropdown when sura changes and reset aya_idx to 1
        sura_dropdown.change(
            update_aya_dropdown, inputs=sura_dropdown, outputs=aya_dropdown
        ).then(
            update_num_words_for_ayah,
            inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=num_words,
        ).then(
            update_uthmani_ref,
            inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=uthmani_text,
        )

        # Keep the default word selection within the selected ayah.
        for component in [aya_dropdown, start_idx]:
            component.change(
                update_num_words_for_ayah,
                inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
                outputs=num_words,
            ).then(
                update_uthmani_ref,
                inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
                outputs=uthmani_text,
            )

        num_words.change(
            update_uthmani_ref,
            inputs=[sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=uthmani_text,
        )

        # Process audio when button is clicked
        analyze_btn.click(
            process_audio,
            inputs=[
                audio_input,
                sura_dropdown,
                aya_dropdown,
                start_idx,
                num_words,
                noise_cancellation,
            ],
            outputs=[cleaned_audio, output_html],
        )

    with gr.Tab("إعدادات المصحف - Moshaf Settings"):
        gr.Markdown("# إعدادات خصائص المصحف")
        gr.Markdown("قم بتعديل خصائص المصحف حسب التلاوة المطلوبة")

        # Create settings inputs directly in the tab
        settings_components = []
        fields = MoshafAttributes.model_fields

        # Create inputs for all required fields
        for field_name in REQUIRED_MOSHAF_FIELDS:
            field_info = fields[field_name]
            input_component = create_gradio_input_for_field(
                field_name, field_info, getattr(default_moshaf, field_name, None)
            )
            settings_components.append(input_component)
            field_names.append(field_name)

        # Save button and status message
        with gr.Row():
            save_btn = gr.Button("حفظ الإعدادات - Save Settings", variant="primary")
            reset_btn = gr.Button("إعادة التعيين - Reset to Default")

        status_message = gr.Markdown()

        # Save settings event
        save_btn.click(
            update_moshaf_settings, inputs=settings_components, outputs=status_message
        )

        # Reset to default event
        reset_btn.click(
            reset_settings, inputs=[], outputs=settings_components + [status_message]
        )

    with gr.Tab("حول - About"):
        gr.Markdown("# حول المعلم القرآني - About Quran Muaalem")
        gr.Markdown(AI_ADVISORY_NOTICE)
        gr.Markdown(
            "الترخيص: [Quran Muaalem License (QML) v1.0]"
            "(https://github.com/obadx/quran-muaalem/blob/main/LICENSE)\n\n"
            "License: [Quran Muaalem License (QML) v1.0]"
            "(https://github.com/obadx/quran-muaalem/blob/main/LICENSE)"
        )


def main(app=app):
    app.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
    # app.launch(server_name="0.0.0.0", share=True)
