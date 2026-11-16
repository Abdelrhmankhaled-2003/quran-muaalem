import logging

from quran_transcript import chunck_phonemes, QuranPhoneticScriptOutput
from transformers import AutoFeatureExtractor
import torch
from numpy.typing import NDArray

from .modeling.multi_level_tokenizer import MultiLevelTokenizer
from .modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from .decode import (
    multilevel_greedy_decode,
    phonemes_level_greedy_decode,
    phoneme_group_frame_indices,
)
from .muaalem_typing import (
    Unit,
    SingleUnit,
    Sifa,
    MuaalemOutput,
    TajweedRuleEvent,
)
from .modeling.vocab import derive_ismat_idhlaq_labels
from .modeling.tajweed_rules import (
    RULE_LEVEL_LABELS,
    decode_tajweed_rule_events,
)


def format_sifat(
    level_to_units: dict[str, list[Unit]],
    chunked_phonemes_batch: list[list[str]],
    multi_level_tokenizer: MultiLevelTokenizer,
) -> list[list[Sifa]]:
    sifat_batch = []
    for seq_idx in range(len(chunked_phonemes_batch)):
        sifat = []
        for idx, ph_group in enumerate(chunked_phonemes_batch[seq_idx]):
            sifa_dict = {}
            derived_labels = derive_ismat_idhlaq_labels(ph_group)
            for level in level_to_units:
                if level == "phonemes":
                    continue
                sifa_idx = idx
                if sifa_idx < len(level_to_units[level][seq_idx].ids):
                    label = int(level_to_units[level][seq_idx].ids[sifa_idx])
                    text = multi_level_tokenizer.sifat_to_en_vocab[level][label]
                    p = level_to_units[level][seq_idx].probs[sifa_idx]
                    sifa_dict[level] = SingleUnit(
                        text=text, prob=float(p), idx=int(label)
                    )
                else:
                    logging.info(
                        f"Sequence: `{seq_idx}` has short Level: {level} we will place it with `None`"
                    )
                    sifa_dict[level] = None
            for level in ("ismat", "idhlaq"):
                label = derived_labels[level] if derived_labels else None
                sifa_dict[level] = (
                    SingleUnit(text=label, prob=1.0, idx=int(label == level))
                    if label is not None
                    else None
                )
            sifat.append(
                Sifa(
                    phonemes_group=chunked_phonemes_batch[seq_idx][idx],
                    **sifa_dict,
                )
            )
        sifat_batch.append(sifat)
    return sifat_batch


class Muaalem:
    def __init__(
        self,
        model_name_or_path: str = "obadx/muaalem-model-v3_2",
        device: str = "cpu",
        dtype=torch.bfloat16,
    ):
        """
        Initializing Muallem Model

        Args:
            model_name_or_path: the huggingface model name or path
            device: the device to run model on
            dtype: the torch dtype. Default is `torch.bfloat16` as the model was trained on
        """
        self.device = device
        self.dtype = dtype

        self.model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_name_or_path)
        self.multi_level_tokenizer = MultiLevelTokenizer(model_name_or_path)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

        self.model.to(device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        waves: list[list[float] | torch.FloatTensor | NDArray],
        ref_quran_phonetic_script_list: list[QuranPhoneticScriptOutput],
        sampling_rate: int,
    ) -> list[MuaalemOutput]:
        """Infrence Funcion for the Quran Muaalem Project

                waves: input waves  batch , seq_len with different formats described above
                ref_quran_phonetic_script_list (list[QuranPhoneticScriptOutput]): list of the
                    phonetized ouput of `quran_transcript.quran_phonetizer` with `remove_space=True`

                sampleing_rate (int): has to be 16000

        Returns:
            list[MuaalemOutput]:
                A list of output objects, each containing phoneme predictions and their
                phonetic features (sifat) for a processed input.

            Each MuaalemOutput contains:
                phonemes (Unit):
                    A dataclass representing the predicted phoneme sequence with:
                        text (str): Concatenated string of all phonemes.
                        probs (Union[torch.FloatTensor, list[float]]):
                            Confidence probabilities for each predicted phoneme.
                        ids (Union[torch.LongTensor, list[int]]):
                            Token IDs corresponding to each phoneme.

                sifat (list[Sifa]):
                    A list of phonetic feature dataclasses (one per phoneme) with the
                    following optional properties (each is a SingleUnit or None):
                        - phonemes_group (str): the phonemes associated with the `sifa`
                        - hams_or_jahr (SingleUnit): either `hams` or `jahr`
                        - shidda_or_rakhawa (SingleUnit): either `shadeed`, `between`, or `rikhw`
                        - tafkheem_or_taqeeq (SingleUnit): either `mofakham`, `moraqaq`, or `low_mofakham`
                        - itbaq (SingleUnit): either `monfateh`, or `motbaq`
                        - safeer (SingleUnit): either `safeer`, or `no_safeer`
                        - qalqla (SingleUnit): eithr `moqalqal`, or `not_moqalqal`
                        - tikraar (SingleUnit): either `mokarar` or `not_mokarar`
                        - tafashie (SingleUnit): either `motafashie`, or `not_motafashie`
                        - istitala (SingleUnit): either `mostateel`, or `not_mostateel`
                        - ghonna (SingleUnit): either `maghnoon`, or `not_maghnoon`
                        - ismat (SingleUnit): either `ismat`, or `not_ismat`
                        - idhlaq (SingleUnit): either `idhlaq`, or `not_idhlaq`

            Each SingleUnit in Sifa properties contains:
                text (str): The feature's categorical label (e.g., "hams", "shidda").
                prob (float): Confidence probability for this feature.
                idx (int): Identifier for the feature class.
        """

        if sampling_rate != 16000:
            raise ValueError(f"`sampling_rate` has to be 16000 got: `{sampling_rate}`")

        # TODO: check input waves

        # Tokanizing Ref
        level_to_ref_ids = self.multi_level_tokenizer.tokenize(
            [r.phonemes for r in ref_quran_phonetic_script_list],
            [r.sifat for r in ref_quran_phonetic_script_list],
            to_dict=True,
            return_tensors="pt",
            padding="longest",
        )["input_ids"]

        features = self.processor(
            waves, sampling_rate=sampling_rate, return_tensors="pt"
        )
        features = {k: v.to(self.device, dtype=self.dtype) for k, v in features.items()}
        outs = self.model(**features, return_dict=False)[0]

        probs = {}
        for level in outs:
            probs[level] = (
                torch.nn.functional.softmax(outs[level], dim=-1).cpu().to(torch.float32)
            )

        rule_events_by_level = {}
        rule_levels = [level for level in RULE_LEVEL_LABELS if level in probs]
        if rule_levels:
            attention_mask = features.get("attention_mask")
            if attention_mask is None:
                output_lengths = [probs["phonemes"].shape[1]] * len(waves)
            else:
                output_lengths = (
                    self.model._get_feat_extract_output_lengths(
                        attention_mask.to(torch.long).sum(dim=-1)
                    )
                    .to(torch.long)
                    .cpu()
                    .tolist()
                )
            for level in rule_levels:
                rule_events_by_level[level] = decode_tajweed_rule_events(
                    probs[level], output_lengths, level
                )

        # Decoding only Phonemes Level
        phonemes_units = phonemes_level_greedy_decode(
            probs["phonemes"], self.multi_level_tokenizer.id_to_vocab["phonemes"]
        )

        chunked_phonemes_batch: list[list[str]] = []
        for phonemes_unit in phonemes_units:
            chunked_phonemes_batch.append(chunck_phonemes(phonemes_unit.text))
        phoneme_group_frames = [
            phoneme_group_frame_indices(
                phonemes_unit,
                chunked_phonemes_batch[index],
                self.multi_level_tokenizer.id_to_vocab["phonemes"],
            )
            for index, phonemes_unit in enumerate(phonemes_units)
        ]

        sifa_level_probs = {
            level: level_probs
            for level, level_probs in probs.items()
            if level in self.multi_level_tokenizer.id_to_vocab
        }
        level_to_units = multilevel_greedy_decode(
            level_to_probs=sifa_level_probs,
            level_to_id_to_vocab=self.multi_level_tokenizer.id_to_vocab,
            level_to_ref_ids=level_to_ref_ids,
            chunked_phonemes_batch=chunked_phonemes_batch,
            ref_chuncked_phonemes_batch=[
                [s.phonemes for s in r.sifat] for r in ref_quran_phonetic_script_list
            ],
            phonemes_units=phonemes_units,
        )

        sifat_batch: list[list[Sifa]] = format_sifat(
            level_to_units,
            chunked_phonemes_batch,
            self.multi_level_tokenizer,
        )

        outs = []
        # looping over the batch
        for idx in range(len(level_to_units["phonemes"])):
            tajweed_rules = []
            for level in RULE_LEVEL_LABELS:
                level_events = rule_events_by_level.get(level)
                if level_events is None:
                    continue
                for rule, prob, frame_index in level_events[idx]:
                    group_index = (
                        min(
                            range(len(phoneme_group_frames[idx])),
                            key=lambda group: abs(
                                phoneme_group_frames[idx][group] - frame_index
                            ),
                        )
                        if phoneme_group_frames[idx]
                        else None
                    )
                    tajweed_rules.append(
                        TajweedRuleEvent(
                            level=level,
                            rule=rule,
                            prob=prob,
                            frame_index=frame_index,
                            phoneme_group_index=group_index,
                        )
                    )
            outs.append(
                MuaalemOutput(
                    phonemes=level_to_units["phonemes"][idx],
                    sifat=sifat_batch[idx],
                    tajweed_rules=tajweed_rules,
                )
            )
        return outs
