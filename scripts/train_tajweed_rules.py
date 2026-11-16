import argparse
from collections import defaultdict
from io import BytesIO
import json
import math
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.nn import functional as F
from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer

from quran_muaalem.modeling.modeling_multi_level_ctc import (
    Wav2Vec2BertForMultilevelCTC,
)
from quran_muaalem.modeling.tajweed_rules import (
    RULE_BLANK_ID,
    RULE_LEVEL_LABELS,
    extract_tajweed_rule_labels,
    rule_labels_to_ids,
)
from quran_muaalem.modeling.vocab import PAD_TOKEN


SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CTC heads for noon/tanween and meem-sakinah rules."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/tajweed-rules-v1")
    )
    parser.add_argument("--base-model", default="obadx/muaalem-model-v3_2")
    parser.add_argument("--train-config", default="moshaf_0.0")
    parser.add_argument("--eval-config", default="moshaf_1.0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-attention-cost", type=float, default=600.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-eval-samples", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    return parser.parse_args()


def iter_parquet_rows(
    config_dir: Path,
    shuffle: bool,
    rng: random.Random,
    limit: int | None = None,
    shuffle_buffer_size: int = 128,
):
    files = sorted(config_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {config_dir}")
    if shuffle:
        rng.shuffle(files)

    def rows():
        for file in files:
            parquet_file = pq.ParquetFile(file)
            for batch in parquet_file.iter_batches(
                columns=[
                    "audio",
                    "uthmani",
                    "sample_rate",
                    "segment_index",
                    "duration_seconds",
                ],
                batch_size=shuffle_buffer_size,
            ):
                yield from batch.to_pylist()

    if not shuffle:
        for count, row in enumerate(rows()):
            if limit is not None and count >= limit:
                break
            yield row
        return

    buffer_size = (
        min(shuffle_buffer_size, limit) if limit is not None else shuffle_buffer_size
    )
    buffer = []
    emitted = 0
    for row in rows():
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
        emitted += 1
        if limit is not None and emitted >= limit:
            return

    rng.shuffle(buffer)
    for row in buffer:
        if limit is not None and emitted >= limit:
            break
        yield row
        emitted += 1


def decode_audio(row: dict) -> np.ndarray:
    audio = row.get("audio")
    if not isinstance(audio, dict) or not audio.get("bytes"):
        raise ValueError(f"Missing embedded audio for segment {row.get('segment_index')}")

    waveform, sample_rate = sf.read(
        BytesIO(audio["bytes"]), dtype="float32", always_2d=True
    )
    waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid audio for segment {row.get('segment_index')}")
    return waveform


def iter_batches(rows, batch_size: int, max_attention_cost: float):
    batch = []
    max_duration = 0.0
    for row in rows:
        duration = float(row["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(
                f"Invalid duration for segment {row.get('segment_index')}: {duration}"
            )
        candidate_max_duration = max(max_duration, duration)
        candidate_attention_cost = (len(batch) + 1) * candidate_max_duration**2
        if batch and candidate_attention_cost > max_attention_cost:
            yield batch
            batch = []
            max_duration = 0.0
        batch.append(row)
        max_duration = max(max_duration, duration)
        if len(batch) == batch_size:
            yield batch
            batch = []
            max_duration = 0.0
    if batch:
        yield batch


def minimum_ctc_frames(target: list[int]) -> int:
    repeated_neighbors = sum(left == right for left, right in zip(target, target[1:]))
    return len(target) + repeated_neighbors


def ctc_loss(logits, targets: list[list[int]], input_lengths):
    flat_targets = torch.tensor(
        [token for target in targets for token in target],
        dtype=torch.long,
        device=logits.device,
    )
    target_lengths = torch.tensor(
        [len(target) for target in targets],
        dtype=torch.long,
        device=logits.device,
    )
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        flat_targets,
        input_lengths,
        target_lengths,
        blank=RULE_BLANK_ID,
        reduction="mean",
        zero_infinity=False,
    )


def greedy_decode(logits, input_lengths):
    frame_ids = logits.argmax(dim=-1)
    decoded = []
    for row, sequence_length in zip(frame_ids, input_lengths.tolist()):
        tokens = []
        previous = RULE_BLANK_ID
        for token_id in row[:sequence_length].tolist():
            if token_id != RULE_BLANK_ID and token_id != previous:
                tokens.append(token_id)
            previous = token_id
        decoded.append(tokens)
    return decoded


def batch_inputs(rows, feature_extractor, device):
    waveforms = [decode_audio(row) for row in rows]
    feature_batch = feature_extractor(
        waveforms,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_features = feature_batch["input_features"].to(
        device=device, dtype=torch.float16
    )
    attention_mask = feature_batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones(
            input_features.shape[:2], dtype=torch.long, device=device
        )
    else:
        attention_mask = attention_mask.to(device=device, dtype=torch.long)

    targets = {level: [] for level in RULE_LEVEL_LABELS}
    for row in rows:
        labels = extract_tajweed_rule_labels(row["uthmani"])
        id_labels = rule_labels_to_ids(labels)
        for level in RULE_LEVEL_LABELS:
            targets[level].append(id_labels[level])
    return input_features, attention_mask, targets


def run_epoch(
    *,
    config_dir: Path,
    training: bool,
    epoch: int,
    model,
    heads,
    optimizer,
    feature_extractor,
    device,
    batch_size: int,
    seed: int,
    sample_limit: int | None,
    batch_limit: int | None,
    max_attention_cost: float,
):
    rng = random.Random(seed + epoch)
    rows = iter_parquet_rows(config_dir, training, rng, sample_limit)
    model.wav2vec2_bert.eval()
    for head in heads.values():
        head.train(training)

    totals = defaultdict(float)
    exact_matches = defaultdict(int)
    sample_count = 0
    batch_count = 0
    for rows_batch in iter_batches(rows, batch_size, max_attention_cost):
        input_features, attention_mask, targets = batch_inputs(
            rows_batch, feature_extractor, device
        )
        input_lengths = model._get_feat_extract_output_lengths(
            attention_mask.sum(dim=-1)
        ).to(dtype=torch.long)

        for level, sequences in targets.items():
            for row, target, input_length in zip(rows_batch, sequences, input_lengths):
                required_frames = minimum_ctc_frames(target)
                if required_frames > int(input_length):
                    raise ValueError(
                        f"CTC target for {level} in segment "
                        f"{row.get('segment_index')} needs {required_frames} frames, "
                        f"but audio provides {int(input_length)}"
                    )

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            hidden_states = model.wav2vec2_bert(
                input_features=input_features,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

        losses = {}
        predictions = {}
        for level, head in heads.items():
            logits = head(hidden_states.float())
            losses[level] = ctc_loss(logits, targets[level], input_lengths)
            predictions[level] = greedy_decode(logits.detach(), input_lengths)

        total_loss = sum(losses.values()) / len(losses)
        if training:
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for head in heads.values() for parameter in head.parameters()],
                max_norm=1.0,
            )
            optimizer.step()

        for level, loss in losses.items():
            totals[f"{level}_loss"] += float(loss.detach()) * len(rows_batch)
            exact_matches[level] += sum(
                prediction == target
                for prediction, target in zip(predictions[level], targets[level])
            )
        totals["loss"] += float(total_loss.detach()) * len(rows_batch)
        sample_count += len(rows_batch)
        batch_count += 1
        if batch_count % 100 == 0:
            print(
                f"{'train' if training else 'eval'} epoch={epoch + 1} "
                f"batches={batch_count} samples={sample_count} "
                f"loss={totals['loss'] / sample_count:.4f}",
                flush=True,
            )
        if batch_limit is not None and batch_count >= batch_limit:
            break

    if sample_count == 0:
        raise ValueError(f"No audio samples found in {config_dir}")
    metrics = {key: value / sample_count for key, value in totals.items()}
    metrics["samples"] = sample_count
    metrics["batches"] = batch_count
    for level in RULE_LEVEL_LABELS:
        metrics[f"{level}_sequence_accuracy"] = exact_matches[level] / sample_count
    return metrics


def main():
    args = parse_args()
    positive_values = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "learning-rate": args.learning_rate,
        "max-attention-cost": args.max_attention_cost,
    }
    if any(value <= 0 for value in positive_values.values()):
        raise ValueError(f"{positive_values} must all be positive")
    optional_limits = {
        "limit-train-samples": args.limit_train_samples,
        "limit-eval-samples": args.limit_eval_samples,
        "max-train-batches": args.max_train_batches,
        "max-eval-batches": args.max_eval_batches,
    }
    if any(value is not None and value < 1 for value in optional_limits.values()):
        raise ValueError(f"{optional_limits} must be positive when provided")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run, but is unavailable")

    dataset_root = args.dataset_root.expanduser().resolve()
    train_dir = dataset_root / args.train_config
    eval_dir = dataset_root / args.eval_config
    if not train_dir.is_dir() or not eval_dir.is_dir():
        raise FileNotFoundError(
            f"Expected both {args.train_config} and {args.eval_config} under {dataset_root}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    print(f"Training on {torch.cuda.get_device_name(device)}", flush=True)

    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
        args.base_model,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False

    heads = nn.ModuleDict()
    existing_rule_levels = set(RULE_LEVEL_LABELS).intersection(
        model.level_to_lm_head
    )
    if existing_rule_levels and existing_rule_levels != set(RULE_LEVEL_LABELS):
        raise ValueError(
            "The base model must contain both Tajweed rule heads or neither"
        )
    for level, class_names in RULE_LEVEL_LABELS.items():
        vocab_size = len(class_names) + 1
        if level in existing_rule_levels:
            head = model.level_to_lm_head[level].to(
                device=device, dtype=torch.float32
            )
        else:
            model.config.level_to_vocab_size[level] = vocab_size
            model.config.level_to_loss_weight[level] = 0.0
            head = nn.Linear(model.config.output_hidden_size, vocab_size).to(
                device=device, dtype=torch.float32
            )
            model.level_to_lm_head[level] = head
        if head.out_features != vocab_size:
            raise ValueError(
                f"Unexpected vocabulary size for {level}: {head.out_features}"
            )
        heads[level] = head

    optimizer = torch.optim.AdamW(
        [parameter for head in heads.values() for parameter in head.parameters()],
        lr=args.learning_rate,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.base_model, local_files_only=True
    )

    metrics = {
        "base_model": args.base_model,
        "train_config": args.train_config,
        "eval_config": args.eval_config,
        "device": torch.cuda.get_device_name(device),
        "epochs": [],
        "rule_labels": {
            level: list(labels) for level, labels in RULE_LEVEL_LABELS.items()
        },
    }

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            config_dir=train_dir,
            training=True,
            epoch=epoch,
            model=model,
            heads=heads,
            optimizer=optimizer,
            feature_extractor=feature_extractor,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            sample_limit=args.limit_train_samples,
            batch_limit=args.max_train_batches,
            max_attention_cost=args.max_attention_cost,
        )
        model.save_pretrained(output_dir)
        feature_extractor.save_pretrained(output_dir)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            args.base_model,
            local_files_only=True,
            pad_token=PAD_TOKEN,
            target_lang="phonemes",
        )
        tokenizer.save_pretrained(output_dir)

        eval_metrics = run_epoch(
            config_dir=eval_dir,
            training=False,
            epoch=epoch,
            model=model,
            heads=heads,
            optimizer=optimizer,
            feature_extractor=feature_extractor,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            sample_limit=args.limit_eval_samples,
            batch_limit=args.max_eval_batches,
            max_attention_cost=args.max_attention_cost,
        )
        metrics["epochs"].append({"train": train_metrics, "eval": eval_metrics})
        (output_dir / "tajweed_training_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        print(
            f"epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"noon_exact={eval_metrics['noon_tanween_sequence_accuracy']:.3f} "
            f"meem_exact={eval_metrics['meem_sakinah_sequence_accuracy']:.3f}",
            flush=True,
        )

    print(f"Saved trained model and metrics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
