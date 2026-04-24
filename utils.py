import argparse
import csv
import json
import socket
import time
from pathlib import Path

import jiwer
from loguru import logger

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"
CSV_COLUMNS = [
    "hostname",
    "dataset",
    "dataset_size",
    "model",
    "device",
    "batch_size",
    "audio_dur",
    "infer_dur",
    "rtf",
    "cer",
    "mer",
    "wer",
    "exp_epoch",
]


def log_args(args: argparse.Namespace):
    for key, value in vars(args).items():
        logger.info(f"ARGS: {key}: {value}")


def write_result(
    dataset: str,
    dataset_size: int,
    model: str,
    device: str,
    batch_size: int,
    audio_dur: float,
    infer_dur: float,
    refs: list[str],
    hyps: list[str],
):

    assert len(refs) == len(hyps) > 0

    rtf = infer_dur / audio_dur
    cer = jiwer.cer(refs, hyps)
    mer = jiwer.mer(refs, hyps)
    wer = jiwer.wer(refs, hyps)

    for i, (ref, hyp) in enumerate(zip(refs[:10], hyps[:10])):
        logger.info(f"Example {i + 1}:")
        logger.info(f"  REF: {ref}")
        logger.info(f"  HYP: {hyp}")

    RESULTS_DIR.mkdir(exist_ok=True)

    exp_epoch = int(time.time())

    json_path = RESULTS_DIR / f"{exp_epoch}.json"
    with open(json_path, "w") as f:
        json.dump({"refs": refs, "hyps": hyps}, f)

    write_header = not RESULTS_CSV.exists()

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        if write_header:
            writer.writeheader()

        hostname = socket.gethostname()

        logger.info(f"Hostname: {hostname}")
        logger.info(f"Dataset: {dataset} (size: {dataset_size})")
        logger.info(f"Model: {model}")
        logger.info(f"Device: {device}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Audio duration: {audio_dur:.2f} seconds")
        logger.info(f"Inference duration: {infer_dur:.2f} seconds")
        logger.info(f"RTF: {rtf:.6f}")
        logger.info(f"CER: {cer:.6f}")
        logger.info(f"MER: {mer:.6f}")
        logger.info(f"WER: {wer:.6f}")
        logger.info(f"Experiment epoch: {exp_epoch}")

        writer.writerow(
            {
                "hostname": hostname,
                "dataset": dataset,
                "dataset_size": dataset_size,
                "model": model,
                "device": device,
                "batch_size": batch_size,
                "audio_dur": round(audio_dur, 2),
                "infer_dur": round(infer_dur, 2),
                "rtf": round(rtf, 4),
                "cer": round(cer, 4),
                "mer": round(mer, 4),
                "wer": round(wer, 4),
                "exp_epoch": exp_epoch,
            }
        )
