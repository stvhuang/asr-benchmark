import argparse
from time import perf_counter

import mlx_whisper
import numpy as np
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm, trange

from utils import log_args, write_result

DATASET = "MERaLiON/Multitask-National-Speech-Corpus-v1"
DEVICE = "mlx"


def transcribe(
    audio_array: np.ndarray,
    model: str,
) -> str:
    result = mlx_whisper.transcribe(
        audio_array,
        path_or_hf_repo=model,
        language="en",
    )

    return result["text"].strip()


def warmup(model: str, duration: int, runs: int):
    logger.info(f"Warmup: duration={duration}s, runs={runs}")

    dummy = np.zeros(
        duration * 16_000,
        dtype=np.float32,
    )

    for _ in trange(runs, desc="Warmup"):
        transcribe(audio_array=dummy, model=model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-large-v3-mlx",
        type=str,
    )
    parser.add_argument("--parts", default="1,2,3,4,5,6", type=str)
    parser.add_argument("--warmup-duration", default=20, type=int)
    parser.add_argument("--warmup-runs", default=5, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    log_args(args)

    logger.info(f"{DEVICE=}")

    part_list = [f"ASR-PART{int(p.strip())}-Test" for p in args.parts.split(",")]
    logger.info(f"{part_list=}")

    for part in part_list:
        warmup(model=args.model, duration=args.warmup_duration, runs=args.warmup_runs)

        dataset = load_dataset(DATASET, data_dir=part)["train"]
        if args.debug:
            dataset = dataset.select(range(min(20, len(dataset))))

        refs = []
        hyps = []
        total_audio_dur = 0.0

        t0 = perf_counter()

        for sample in tqdm(dataset, desc=part):
            refs.append(sample["answer"])

            audio = sample["context"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            sampling_rate = audio["sampling_rate"]

            total_audio_dur += len(audio_array) / sampling_rate

            hypothesis = transcribe(audio_array=audio_array, model=args.model)
            hyps.append(hypothesis)

        total_infer_dur = perf_counter() - t0

        write_result(
            dataset=f"{DATASET}/{part}" + ("_debug" if args.debug else ""),
            dataset_size=len(dataset),
            model=args.model,
            device=DEVICE,
            batch_size=1,
            audio_dur=total_audio_dur,
            infer_dur=total_infer_dur,
            refs=refs,
            hyps=hyps,
        )


if __name__ == "__main__":
    main()
