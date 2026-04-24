import argparse
from time import perf_counter

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm, trange
from transformers import pipeline

from utils import log_args, write_result

DATASET = "MERaLiON/Multitask-National-Speech-Corpus-v1"
DEVICE = "cuda:0"


def build_pipeline(model):
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        device=DEVICE,
        dtype=torch.float16,
    )


def transcribe_batch(pipe, audio_inputs: list[dict]) -> list[str]:
    results = pipe(audio_inputs, batch_size=len(audio_inputs))

    return [r["text"].strip() for r in results]


def warmup(pipe, duration: int, runs: int):
    logger.info(f"Warmup: duration={duration}s, runs={runs}")

    dummy = np.zeros(duration * 16_000, dtype=np.float32)

    for _ in trange(runs, desc="Warmup"):
        pipe([{"raw": dummy, "sampling_rate": 16_000}], batch_size=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-large-v3", type=str)
    parser.add_argument("--parts", default="1,2,3,4,5,6", type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--warmup-duration", default=20, type=int)
    parser.add_argument("--warmup-runs", default=5, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    log_args(args)

    logger.info(f"{DEVICE=}")

    pipe = build_pipeline(model=args.model)

    part_list = [f"ASR-PART{int(p.strip())}-Test" for p in args.parts.split(",")]
    logger.info(f"{part_list=}")

    for part in part_list:
        warmup(pipe, duration=args.warmup_duration, runs=args.warmup_runs)

        dataset = load_dataset(DATASET, data_dir=part)["train"]
        if args.debug:
            dataset = dataset.select(range(min(20, len(dataset))))

        refs = []
        hyps = []
        inputs = []
        total_audio_dur = 0.0

        for sample in tqdm(dataset, desc=f"{part} (loading)"):
            audio = sample["context"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            sampling_rate = audio["sampling_rate"]

            total_audio_dur += len(audio_array) / sampling_rate
            refs.append(sample["answer"])
            inputs.append({"raw": audio_array, "sampling_rate": sampling_rate})

        t0 = perf_counter()

        for i in tqdm(
            range(0, len(inputs), args.batch_size), desc=f"{part} (inference)"
        ):
            batch = inputs[i : i + args.batch_size]
            hyps.extend(transcribe_batch(pipe, batch))

        total_infer_dur = perf_counter() - t0

        write_result(
            dataset=f"{DATASET}/{part}" + ("_debug" if args.debug else ""),
            dataset_size=len(dataset),
            model=args.model,
            device=DEVICE,
            batch_size=args.batch_size,
            audio_dur=total_audio_dur,
            infer_dur=total_infer_dur,
            refs=refs,
            hyps=hyps,
        )


if __name__ == "__main__":
    main()
