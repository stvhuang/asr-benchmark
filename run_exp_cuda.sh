#!/usr/bin/env sh

for model in openai/whisper-large-v3 openai/whisper-large-v3-turbo; do
  for batch_size in 1 4 16 64 128; do
    for i in {1..5}; do
      printf "$model (iteration $i)\n"
      ./.venv/bin/python ./run_mnscv1_whisper_cuda.py --model $model --batch_size $batch_size
    done
  done
done
