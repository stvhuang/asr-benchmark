#!/usr/bin/env sh

# mlx-community
for model in mlx-community/whisper-large-v3-mlx mlx-community/whisper-large-v3-turbo; do
  for i in {1..5}; do
    printf "$model (iteration $i)\n"
    python ./run_mnscv1_whisper_mlx.py --model $model
  done
done

# stvhuang
for model in stvhuang/whisper-large-v3_mlx stvhuang/whisper-large-v3-turbo_mlx; do
  for dtype in float16 float32; do
    model_dtype=${model}_${dtype}
    for i in {1..5}; do
      printf "$model_dtype (iteration $i)\n"
      python ./run_mnscv1_whisper_mlx.py --model $model_dtype
    done
  done
done

# stvhuang (quantized)
for model in stvhuang/whisper-large-v3_mlx stvhuang/whisper-large-v3-turbo_mlx; do
  for dtype in float16 float32; do
    for qbits in 8 4; do
      model_dtype_qbits=${model}_${dtype}_q${qbits}
      for i in {1..5}; do
        printf "$model_dtype_qbits (iteration $i)\n"
        python ./run_mnscv1_whisper_mlx.py --model $model_dtype_qbits
      done
    done
  done
done
