# KeplerGPT Training Log

This document serves as a persistent record of all training runs, model configurations, hyperparameters, and generation metrics.

---

## Run 1: Smoke Test Run (1,000 Steps)
* **Date**: 2026-05-24 23:28
* **Model Configuration**: `deep_384d_4l` preset (384 embedding dim, 6 heads, 4 layers, context context=128)
* **Vocab Size**: 4,096 tokens
* **Dataset**: FineWeb 100MB (549,278 documents)
* **Batch Size / Context**: B=1, T=128
* **Hyperparameters**:
  * **Learning Rate**: `1.50e-05` (constant)
  * **Total Steps**: 1,000
* **Performance / Resource Usage**:
  * **Total Duration**: 22m 19s (Avg step: 1119.8ms)
  * **Throughput**: ~125 tok/s
  * **VRAM Peak**: 1093 MB
* **Loss & Convergence**:
  * **Initial Loss**: `8.3150` (PPL: 4084.72)
  * **Final Loss**: `7.5461` (PPL: 1893.43)
  * **Min Loss**: `7.4728` (PPL: 1759.50)
  * **Last Val Loss**: `7.6971` (PPL: 2201.93)
* **Generation Quality (1,000 steps)**:
  * *Greedy decode ('the')*: `'the  '`
  * *Sampled decode (temp=0.8, top_p=0.9, rep_pen=1.15)*: `'theじセapplicationAfterપ採make平openforceĐ\rmoney่埃motherjob址leadםे理瑜於めquickly친ếᄅovertownwellrequirementscontentvarietyapp플voice緊寫床My念500จ發બ顾stoo'`
  * *Logits step 1*: Top token is space `' '` (logit: 0.59), followed by empty `''` (logit: 0.40) and letters `'e'`, `'i'`, `'a'`. Coherent words have not yet formed.

### Logs
```text
[2026-05-24 23:28:16] [KepleGPT] [    INFO] [train][cuda] step=993/1000 loss=7.4856 avg_loss=8.0049 val_loss=7.7042 val_ppl=2217.73 lr=1.50e-05 ppl=1782.23 grad_norm=1.5118 elapsed=21m20s eta=0m07s step_ms=1020.30 avg_step_ms=1120.51 tok/s=125.5 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-24 23:28:17] [KepleGPT] [    INFO] [train][cuda] step=994/1000 loss=7.6015 avg_loss=8.0045 val_loss=7.7032 val_ppl=2215.34 lr=1.50e-05 ppl=2001.15 grad_norm=1.0764 elapsed=21m21s eta=0m06s step_ms=1015.18 avg_step_ms=1120.40 tok/s=126.1 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-24 23:28:18] [KepleGPT] [    INFO] [train][cuda] step=995/1000 loss=7.4728 avg_loss=8.0040 val_loss=7.7021 val_ppl=2213.03 lr=1.50e-05 ppl=1759.50 grad_norm=1.5685 elapsed=21m22s eta=0m05s step_ms=1015.21 avg_step_ms=1120.30 tok/s=126.1 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-24 23:28:19] [KepleGPT] [    INFO] [train][cuda] step=996/1000 loss=7.4753 avg_loss=8.0035 val_loss=7.7011 val_ppl=2210.77 lr=1.50e-05 ppl=1764.00 grad_norm=1.5292 elapsed=21m23s eta=0m04s step_ms=1017.60 avg_step_ms=1120.19 tok/s=125.8 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-24 23:28:20] [KepleGPT] [    INFO] [train][cuda] step=997/1000 loss=7.4917 avg_loss=8.0030 val_loss=7.7001 val_ppl=2208.54 lr=1.50e-05 ppl=1793.06 grad_norm=1.4591 elapsed=21m24s eta=0m03s step_ms=1014.03 avg_step_ms=1120.09 tok/s=126.2 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-24 23:28:21] [KepleGPT] [    INFO] [train][cuda] step=998/1000 loss=7.7577 avg_loss=8.0027 val_loss=7.6991 val_ppl=2206.32 lr=1.50e-05 ppl=2339.59 grad_norm=0.8144 elapsed=21m25s eta=0m02s step_ms=1022.09 avg_step_ms=1119.99 tok/s=125.2 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-24 23:28:22] [KepleGPT] [    INFO] [train][cuda] step=999/1000 loss=7.5700 avg_loss=8.0023 val_loss=7.6981 val_ppl=2204.13 lr=1.50e-05 ppl=1939.05 grad_norm=1.1661 elapsed=21m26s eta=0m01s step_ms=1040.48 avg_step_ms=1119.91 tok/s=123.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-24 23:28:23] [KepleGPT] [    INFO] [train][cuda] step=1000/1000 loss=7.5461 avg_loss=8.0018 val_loss=7.6971 val_ppl=2201.93 lr=1.50e-05 ppl=1893.43 grad_norm=1.1967 elapsed=21m27s eta=0m00s step_ms=1022.41 avg_step_ms=1119.81 tok/s=125.2 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-24 23:28:23] [KepleGPT] [    INFO] [PROBE@100%_step1000] Saving checkpoint to output/checkpoints/gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step1000.p100.npz...
[2026-05-24 23:28:23] [KepleGPT] [    INFO] [SAVE] Archiving parameter weights to disk checkpoint: output/checkpoints/gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step1000.p100.npz
[2026-05-24 23:28:25] [KepleGPT] [    INFO] [OK] Checkpoint write operation finalized successfully.
[2026-05-24 23:28:25] [KepleGPT] [    INFO] [PROBE@100%_step1000] Checkpoint saved
[2026-05-24 23:28:56] [KepleGPT] [    INFO] Loaded shared tokenizer corpus: 521,814 documents (fineweb)
[2026-05-24 23:28:56] [KepleGPT] [    INFO] [CFG] Using checkpoint config: vocab=4096, ctx=128, embed=384, heads=6, layers=4, attention=strided
[2026-05-24 23:28:57] [KepleGPT] [    INFO] 📂 Hydrating VRAM configurations from disk checkpoint: output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step1000.p100.npz
[2026-05-24 23:28:57] [KepleGPT] [    INFO] ✅ Core parameters statefully hydrated. Model ready for instant execution.
[2026-05-24 23:29:15] [KepleGPT] [    INFO] 
Generation probes:
Checkpoint: output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step1000.p100.npz
Tokenizer roundtrip:
  - 'the' -> 'the'
  - ' the' -> ' the'
  - '.' -> '.'
  - 'hello world' -> 'hello world'
  - '\n\nline1\nline2\n' -> '\n\nline1\nline2\n'
Greedy decode ['the']: 'the  '
Sampled decode [temp=0.8, top_p=0.9, rep_pen=1.15, prompt='the']: 'theじセapplicationAfterપ採make平openforceĐ\rmoney่埃motherjob址leadםे理瑜於めquickly친ếᄅovertownwellrequirementscontentvarietyapp플voice緊寫床My念500จ發બ顾stoo'
Memorization prefix ['|Viewing Single Post From: Spoil']: '|Viewing Single Post From: Spoil '
Logits step 1:
   1. id=0 piece=' ' logit=0.593848
   2. id=4095 piece='' logit=0.408637
   3. id=1 piece='e' logit=0.355558
   4. id=5 piece='i' logit=0.340856
   5. id=3 piece='a' logit=0.334482
   6. id=4 piece='o' logit=0.330078
   7. id=8 piece='r' logit=0.329261
   8. id=2 piece='t' logit=0.310556
   9. id=7 piece='s' logit=0.302319
  10. id=6 piece='n' logit=0.292802
[2026-05-24 23:29:15] [KepleGPT] [    INFO] =========================================================================
[2026-05-24 23:29:15] [KepleGPT] [    INFO] [OK] Training complete!
[2026-05-24 23:29:15] [KepleGPT] [    INFO] ================================================================================
[2026-05-24 23:29:15] [KepleGPT] [    INFO] TRAINING COMPLETE - FINAL STATISTICS
[2026-05-24 23:29:15] [KepleGPT] [    INFO] ================================================================================
[2026-05-24 23:29:15] [KepleGPT] [    INFO] 
Loss Statistics:
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Initial loss:     8.315008
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Final loss:       7.546144
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Min loss:         7.472783
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Max loss:         8.319295
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Mean loss:        8.001820
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Loss improvement: 0.768864 (9.2%)
[2026-05-24 23:29:15] [KepleGPT] [    INFO] 
Perplexity Statistics:
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Initial PPL:      4084.72
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Final PPL:        1893.43
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Best PPL:         1759.50
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Mean PPL:         3050.06
[2026-05-24 23:29:15] [KepleGPT] [    INFO] 
Validation Stats:
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Last val loss:    7.697087
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Last val PPL:     2201.93
[2026-05-24 23:29:15] [KepleGPT] [    INFO] 
Timing Statistics:
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Total time:       1339.9s (0:22:19)
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Avg step time:    1119.8ms
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Min step time:    1006.7ms
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Max step time:    3190.8ms
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Steps/sec:        0.75
[2026-05-24 23:29:15] [KepleGPT] [    INFO] 
GPU Memory:
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Peak usage:       1093MB
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Current usage:    1093MB
[2026-05-24 23:29:15] [KepleGPT] [    INFO] ================================================================================
[2026-05-24 23:29:15] [KepleGPT] [    INFO] 
Goal Metrics:
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Target loss:      < 2.00
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Target PPL:       < 5.00
[2026-05-24 23:29:15] [KepleGPT] [    INFO]   Reached:          NO
[2026-05-24 23:29:15] [KepleGPT] [    INFO] Saving checkpoint to output/checkpoints/gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.npz...
[2026-05-24 23:29:15] [KepleGPT] [    INFO] [SAVE] Archiving parameter weights to disk checkpoint: output/checkpoints/gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.npz
[2026-05-24 23:29:18] [KepleGPT] [    INFO] [OK] Checkpoint write operation finalized successfully.
[2026-05-24 23:29:18] [KepleGPT] [    INFO] [OK] Checkpoint saved
[2026-05-24 23:29:18] [KepleGPT] [    INFO] [OK] Saved run config to output/last_run_config.json
[2026-05-24 23:29:18] [KepleGPT] [    INFO] Cleaning up GPU memory...
[2026-05-24 23:29:18] [KepleGPT] [    INFO] [OK] GPU memory freed
[2026-05-24 23:29:18] [KepleGPT] [    INFO] Generating from prompt: 'the'
[2026-05-24 23:29:18] [KepleGPT] [    INFO] Max tokens: 50
[2026-05-24 23:29:18] [KepleGPT] [    INFO] Checkpoint: output/checkpoints/gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.npz
[2026-05-24 23:29:18] [KepleGPT] [    INFO] [GEN] Loading checkpoint: output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.npz
[2026-05-24 23:29:55] [KepleGPT] [    INFO] Loaded shared tokenizer corpus: 549,278 documents (fineweb_100mb.txt)
[2026-05-24 23:29:55] [KepleGPT] [    INFO] [CFG] Using checkpoint config: vocab=4096, ctx=128, embed=384, heads=6, layers=4, attention=strided
[2026-05-24 23:29:55] [KepleGPT] [    INFO] 📂 Hydrating VRAM configurations from disk checkpoint: output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.npz
[2026-05-24 23:29:56] [KepleGPT] [    INFO] ✅ Core parameters statefully hydrated. Model ready for instant execution.
[2026-05-24 23:29:56] [KepleGPT] [    INFO] [OK] Checkpoint loaded
[2026-05-24 23:29:56] [KepleGPT] [    INFO] [GEN] Seed: 'the' -> 1 tokens
[2026-05-24 23:30:00] [KepleGPT] [    INFO] [OK] Generation complete: 50 generated tokens
[2026-05-24 23:30:00] [KepleGPT] [    INFO] 
[OK] Generated text:
the😛designedknowledge把́houseъrange治🚊目(getsВ言Γchoose배13hadtownincrease择effect功making.번need꼬UK🔴level목☑createdTHEも c裹🔔question🔔ÞalsoFree
[2026-05-24 23:30:32] [KepleGPT] [    INFO] Loaded shared tokenizer corpus: 521,814 documents (fineweb)
[2026-05-24 23:30:32] [KepleGPT] [    INFO] [CFG] Using checkpoint config: vocab=4096, ctx=128, embed=384, heads=6, layers=4, attention=strided
[2026-05-24 23:30:33] [KepleGPT] [    INFO] 📂 Hydrating VRAM configurations from disk checkpoint: output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.npz
[2026-05-24 23:30:33] [KepleGPT] [    INFO] ✅ Core parameters statefully hydrated. Model ready for instant execution.
[2026-05-24 23:30:52] [KepleGPT] [    INFO] 
Generation probes:
Checkpoint: output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.npz
Tokenizer roundtrip:
  - 'the' -> 'the'
  - ' the' -> ' the'
  - '.' -> '.'
  - 'hello world' -> 'hello world'
  - '\n\nline1\nline2\n' -> '\n\nline1\nline2\n'
Greedy decode ['the']: 'the  '
Sampled decode [temp=0.8, top_p=0.9, rep_pen=1.15, prompt='the']: 'theじセapplicationAfterપ採make平openforceĐ\rmoney่埃motherjob址leadםे理瑜於めquickly친ếᄅovertownwellrequirementscontentvarietyapp플voice緊寫床My念500จ發બ顾stoo'
Memorization prefix ['|Viewing Single Post From: Spoil']: '|Viewing Single Post From: Spoil '
Logits step 1:
   1. id=0 piece=' ' logit=0.593848
   2. id=4095 piece='' logit=0.408637
   3. id=1 piece='e' logit=0.355558
   5. id=5 piece='i' logit=0.340856
   6. id=3 piece='a' logit=0.334482
   7. id=4 piece='o' logit=0.330078
   8. id=8 piece='r' logit=0.329261
   9. id=2 piece='t' logit=0.310556
  10. id=7 piece='s' logit=0.302319
  11. id=6 piece='n' logit=0.292802
[2026-05-24 23:30:52] [KepleGPT] [    INFO] [OK] Saved run config to output/last_run_config.json
```
_________________________________________________________________________________________________
_________________________________________________________________________________________________
_________________________________________________________________________________________________
_________________________________________________________________________________________________



_________________________________________________________________________________________________
_________________________________________________________________________________________________
_________________________________________________________________________________________________

## Run 2: First Major Realworld Run After Updates (20000 Steps)
[INFO] Logging initialized | Level: INFO | File: output/logs\training_20260524_233745.log

=========================================================================
INTERACTIVE GPT TRAINING LAUNCHER
=========================================================================

[INFO] Found previous run configuration
Reuse last run's settings? [y/n] (default: n):

=========================================================================
STEP 1: SELECT DATASET
=========================================================================

Available datasets:
  1) Minimal       - 3 test sentences (quick training)
  2) FineWeb 100MB - Real web text (recommend for quality)
  3) Custom        - Load from data/*.txt file

Select dataset [1-3] (default: 2):
[INFO] [OK] Loaded FineWeb dataset: 549,278 documents

Dataset has 549,278 documents.
Limit to how many docs? (default: 5000, max: 549,278): max
[INFO]   [OK] Using all 549,278 documents

=========================================================================
STEP 2: MODEL ARCHITECTURE
=========================================================================

⚠️  GT730 KEPLER NOTE: still keep models to 1 layer for reliability.
[INFO] Preset parameter counts are approximate and assume char vocab ~256, ctx=64.
[INFO] Larger presets are available, but >1M params will be slow or may exceed VRAM on GT730.
[INFO] Preset 8 can run with either 1 layer or 4 layers.
[INFO] Preset 9 is the 384D, 4-layer alternative when the 896D wide model plateaus early.

Parameter-scale presets:
  1) MICRO   -   13.6K params |  16D,  1 heads, 1 layer | smoke tests
  2) TINY    -   33.2K params |  32D,  1 heads, 1 layer | quick debug runs
  3) SMALL   -   91.1K params |  64D,  2 heads, 1 layer | recommended GT730 baseline [RECOMMENDED]
  4) BASE    -  280.4K params | 128D,  4 heads, 1 layer | larger single-layer test
  5) LARGE   -  954.1K params | 256D,  8 heads, 1 layer | near 1M params
  6) XL      -   3.48M params | 512D,  8 heads, 1 layer | experimental on GT730
  7) XXL     -   5.33M params | 640D, 10 heads, 1 layer | very slow / experimental
  8) GIANT   -  10.22M params | 896D, 14 heads, 1 or 4 layers | ~10M params at 1L, optional 4L override
  9) DEEP    -   7.34M params | 384D,  6 heads, 4 layer | narrower 4L alternative to the wide giant preset

Select model [1-9] (default: 3): 9
[INFO] Model 9 is fixed at 4 layers to give the narrower embedding more sequential depth.

[OK] Selected DEEP
     Approx params: 7.34M (vocab~256, ctx=128)
     Embedding dim: 384
     Num heads: 6
     Num layers: 4

=========================================================================
STEP 3: TRAINING HYPERPARAMETERS
=========================================================================

Learning rate (2M-10M params: safe range 0.0005-0.005, experimental on GT730)
  Tip: If loss plateaus (not changing), increase LR
  Tip: If NaN occurs, decrease LR and use gradient clipping
Learning rate (default: 0.001): 0.0005
Training steps (default: 100): 20000
Sequence length T (default: 128): 

[OK] Training config:
     Learning rate: 0.0005
     Steps: 20000
     Sequence length: 128

=========================================================================
STEP 4: LOGGING & CHECKPOINT
=========================================================================

[OK] Auto-generated names from training setup:
     Format: training_<steps>steps_<lr>lr_ctx<context>_<timestamp>
     Log file: training_20000steps_0p0005lr_ctx128_deep_384d_4l_20260524_233809.log
     Checkpoint: output/checkpoints/gpt_20000steps_0p0005lr_ctx128_deep_384d_4l_20260524_233809.npz

=========================================================================
OPTIONAL CHECKPOINT INIT
=========================================================================
Leave blank to start from random weights.

Recent checkpoints:
  1) output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.npz
  2) output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step1000.p100.npz
  3) output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step750.p75.npz
  4) output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step500.p50.npz
  5) output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step250.p25.npz
Checkpoint path or recent index (blank = random init):
[OK] Using random initialization

=========================================================================
STEP 5: POST-TRAINING OPTIONS
=========================================================================

Test generation after training? [y/n] (default: y):
Generation prompt (default: 'the'): 
Max tokens to generate (default: 50): 

[OK] Generation config:
     Prompt: 'the'
     Max tokens: 50

=========================================================================
TRAINING CONFIGURATION SUMMARY
=========================================================================

Dataset:        fineweb
  Docs:         549,278

Model:          DEEP
  Params:       ~7.34M (selector estimate)
  Embedding:    384D
  Heads:        6
  Layers:       4
  Attention:    strided

Training:
  LR:           0.0005
  Steps:        20000
  Seq length:   128

Checkpoint:     output/checkpoints/gpt_20000steps_0p0005lr_ctx128_deep_384d_4l_20260524_233809.npz

Logging:        training_20000steps_0p0005lr_ctx128_deep_384d_4l_20260524_233809.log

Generation:
  Prompt:       'the'
  Max tokens:   50

Start training now? [y/n] (default: y): y

Preparing tokenizer and VRAM estimate...
Estimated tokenizer vocab: 4096

=========================================================================
VRAM ESTIMATION
=========================================================================

Estimated GPU Memory:
  Model weights:    ~39MB
  Training overhead: ~196MB
  Total needed:     ~235MB
  GT730 v2 available: ~3500MB (4GB DDR3 total)

[OK] Model should fit in VRAM

Starting training...

=========================================================================
STARTING TRAINING
=========================================================================

[INFO] Logging initialized | Level: INFO | File: output/logs\training_20000steps_0p0005lr_ctx128_deep_384d_4l_20260524_233809.log
[INFO] [OK] PyCUDA pooled allocator enabled
[INFO] Building tokenizer from 521,814 training documents...
[INFO] [OK] Shared tokenizer vocab built from 521,814 documents (fineweb)
[INFO] [OK] Vocab size: 4096
[INFO] [OK] Estimated actual parameter count: 10.29M
[INFO] [OK] Model config created
[INFO] [OK] Model instantiated
[INFO] Encoding corpus...




run failed due to lr error 
RE RUN WITH 0.000015

_________________________________________________________________________________________________

# RUN 3 (RESTARTED WITH LOWER LR)
(venv) PS C:\dev\llm gpu 5> python .\auto_train.py
[INFO] Logging initialized | Level: INFO | File: output/logs\training_20260524_234714.log

=========================================================================
INTERACTIVE GPT TRAINING LAUNCHER
=========================================================================

[INFO] Found previous run configuration
Reuse last run's settings? [y/n] (default: n):

=========================================================================
STEP 1: SELECT DATASET
=========================================================================

Available datasets:
  1) Minimal       - 3 test sentences (quick training)
  2) FineWeb 100MB - Real web text (recommend for quality)
  3) Custom        - Load from data/*.txt file

Select dataset [1-3] (default: 2):
[INFO] [OK] Loaded FineWeb dataset: 549,278 documents

Dataset has 549,278 documents.
Limit to how many docs? (default: 5000, max: 549,278): MAX
[INFO]   [OK] Using all 549,278 documents

=========================================================================
STEP 2: MODEL ARCHITECTURE
=========================================================================

⚠️  GT730 KEPLER NOTE: still keep models to 1 layer for reliability.
[INFO] Preset parameter counts are approximate and assume char vocab ~256, ctx=64.
[INFO] Larger presets are available, but >1M params will be slow or may exceed VRAM on GT730.
[INFO] Preset 8 can run with either 1 layer or 4 layers.
[INFO] Preset 9 is the 384D, 4-layer alternative when the 896D wide model plateaus early.

Parameter-scale presets:
  1) MICRO   -   13.6K params |  16D,  1 heads, 1 layer | smoke tests
  2) TINY    -   33.2K params |  32D,  1 heads, 1 layer | quick debug runs
  3) SMALL   -   91.1K params |  64D,  2 heads, 1 layer | recommended GT730 baseline [RECOMMENDED]
  4) BASE    -  280.4K params | 128D,  4 heads, 1 layer | larger single-layer test
  5) LARGE   -  954.1K params | 256D,  8 heads, 1 layer | near 1M params
  6) XL      -   3.48M params | 512D,  8 heads, 1 layer | experimental on GT730
  7) XXL     -   5.33M params | 640D, 10 heads, 1 layer | very slow / experimental
  8) GIANT   -  10.22M params | 896D, 14 heads, 1 or 4 layers | ~10M params at 1L, optional 4L override
  9) DEEP    -   7.34M params | 384D,  6 heads, 4 layer | narrower 4L alternative to the wide giant preset

Select model [1-9] (default: 3): 9
[INFO] Model 9 is fixed at 4 layers to give the narrower embedding more sequential depth.

[OK] Selected DEEP
     Approx params: 7.34M (vocab~256, ctx=128)
     Embedding dim: 384
     Num heads: 6
     Num layers: 4

=========================================================================
STEP 3: TRAINING HYPERPARAMETERS
=========================================================================

Learning rate (2M-10M params: safe range 0.0005-0.005, experimental on GT730)
  Tip: If loss plateaus (not changing), increase LR
  Tip: If NaN occurs, decrease LR and use gradient clipping
Learning rate (default: 0.001): 0.000015
Training steps (default: 100): 20000
Sequence length T (default: 128): 

[OK] Training config:
     Learning rate: 1.5e-05
     Steps: 20000
     Sequence length: 128

=========================================================================
STEP 4: LOGGING & CHECKPOINT
=========================================================================

[OK] Auto-generated names from training setup:
     Format: training_<steps>steps_<lr>lr_ctx<context>_<timestamp>
     Log file: training_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.log
     Checkpoint: output/checkpoints/gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.npz

=========================================================================
OPTIONAL CHECKPOINT INIT
=========================================================================
Leave blank to start from random weights.

Recent checkpoints:
  1) output\checkpoints\ERROR WITH LR DO NOT USE_gpt_20000steps_0p0005lr_ctx128_deep_384d_4l_20260524_233809.best.npz
  2) output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.npz
  3) output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step1000.p100.npz
  4) output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step750.p75.npz
  5) output\checkpoints\gpt_1000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_230426.step500.p50.npz
Checkpoint path or recent index (blank = random init):
[OK] Using random initialization

=========================================================================
STEP 5: POST-TRAINING OPTIONS
=========================================================================

Test generation after training? [y/n] (default: y):
Generation prompt (default: 'the'): 
Max tokens to generate (default: 50): 

[OK] Generation config:
     Prompt: 'the'
     Max tokens: 50

=========================================================================
TRAINING CONFIGURATION SUMMARY
=========================================================================

Dataset:        fineweb
  Docs:         549,278

Model:          DEEP
  Params:       ~7.34M (selector estimate)
  Embedding:    384D
  Heads:        6
  Layers:       4
  Attention:    strided

Training:
  LR:           1.5e-05
  Steps:        20000
  Seq length:   128

Checkpoint:     output/checkpoints/gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.npz

Logging:        training_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.log

Generation:
  Prompt:       'the'
  Max tokens:   50

Start training now? [y/n] (default: y):y

[2026-05-25 05:28:00] [KepleGPT] [    INFO] [train][cuda] step=19959/20000 loss=0.5851 avg_loss=2.9388 val_loss=2.9160 val_ppl=18.47 lr=1.50e-05 ppl=1.80 grad_norm=1.1092 elapsed=5h38m27s eta=0m40s step_ms=1000.06 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:01] [KepleGPT] [    INFO] [train][cuda] step=19960/20000 loss=3.2034 avg_loss=2.9388 val_loss=2.9157 val_ppl=18.46 lr=1.50e-05 ppl=24.62 grad_norm=2.6334 elapsed=5h38m28s eta=0m39s step_ms=999.64 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:02] [KepleGPT] [    INFO] [train][cuda] step=19961/20000 loss=0.3438 avg_loss=2.9387 val_loss=2.9155 val_ppl=18.46 lr=1.50e-05 ppl=1.41 grad_norm=1.0211 elapsed=5h38m29s eta=0m38s step_ms=994.23 avg_step_ms=1009.13 tok/s=128.7 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:03] [KepleGPT] [    INFO] [train][cuda] step=19962/20000 loss=0.8621 avg_loss=2.9386 val_loss=2.9156 val_ppl=18.46 lr=1.50e-05 ppl=2.37 grad_norm=1.0766 elapsed=5h38m30s eta=0m38s step_ms=997.53 avg_step_ms=1009.13 tok/s=128.3 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:04] [KepleGPT] [    INFO] [train][cuda] step=19963/20000 loss=1.9519 avg_loss=2.9385 val_loss=2.9156 val_ppl=18.46 lr=1.50e-05 ppl=7.04 grad_norm=2.9754 elapsed=5h38m31s eta=0m36s step_ms=1008.38 avg_step_ms=1009.13 tok/s=126.9 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:05] [KepleGPT] [    INFO] [train][cuda] step=19964/20000 loss=1.4811 avg_loss=2.9385 val_loss=2.9153 val_ppl=18.45 lr=1.50e-05 ppl=4.40 grad_norm=1.3665 elapsed=5h38m32s eta=0m35s step_ms=1000.18 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:06] [KepleGPT] [    INFO] [train][cuda] step=19965/20000 loss=1.2401 avg_loss=2.9384 val_loss=2.9146 val_ppl=18.44 lr=1.50e-05 ppl=3.46 grad_norm=1.5721 elapsed=5h38m33s eta=0m35s step_ms=999.90 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:07] [KepleGPT] [    INFO] [train][cuda] step=19966/20000 loss=1.1922 avg_loss=2.9383 val_loss=2.9142 val_ppl=18.43 lr=1.50e-05 ppl=3.29 grad_norm=1.8019 elapsed=5h38m34s eta=0m34s step_ms=1000.08 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:08] [KepleGPT] [    INFO] [train][cuda] step=19967/20000 loss=2.9234 avg_loss=2.9383 val_loss=2.9141 val_ppl=18.43 lr=1.50e-05 ppl=18.61 grad_norm=3.3650 elapsed=5h38m35s eta=0m33s step_ms=991.85 avg_step_ms=1009.13 tok/s=129.1 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:09] [KepleGPT] [    INFO] [train][cuda] step=19968/20000 loss=2.5338 avg_loss=2.9383 val_loss=2.9141 val_ppl=18.43 lr=1.50e-05 ppl=12.60 grad_norm=2.3610 elapsed=5h38m36s eta=0m31s step_ms=992.34 avg_step_ms=1009.13 tok/s=129.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:10] [KepleGPT] [    INFO] [train][cuda] step=19969/20000 loss=0.7864 avg_loss=2.9382 val_loss=2.9141 val_ppl=18.43 lr=1.50e-05 ppl=2.20 grad_norm=1.0372 elapsed=5h38m37s eta=0m30s step_ms=1000.13 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:11] [KepleGPT] [    INFO] [train][cuda] step=19970/20000 loss=1.1323 avg_loss=2.9381 val_loss=2.9142 val_ppl=18.43 lr=1.50e-05 ppl=3.10 grad_norm=1.7110 elapsed=5h38m38s eta=0m29s step_ms=999.90 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:12] [KepleGPT] [    INFO] [train][cuda] step=19971/20000 loss=2.9228 avg_loss=2.9381 val_loss=2.9143 val_ppl=18.44 lr=1.50e-05 ppl=18.59 grad_norm=2.0426 elapsed=5h38m39s eta=0m29s step_ms=1018.10 avg_step_ms=1009.13 tok/s=125.7 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:13] [KepleGPT] [    INFO] [train][cuda] step=19972/20000 loss=0.5989 avg_loss=2.9380 val_loss=2.9147 val_ppl=18.44 lr=1.50e-05 ppl=1.82 grad_norm=2.1810 elapsed=5h38m40s eta=0m28s step_ms=1000.10 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:14] [KepleGPT] [    INFO] [train][cuda] step=19973/20000 loss=2.8921 avg_loss=2.9380 val_loss=2.9144 val_ppl=18.44 lr=1.50e-05 ppl=18.03 grad_norm=3.9670 elapsed=5h38m41s eta=0m27s step_ms=999.77 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:15] [KepleGPT] [    INFO] [train][cuda] step=19974/20000 loss=1.8891 avg_loss=2.9379 val_loss=2.9141 val_ppl=18.43 lr=1.50e-05 ppl=6.61 grad_norm=1.3812 elapsed=5h38m42s eta=0m26s step_ms=1031.81 avg_step_ms=1009.13 tok/s=124.1 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:16] [KepleGPT] [    INFO] [train][cuda] step=19975/20000 loss=2.8186 avg_loss=2.9379 val_loss=2.9132 val_ppl=18.42 lr=1.50e-05 ppl=16.75 grad_norm=2.2323 elapsed=5h38m43s eta=0m25s step_ms=1000.28 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:17] [KepleGPT] [    INFO] [train][cuda] step=19976/20000 loss=0.8578 avg_loss=2.9378 val_loss=2.9126 val_ppl=18.40 lr=1.50e-05 ppl=2.36 grad_norm=1.5047 elapsed=5h38m44s eta=0m24s step_ms=999.93 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:18] [KepleGPT] [    INFO] [train][cuda] step=19977/20000 loss=3.1450 avg_loss=2.9378 val_loss=2.9120 val_ppl=18.39 lr=1.50e-05 ppl=23.22 grad_norm=2.6224 elapsed=5h38m45s eta=0m23s step_ms=1000.05 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:19] [KepleGPT] [    INFO] [train][cuda] step=19978/20000 loss=3.0058 avg_loss=2.9378 val_loss=2.9111 val_ppl=18.38 lr=1.50e-05 ppl=20.20 grad_norm=2.3412 elapsed=5h38m46s eta=0m22s step_ms=999.60 avg_step_ms=1009.13 tok/s=128.1 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:20] [KepleGPT] [    INFO] [train][cuda] step=19979/20000 loss=3.1549 avg_loss=2.9378 val_loss=2.9104 val_ppl=18.36 lr=1.50e-05 ppl=23.45 grad_norm=2.1551 elapsed=5h38m47s eta=0m21s step_ms=1006.48 avg_step_ms=1009.13 tok/s=127.2 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:21] [KepleGPT] [    INFO] [train][cuda] step=19980/20000 loss=3.1221 avg_loss=2.9378 val_loss=2.9101 val_ppl=18.36 lr=1.50e-05 ppl=22.69 grad_norm=3.3460 elapsed=5h38m48s eta=0m20s step_ms=1012.09 avg_step_ms=1009.13 tok/s=126.5 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:22] [KepleGPT] [    INFO] [train][cuda] step=19981/20000 loss=1.1912 avg_loss=2.9377 val_loss=2.9100 val_ppl=18.36 lr=1.50e-05 ppl=3.29 grad_norm=1.6082 elapsed=5h38m49s eta=0m19s step_ms=999.81 avg_step_ms=1009.13 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:23] [KepleGPT] [    INFO] [train][cuda] step=19982/20000 loss=2.9694 avg_loss=2.9377 val_loss=2.9104 val_ppl=18.36 lr=1.50e-05 ppl=19.48 grad_norm=3.0373 elapsed=5h38m50s eta=0m18s step_ms=1000.34 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:24] [KepleGPT] [    INFO] [train][cuda] step=19983/20000 loss=0.5426 avg_loss=2.9376 val_loss=2.9109 val_ppl=18.37 lr=1.50e-05 ppl=1.72 grad_norm=1.1803 elapsed=5h38m51s eta=0m17s step_ms=999.90 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:25] [KepleGPT] [    INFO] [train][cuda] step=19984/20000 loss=0.7740 avg_loss=2.9375 val_loss=2.9115 val_ppl=18.38 lr=1.50e-05 ppl=2.17 grad_norm=1.2688 elapsed=5h38m52s eta=0m16s step_ms=999.76 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:26] [KepleGPT] [    INFO] [train][cuda] step=19985/20000 loss=3.1618 avg_loss=2.9375 val_loss=2.9121 val_ppl=18.40 lr=1.50e-05 ppl=23.61 grad_norm=2.2393 elapsed=5h38m53s eta=0m15s step_ms=1000.10 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:27] [KepleGPT] [    INFO] [train][cuda] step=19986/20000 loss=2.8227 avg_loss=2.9375 val_loss=2.9127 val_ppl=18.41 lr=1.50e-05 ppl=16.82 grad_norm=1.7610 elapsed=5h38m54s eta=0m14s step_ms=1000.19 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:28] [KepleGPT] [    INFO] [train][cuda] step=19987/20000 loss=1.0385 avg_loss=2.9374 val_loss=2.9133 val_ppl=18.42 lr=1.50e-05 ppl=2.83 grad_norm=1.1601 elapsed=5h38m55s eta=0m13s step_ms=999.68 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:29] [KepleGPT] [    INFO] [train][cuda] step=19988/20000 loss=1.0612 avg_loss=2.9373 val_loss=2.9142 val_ppl=18.43 lr=1.50e-05 ppl=2.89 grad_norm=1.3390 elapsed=5h38m56s eta=0m12s step_ms=1000.16 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:30] [KepleGPT] [    INFO] [train][cuda] step=19989/20000 loss=0.5598 avg_loss=2.9372 val_loss=2.9149 val_ppl=18.45 lr=1.50e-05 ppl=1.75 grad_norm=1.1090 elapsed=5h38m57s eta=0m11s step_ms=999.83 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:31] [KepleGPT] [    INFO] [train][cuda] step=19990/20000 loss=2.1569 avg_loss=2.9372 val_loss=2.9156 val_ppl=18.46 lr=1.50e-05 ppl=8.64 grad_norm=1.7148 elapsed=5h38m58s eta=0m09s step_ms=1000.06 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:32] [KepleGPT] [    INFO] [train][cuda] step=19991/20000 loss=1.0362 avg_loss=2.9371 val_loss=2.9161 val_ppl=18.47 lr=1.50e-05 ppl=2.82 grad_norm=1.5808 elapsed=5h38m59s eta=0m09s step_ms=1000.09 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:33] [KepleGPT] [    INFO] [train][cuda] step=19992/20000 loss=0.9041 avg_loss=2.9370 val_loss=2.9166 val_ppl=18.48 lr=1.50e-05 ppl=2.47 grad_norm=1.0822 elapsed=5h39m00s eta=0m07s step_ms=1000.10 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:34] [KepleGPT] [    INFO] [train][cuda] step=19993/20000 loss=0.6898 avg_loss=2.9369 val_loss=2.9171 val_ppl=18.49 lr=1.50e-05 ppl=1.99 grad_norm=0.9618 elapsed=5h39m01s eta=0m06s step_ms=999.94 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:35] [KepleGPT] [    INFO] [train][cuda] step=19994/20000 loss=1.3624 avg_loss=2.9368 val_loss=2.9176 val_ppl=18.50 lr=1.50e-05 ppl=3.91 grad_norm=1.1848 elapsed=5h39m02s eta=0m06s step_ms=1000.05 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:36] [KepleGPT] [    INFO] [train][cuda] step=19995/20000 loss=1.9507 avg_loss=2.9367 val_loss=2.9182 val_ppl=18.51 lr=1.50e-05 ppl=7.03 grad_norm=1.4366 elapsed=5h39m03s eta=0m05s step_ms=999.94 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:37] [KepleGPT] [    INFO] [train][cuda] step=19996/20000 loss=0.4526 avg_loss=2.9366 val_loss=2.9189 val_ppl=18.52 lr=1.50e-05 ppl=1.57 grad_norm=0.9212 elapsed=5h39m04s eta=0m04s step_ms=1000.20 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:38] [KepleGPT] [    INFO] [train][cuda] step=19997/20000 loss=3.0627 avg_loss=2.9366 val_loss=2.9183 val_ppl=18.51 lr=1.50e-05 ppl=21.39 grad_norm=4.6671 elapsed=5h39m05s eta=0m03s step_ms=1000.02 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:39] [KepleGPT] [    INFO] [train][cuda] step=19998/20000 loss=3.0303 avg_loss=2.9366 val_loss=2.9178 val_ppl=18.50 lr=1.50e-05 ppl=20.70 grad_norm=2.1384 elapsed=5h39m06s eta=0m02s step_ms=999.85 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:40] [KepleGPT] [    INFO] [train][cuda] step=19999/20000 loss=2.0901 avg_loss=2.9366 val_loss=2.9166 val_ppl=18.48 lr=1.50e-05 ppl=8.09 grad_norm=1.8626 elapsed=5h39m07s eta=0m01s step_ms=1000.03 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:41] [KepleGPT] [    INFO] [train][cuda] step=20000/20000 loss=2.9140 avg_loss=2.9366 val_loss=2.9150 val_ppl=18.45 lr=1.50e-05 ppl=18.43 grad_norm=2.8580 elapsed=5h39m08s eta=0m00s step_ms=1000.12 avg_step_ms=1009.12 tok/s=128.0 pool_used_mb=157.1 pool_total_mb=428.3 device_used_mb=1093.4
[2026-05-25 05:28:41] [KepleGPT] [    INFO] [PROBE@100%_step20000] Saving checkpoint to output/checkpoints/gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.step20000.p100.npz...
[2026-05-25 05:28:41] [KepleGPT] [    INFO] [SAVE] Archiving parameter weights to disk checkpoint: output/checkpoints/gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.step20000.p100.npz
[2026-05-25 05:28:43] [KepleGPT] [    INFO] [OK] Checkpoint write operation finalized successfully.
[2026-05-25 05:28:43] [KepleGPT] [    INFO] [PROBE@100%_step20000] Checkpoint saved
[2026-05-25 05:29:11] [KepleGPT] [    INFO] Loaded shared tokenizer corpus: 521,814 documents (fineweb)
[2026-05-25 05:29:11] [KepleGPT] [    INFO] [CFG] Using checkpoint config: vocab=4096, ctx=128, embed=384, heads=6, layers=4, attention=strided
[2026-05-25 05:29:12] [KepleGPT] [    INFO] 📂 Hydrating VRAM configurations from disk checkpoint: output\checkpoints\gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.step20000.p100.npz
[2026-05-25 05:29:12] [KepleGPT] [    INFO] ✅ Core parameters statefully hydrated. Model ready for instant execution.
[2026-05-25 05:29:27] [KepleGPT] [    INFO] 
Generation probes:
Checkpoint: output\checkpoints\gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.step20000.p100.npz
Tokenizer roundtrip:
  - 'the' -> 'the'
  - ' the' -> ' the'
  - '.' -> '.'
  - 'hello world' -> 'hello world'
  - '\n\nline1\nline2\n' -> '\n\nline1\nline2\n'
Greedy decode ['the']: 'the arer a a anterer the the the the the the the the the the the the the the the the '
Sampled decode [temp=0.8, top_p=0.9, rep_pen=1.15, prompt='the']: 'the ulintatig It stronce d Gamee and to in out would bring Cos to'
Memorization prefix ['|Viewing Single Post From: Spoil']: '|Viewing Single Post From: Spoilerer the the the the the the the the the the the the the the the the the the the the the the the'
Logits step 1:
   1. id=0 piece=' ' logit=4.892234
   2. id=25 piece='\n' logit=3.521250
   3. id=21 piece='.' logit=3.259093
   4. id=22 piece=',' logit=3.016595
   5. id=30 piece='-' logit=2.842687
   6. id=32 piece='0' logit=2.605275
   7. id=42 piece='’' logit=2.148524
   8. id=49 piece="'" logit=1.909479
   9. id=50 piece=':' logit=1.712459
  10. id=60 piece='9' logit=1.693165
[2026-05-25 05:29:27] [KepleGPT] [    INFO] =========================================================================
[2026-05-25 05:29:27] [KepleGPT] [    INFO] [OK] Training complete!
[2026-05-25 05:29:27] [KepleGPT] [    INFO] ================================================================================
[2026-05-25 05:29:27] [KepleGPT] [    INFO] TRAINING COMPLETE - FINAL STATISTICS
[2026-05-25 05:29:27] [KepleGPT] [    INFO] ================================================================================
[2026-05-25 05:29:27] [KepleGPT] [    INFO] 
Loss Statistics:
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Initial loss:     8.320167
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Final loss:       2.913984
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Min loss:         0.176825
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Max loss:         8.320167
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Mean loss:        2.936577
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Loss improvement: 5.406182 (65.0%)
[2026-05-25 05:29:27] [KepleGPT] [    INFO] 
Perplexity Statistics:
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Initial PPL:      4105.84
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Final PPL:        18.43
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Best PPL:         1.19
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Mean PPL:         231.80
[2026-05-25 05:29:27] [KepleGPT] [    INFO] 
Validation Stats:
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Last val loss:    2.915011
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Last val PPL:     18.45
[2026-05-25 05:29:27] [KepleGPT] [    INFO] 
Timing Statistics:
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Total time:       20395.3s (5:39:55)
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Avg step time:    1009.1ms
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Min step time:    980.0ms
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Max step time:    3723.0ms
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Steps/sec:        0.98
[2026-05-25 05:29:27] [KepleGPT] [    INFO] 
GPU Memory:
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Peak usage:       1093MB
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Current usage:    1093MB
[2026-05-25 05:29:27] [KepleGPT] [    INFO] ================================================================================
[2026-05-25 05:29:27] [KepleGPT] [    INFO] 
Goal Metrics:
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Target loss:      < 2.00
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Target PPL:       < 5.00
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Reached:          YES at step 15573
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Best loss:        0.176825
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Best PPL:         1.19
[2026-05-25 05:29:27] [KepleGPT] [    INFO]   Best checkpoint:  output/checkpoints/gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.best.npz
[2026-05-25 05:29:27] [KepleGPT] [    INFO] Saving checkpoint to output/checkpoints/gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.npz...
[2026-05-25 05:29:27] [KepleGPT] [    INFO] [SAVE] Archiving parameter weights to disk checkpoint: output/checkpoints/gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.npz
[2026-05-25 05:29:30] [KepleGPT] [    INFO] [OK] Checkpoint write operation finalized successfully.
[2026-05-25 05:29:30] [KepleGPT] [    INFO] [OK] Checkpoint saved
[2026-05-25 05:29:30] [KepleGPT] [    INFO] [OK] Saved run config to output/last_run_config.json
[2026-05-25 05:29:30] [KepleGPT] [    INFO] Cleaning up GPU memory...
[2026-05-25 05:29:30] [KepleGPT] [    INFO] [OK] GPU memory freed
[2026-05-25 05:29:30] [KepleGPT] [    INFO] Generating from prompt: 'the'
[2026-05-25 05:29:30] [KepleGPT] [    INFO] Max tokens: 50
[2026-05-25 05:29:30] [KepleGPT] [    INFO] Checkpoint: output/checkpoints/gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.best.npz
[2026-05-25 05:29:30] [KepleGPT] [    INFO] [GEN] Loading checkpoint: output\checkpoints\gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.best.npz
[2026-05-25 05:30:01] [KepleGPT] [    INFO] Loaded shared tokenizer corpus: 549,278 documents (fineweb_100mb.txt)
[2026-05-25 05:30:01] [KepleGPT] [    INFO] [CFG] Using checkpoint config: vocab=4096, ctx=128, embed=384, heads=6, layers=4, attention=strided
[2026-05-25 05:30:02] [KepleGPT] [    INFO] 📂 Hydrating VRAM configurations from disk checkpoint: output\checkpoints\gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.best.npz
[2026-05-25 05:30:02] [KepleGPT] [    INFO] ✅ Core parameters statefully hydrated. Model ready for instant execution.
[2026-05-25 05:30:02] [KepleGPT] [    INFO] [OK] Checkpoint loaded
[2026-05-25 05:30:02] [KepleGPT] [    INFO] [GEN] Seed: 'the' -> 1 tokens
[2026-05-25 05:30:06] [KepleGPT] [    INFO] [OK] Generation complete: 50 generated tokens
[2026-05-25 05:30:06] [KepleGPT] [    INFO] 
[OK] Generated text:
the aseting cerd lis alen, 💔 mesuterer 📦 💔 💘 💥 ngha 💡
[2026-05-25 05:30:34] [KepleGPT] [    INFO] Loaded shared tokenizer corpus: 521,814 documents (fineweb)
[2026-05-25 05:30:34] [KepleGPT] [    INFO] [CFG] Using checkpoint config: vocab=4096, ctx=128, embed=384, heads=6, layers=4, attention=strided
[2026-05-25 05:30:34] [KepleGPT] [    INFO] 📂 Hydrating VRAM configurations from disk checkpoint: output\checkpoints\gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.best.npz
[2026-05-25 05:30:34] [KepleGPT] [    INFO] ✅ Core parameters statefully hydrated. Model ready for instant execution.
[2026-05-25 05:30:49] [KepleGPT] [    INFO] 
Generation probes:
Checkpoint: output\checkpoints\gpt_20000steps_1p5e-05lr_ctx128_deep_384d_4l_20260524_234748.best.npz
Tokenizer roundtrip:
  - 'the' -> 'the'
  - ' the' -> ' the'
  - '.' -> '.'
  - 'hello world' -> 'hello world'
  - '\n\nline1\nline2\n' -> '\n\nline1\nline2\n'
Greedy decode ['the']: 'the are the the the the the the the the the the the the the the the the the the the the the the the'
Sampled decode [temp=0.8, top_p=0.9, rep_pen=1.15, prompt='the']: 'the fister to cul 14\n'
Memorization prefix ['|Viewing Single Post From: Spoil']: '|Viewing Single Post From: Spoilererer the the the the the the the the the the the the the the the the the the the the the the'
Logits step 1:
   1. id=0 piece=' ' logit=5.379583
   2. id=25 piece='\n' logit=3.643866
   3. id=21 piece='.' logit=3.352753
   4. id=22 piece=',' logit=3.118808
   5. id=30 piece='-' logit=2.555902
   6. id=32 piece='0' logit=1.843201
   7. id=4095 piece='' logit=1.753486
   8. id=42 piece='’' logit=1.645343
   9. id=49 piece="'" logit=1.430783
  10. id=7 piece='s' logit=1.353934
[2026-05-25 05:30:49] [KepleGPT] [    INFO] [OK] Saved run config to output/last_run_config.json
