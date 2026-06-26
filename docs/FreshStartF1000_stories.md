Stage 1 -> Stage 2 Handoff:
[INFO]   Stage 1 avg loss:    2.7471 (target <= 3.00, strong <= 2.90)
[INFO]   Probe markers ready: YES
[INFO]   Probe readable:      YES
[INFO]   Probe trend ok:      YES
[INFO]   story bad_samples=0: NO
[INFO]   Stage 2 ready:       NO
[INFO] Saving checkpoint to output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.npz...
[INFO] [SAVE] Archiving parameter weights to disk checkpoint: output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.npz
[INFO] [OK] Checkpoint write operation finalized successfully.
[INFO] [OK] Checkpoint saved
[INFO] [OK] Saved run config to output/last_run_config.json
[INFO] Cleaning up GPU memory...
[INFO] [OK] GPU memory freed

=========================================================================
TESTING GENERATION
=========================================================================

[INFO] Generating from prompt: 'Once'
[INFO] Max tokens: 128
[INFO] Checkpoint: output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz   
[INFO] [GEN] Loading checkpoint: output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz
[INFO] Loaded shared tokenizer corpus: 9,500 documents (fineweb)
[INFO] [CFG] Using checkpoint config: vocab=4096, ctx=128, embed=384, heads=6, layers=4, attention=strided
[INFO] 📂 Hydrating VRAM configurations from disk checkpoint: output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz
[INFO] ✅ Core parameters statefully hydrated. Model ready for instant execution.
[INFO] [OK] Checkpoint loaded
[INFO] [GEN] Seed: 'Once' -> 1 tokens
[INFO] [OK] Generation complete: 128 generated tokens
[INFO] 
[OK] Generated text:
Once to a very was and,
They.
the the the"
" was a. her,.
a of.
 her!"
 The,
't. the to that!"
was smiled.
, and!
"
and as. girl.
was and the,


✨ Generated: Once to a very was and,
They.
the the the"
" was a. her,.
a of.
 her!"
 The,
't. the to that!"
was ...
[INFO] Loaded shared tokenizer corpus: 9,500 documents (fineweb)
[INFO] [CFG] Using checkpoint config: vocab=4096, ctx=128, embed=384, heads=6, layers=4, attention=strided
[INFO] 📂 Hydrating VRAM configurations from disk checkpoint: output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz
[INFO] ✅ Core parameters statefully hydrated. Model ready for instant execution.
[INFO] 
Generation probes:
Checkpoint: output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz
Tokenizer roundtrip:
  - 'the' -> 'the'
  - ' the' -> ' the'
  - '.' -> '.'
  - 'hello world' -> 'hello world'
  - '\n\nline1\nline2\n' -> '\n\nline1\nline2\n'
Greedy decode ['Once']: 'Once and the a was and.\n the and a and was.\n'
Sampled decode [temp=0.6, top_p=0.9, rep_pen=1.15, prompt='Once']: 'Once the her and the.\n. the was."\nser\'s!"\n to and, and to in. said.\nthe and.\n"\nthe!,\'t,\n They The and. went she and and \na.\n!"\nhe,. was\'! smiled.\ns to He and\n,'
Memorization prefix ['One day, a little girl named Lil']: 'One day, a little girl named Lils.\n and the and 
a was.\n'
Logits step 1:
   1. id=0 piece=' ' logit=8.623530
   2. id=16 piece='.' logit=5.192091
   3. id=23 piece=',' logit=4.611994
   4. id=4095 piece='' logit=4.238285
   5. id=27 piece='"' logit=4.137949
   6. id=31 piece="'" logit=3.406294
   7. id=33 piece='!' logit=3.331724
   8. id=26 piece='\n' logit=3.222285

Story bad_samples from latest audit (blank = unknown):                                     

=========================================================================
TRAINING COMPLETE!
=========================================================================

Checkpoint saved to: output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.npz      
Best checkpoint:    output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz  
Recommended:        output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz  

  Stage 1 avg loss:           2.7471 (target <= 3.00)
  Probe markers present:      yes
  Probe outputs readable:     yes
  Probe trend improving:      yes
  story bad_samples:          unknown (must be 0)
  Ready for Stage 2:          NO

Next steps:
  - Test generation: python generate.py --checkpoint output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz
  - Test final checkpoint: python generate.py --checkpoint output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.npz
  - View logs: tail -f output/logs/training_*.log
  - Reuse these settings: Run auto_train.py again and select 'y' to reuse config
[INFO] [OK] Saved run config to output/last_run_config.json
(venv) PS C:\dev\llm gpu 5> python generate.py --checkpoint output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz
[INFO] Logging initialized | Level: INFO | File: output/logs\training_20260618_202527.log
[INFO] =========================================================================
[INFO] [INIT] INITIALIZING AUTOREGRESSIVE TEXT GENERATION ENGINE
[INFO] =========================================================================
[INFO] Loaded shared tokenizer corpus: 9,500 documents (fineweb_100mb.txt)
[INFO] [CFG] Using checkpoint config: vocab=4096, ctx=128, embed=384, heads=6, layers=4, attention=strided
[INFO] 📂 Hydrating VRAM configurations from disk checkpoint: output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz
[INFO] ✅ Core parameters statefully hydrated. Model ready for instant execution.
[INFO] Vocabulary Size: 4096
[INFO] Max Context Length: 128
[INFO] Generation Temperature: 0.8
[INFO] Checkpoint: output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz   ep_384d_4l_20260618_195738.best.npz
[INFO] [OK] Model architecture initialized
[INFO] Input Seed Prompt: 'cuda '
[INFO] Seed Token IDs: [18, 14, 7, 2, 0]                              ==========
[INFO] =========================================================================

✍️ Generating: cuda called adventures., want's. bright and!
"
and looked
of,"!€'would. big.                                                    ==========

[INFO] ===================================================================================
[INFO] [OK] Generation complete!
[INFO] =========================================================================
[INFO] 🧹 Initiating physical layer allocation scrubbing...
[INFO] [OK] VRAM allocations cleared down safely.
(venv) PS C:\dev\llm gpu 5>

[INFO] ===================================================================================
[INFO] ===================================================================================
[INFO] ===================================================================================




(venv) PS C:\dev\llm gpu 5> python .\auto_train.py
[INFO] Logging initialized | Level: INFO | File: output/logs\training_20260618_202834.log       

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

Select dataset [1-3] (default: 2): 2
[INFO] [OK] Loaded FineWeb dataset: 619,541 documents

Dataset has 619,541 documents.
Limit to how many docs? (default: 5000, max/all = full corpus): 10000
[INFO]   [OK] Using first 10,000 documents

=========================================================================
STEP 2: MODEL ARCHITECTURE
=========================================================================

⚠️  GT730 KEPLER NOTE: still keep models to 1 layer for reliability.
[INFO] Preset parameter counts are approximate and assume char vocab ~256, ctx=64.
[INFO] Larger presets are available, but >1M params will be slow or may exceed VRAM on GT730.   
[INFO] Preset 8 can run with either 1 layer or 4 layers.
[INFO] Preset 9 is the 384D, 4-layer alternative when the 896D wide model plateaus early.       

Parameter-scale presets:
  1) MICRO   -   10.4K params |  16D,  1 heads, 1 layer | smoke tests
  2) TINY    -   26.8K params |  32D,  1 heads, 1 layer | quick debug runs
  3) SMALL   -   78.3K params |  64D,  2 heads, 1 layer | recommended GT730 baseline [RECOMMENDED]
  4) BASE    -  254.8K params | 128D,  4 heads, 1 layer | larger single-layer test
  5) LARGE   -  902.9K params | 256D,  8 heads, 1 layer | near 1M params
  6) XL      -   3.38M params | 512D,  8 heads, 1 layer | experimental on GT730
  7) XXL     -   5.21M params | 640D, 10 heads, 1 layer | very slow / experimental
  8) GIANT   -  10.04M params | 896D, 14 heads, 1 or 4 layers | ~10M params at 1L, optional 4L override
  9) DEEP    -   7.27M params | 384D,  6 heads, 4 layer | narrower 4L alternative to the wide giant preset

Select model [1-9] (default: 3): 9 
[INFO] Model 9 is fixed at 4 layers to give the narrower embedding more sequential depth.

[OK] Selected DEEP
     Approx params: 7.27M (vocab~156, ctx=128)
     Embedding dim: 384
     Num heads: 6
     Num layers: 4

=========================================================================
STEP 3: TRAINING HYPERPARAMETERS
=========================================================================

Learning rate (2M-10M params: safe range 0.0005-0.005, experimental on GT730)
  Tip: If loss plateaus (not changing), increase LR
  Tip: If NaN occurs, decrease LR and use gradient clipping
Learning rate (default: 0.001): 0.0003
Training steps (default: 100): 1000
Sequence length T (default: 128): 

[OK] Training config:
     Learning rate: 0.0003
     Steps: 1000
     Sequence length: 128

=========================================================================
STEP 4: LOGGING & CHECKPOINT
=========================================================================

[OK] Auto-generated names from training setup:
     Format: training_<steps>steps_<lr>lr_ctx<context>_<timestamp>
     Log file: training_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_202914.log
     Checkpoint: output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_202914.npz

=========================================================================
OPTIONAL CHECKPOINT INIT
=========================================================================
Leave blank to start from random weights.

Recent checkpoints:
  1) output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.npz
  2) output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.step1000.p100.npz
  3) output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz     
  4) output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.step750.p75.npz
  5) output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.step500.p50.npz
Checkpoint path or recent index (blank = random init): 3

Checkpoint config:
  ctx=128, embed=384, heads=6, layers=4, attention=strided
[OK] Init checkpoint: output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz

=========================================================================
STEP 5: POST-TRAINING OPTIONS
=========================================================================

Test generation after training? [y/n] (default: y):
Generation prompt (default: 'the'): Once 
Max tokens to generate (default: 50): 128
Generation temperature (default: 0.6): 0.8
Generation top_p (default: 0.9): 
Repetition penalty (default: 1.15): 

[OK] Generation config:
     Prompt: 'Once'
     Max tokens: 128
     Temperature: 0.8
     Top-p: 0.9
     Repetition penalty: 1.15

=========================================================================
TRAINING CONFIGURATION SUMMARY
=========================================================================

Dataset:        fineweb
  Docs:         10,000
  Init from:    output\checkpoints\gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_195738.best.npz

Model:          DEEP
  Params:       ~7.27M (selector estimate)
  Embedding:    384D
  Heads:        6
  Layers:       4
  Attention:    strided

Training:
  LR:           0.0003
  Steps:        1000
  Seq length:   128

Checkpoint:     output/checkpoints/gpt_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_202914.npz

Logging:        training_1000steps_0p0003lr_ctx128_deep_384d_4l_20260618_202914.log

Generation:
  Prompt:       'Once'
  Max tokens:   128
  Temperature:  0.8
  Top-p:        0.9
  Rep penalty:  1.15

Start training now? [y/n] (default: y):


