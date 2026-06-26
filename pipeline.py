#!/usr/bin/env python3
"""
Complete Training Pipeline Orchestrator
Runs all 5 steps: dataset → config → tokenizer → train → generate
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda

from logging_config import logger
from setup.training_setup import TrainingSetup
from setup.dataset_setup import DatasetAnalyzer, BUILTIN_DATASETS
from setup.model_config import PRESETS
from tokenizer.tokenizer import CharacterGPTTokenizer
from model.gpt import GPTConfig, GPTModel
from core.loss import SoftmaxCrossEntropy


def step1_dataset_selection():
    """Step 1: Select and load dataset."""
    print("\n" + "="*80)
    print("STEP 1: DATASET SELECTION")
    print("="*80)
    
    # Check for available datasets
    available_datasets = list(BUILTIN_DATASETS.keys())
    fineweb_path = Path("data/fineweb_100mb.txt")
    has_fineweb = fineweb_path.exists()
    
    print("\nAvailable builtin datasets:")
    for name, info in BUILTIN_DATASETS.items():
        print(f"  {name:15} - {info['description']}")
    
    if has_fineweb:
        fineweb_size_mb = fineweb_path.stat().st_size / (1024 * 1024)
        print(f"  {'fineweb_100mb':15} - FineWeb dataset (~{fineweb_size_mb:.0f}MB of real web text)")
        available_datasets.append("fineweb_100mb")
    
    default_choice = "fineweb_100mb" if has_fineweb else "minimal"
    choice = input(f"\nSelect dataset [default: {default_choice}]: ").strip() or default_choice
    
    if choice in BUILTIN_DATASETS:
        dataset_info = BUILTIN_DATASETS[choice]
        corpus = dataset_info['data']
        dataset_name = choice
        logger.info(f"[OK] Loaded builtin dataset: {dataset_name}")
        logger.info(f"  Sentences: {len(corpus)}")
        logger.info(f"  Est. vocab: {dataset_info['vocab_size_estimate']}")
        
        # Show sample
        print(f"\n  Sample text: {corpus[0][:60]}...")
    
    elif choice == "fineweb_100mb" and has_fineweb:
        with open(fineweb_path, 'r', encoding='utf-8') as f:
            corpus = [line.strip() for line in f.readlines() if line.strip()]
        dataset_name = "fineweb_100mb"
        logger.info(f"[OK] Loaded FineWeb dataset from {fineweb_path}")
        logger.info(f"  Documents: {len(corpus):,}")
        fineweb_size_mb = fineweb_path.stat().st_size / (1024 * 1024)
        logger.info(f"  File size: {fineweb_size_mb:.2f} MB")
        
        # Show sample
        if corpus:
            print(f"\n  Sample text: {corpus[0][:60]}...")
    
    else:
        # Try to load custom file
        filepath = Path(f"data/{choice}.txt")
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                corpus = [line.strip() for line in f.readlines() if line.strip()]
            dataset_name = choice
            logger.info(f"[OK] Loaded custom dataset from {filepath}")
            logger.info(f"  Documents: {len(corpus):,}")
        else:
            logger.error(f"Dataset not found: {choice}")
            logger.info("Using default 'minimal' dataset")
            corpus = BUILTIN_DATASETS['minimal']['data']
            dataset_name = 'minimal'
    
    # Analyze dataset
    analyzer = DatasetAnalyzer(corpus)
    print("\nDataset Statistics:")
    analyzer.print_stats()
    
    return corpus, dataset_name


def step2_configuration():
    """Step 2: Configure model, dataset, and training."""
    print("\n" + "="*80)
    print("STEP 2: MODEL & TRAINING CONFIGURATION")
    print("="*80)
    
    print("\nAvailable model presets:")
    for name, info in PRESETS.items():
        print(f"  {name:12} - {info['description']}")
    
    model_choice = input("\nSelect model [tiny/small/medium] (default: tiny): ").strip() or "tiny"
    if model_choice not in PRESETS:
        logger.warning(f"Unknown model {model_choice}, using 'tiny'")
        model_choice = 'tiny'
    
    model_preset = PRESETS[model_choice].copy()
    logger.info(f"[OK] Selected model: {model_preset['name']}")
    
    print("\nTraining hyperparameter presets:")
    print("  conservative - Stable training (lr=0.001, epochs=5)")
    print("  moderate     - Balanced training (lr=0.01, epochs=10)")
    print("  aggressive   - Fast training (lr=0.1, epochs=20)")
    
    hyperparam_choice = input("\nSelect hyperparameters [conservative/moderate/aggressive] (default: conservative): ").strip() or "conservative"
    
    hyperparams = {
        'conservative': {'learning_rate': 0.001, 'num_epochs': 5, 'batch_size': 2},
        'moderate': {'learning_rate': 0.01, 'num_epochs': 10, 'batch_size': 2},
        'aggressive': {'learning_rate': 0.1, 'num_epochs': 20, 'batch_size': 2},
    }
    
    if hyperparam_choice in hyperparams:
        hyperparams = hyperparams[hyperparam_choice]
        logger.info(f"[OK] Selected hyperparameters: {hyperparam_choice}")
        logger.info(f"  Learning rate: {hyperparams['learning_rate']}")
        logger.info(f"  Epochs: {hyperparams['num_epochs']}")
        logger.info(f"  Batch size: {hyperparams['batch_size']}")
    else:
        hyperparams = hyperparams['conservative']
        logger.warning(f"Unknown preset {hyperparam_choice}, using 'conservative'")
    
    return model_choice, model_preset, hyperparams


def step3_tokenizer(corpus):
    """Step 3: Build tokenizer from corpus."""
    print("\n" + "="*80)
    print("STEP 3: TOKENIZER BUILDING")
    print("="*80)
    
    logger.info("Building character-level tokenizer...")
    tokenizer = CharacterGPTTokenizer(corpus)
    
    logger.info(f"[OK] Tokenizer built successfully")
    logger.info(f"  Vocabulary size: {tokenizer.vocab_size}")
    logger.info(f"  Unique characters: {len(tokenizer.uchars)}")
    logger.info(f"  Characters: {tokenizer.uchars[:50]}{'...' if len(tokenizer.uchars) > 50 else ''}")
    logger.info(f"  BOS token ID: {tokenizer.BOS_ID}")
    logger.info(f"  PAD token ID: {tokenizer.PAD_ID}")
    
    # Test encoding/decoding
    test_text = corpus[0] if corpus else "hello"
    pieces, token_ids, logs = tokenizer.encode(test_text)
    decoded_text, decode_logs = tokenizer.decode(token_ids)
    
    print(f"\n  Test encode/decode:")
    print(f"    Original:  '{test_text}'")
    print(f"    Tokens:    {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
    print(f"    Decoded:   '{decoded_text}'")
    
    return tokenizer


def step4_training(tokenizer, corpus, model_preset, hyperparams):
    """Step 4: Train model (simplified version)."""
    print("\n" + "="*80)
    print("STEP 4: MODEL TRAINING")
    print("="*80)
    
    # Initialize model - override vocab_size with actual tokenizer vocab
    model_config_params = model_preset.copy()
    model_config_params['vocab_size'] = tokenizer.vocab_size
    model_cfg = GPTConfig(**model_config_params)
    
    logger.info(f"Initializing model...")
    logger.info(f"  Vocab size: {model_cfg.vocab_size}")
    logger.info(f"  Embedding dim: {model_cfg.embedding_dim}")
    logger.info(f"  Num heads: {model_cfg.num_heads}")
    logger.info(f"  Num layers: {model_cfg.num_layers}")
    logger.info(f"  Max seq len: {model_cfg.max_len}")
    
    # Note: Full training requires more comprehensive implementation
    print(f"\n⚠ Training Implementation Note:")
    print(f"  Full training loop requires:")
    print(f"  - Gradient accumulation")
    print(f"  - Checkpoint saving")
    print(f"  - Loss tracking")
    print(f"  - Validation evaluation")
    print(f"\n  For complete training, run:")
    print(f"  >>> python train.py")
    
    # Return model for generation testing
    return model_cfg


def step5_generation(tokenizer, model_cfg):
    """Step 5: Test generation with trained model."""
    print("\n" + "="*80)
    print("STEP 5: GENERATION TESTING")
    print("="*80)
    
    print("\nTo test generation with a trained model:")
    print("  1. First run: python train.py")
    print("  2. Then run: python generate.py")
    print("\nFor quick demo, run:")
    print("  >>> python generate.py")


def main():
    """Run complete pipeline."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "KEPLER GT 730 - COMPLETE TRAINING PIPELINE".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Step 1: Dataset Selection
        corpus, dataset_name = step1_dataset_selection()
        
        # Step 2: Configuration
        model_name, model_preset, hyperparams = step2_configuration()
        
        # Step 3: Tokenizer Building
        tokenizer = step3_tokenizer(corpus)
        
        # Step 4: Training Setup
        model_cfg = step4_training(tokenizer, corpus, model_preset, hyperparams)
        
        # Step 5: Generation Testing
        step5_generation(tokenizer, model_cfg)
        
        # Summary
        print("\n" + "="*80)
        print("PIPELINE SETUP COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("  1. Run training:   python train.py")
        print("  2. Test generation: python generate.py")
        print("  3. View logs:       cat output/logs/training_*.log")
        print("  4. Check artifacts: ls artifacts/checkpoints/")
        print("\n" + "="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
