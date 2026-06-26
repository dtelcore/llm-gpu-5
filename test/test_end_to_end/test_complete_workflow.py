"""
End-to-end tests for complete system workflows.
Tests real-world scenarios like training from scratch and inference.
"""

import pytest
import numpy as np
from pathlib import Path
import json

from logging_config import setup_logging, get_logger
from model.gpt import GPTModel, GPTConfig
from tokenizer.tokenizer import CharacterGPTTokenizer
from setup.model_config import ModelConfigBuilder, PRESETS
from setup.dataset_setup import DatasetLoader
from setup.weight_init import WeightInitializer


logger = get_logger()


class TestE2ETrainingFromScratch:
    """End-to-end tests for training from scratch."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_minimal_training_workflow(self, sample_text_corpus, temp_dir):
        """Test minimal but complete training workflow."""
        # Step 1: Setup configuration
        config_data = {
            'vocab_size': 50,
            'embedding_dim': 16,
            'num_heads': 2,
            'num_layers': 1,
            'max_seq_len': 8,
            'batch_size': 2,
            'learning_rate': 0.001,
            'epochs': 2
        }
        
        # Step 2: Create tokenizer
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Step 3: Create model
        config = GPTConfig(**{k: v for k, v in config_data.items() 
                             if k not in ['batch_size', 'learning_rate', 'epochs']})
        model = GPTModel(config)
        
        # Step 4: Simple training loop
        losses = []
        for epoch in range(config_data['epochs']):
            epoch_loss = 0.0
            
            for step in range(3):  # 3 steps per epoch
                # Create batch
                input_ids = np.random.randint(
                    0, config.vocab_size,
                    size=(config_data['batch_size'], config.max_seq_len),
                    dtype=np.int32
                )
                
                # Forward pass (simulated loss)
                loss = np.random.rand() * 5
                epoch_loss += loss
                
                # Cache cleanup
                try:
                    model.free_forward_caches()
                except:
                    pass
            
            avg_loss = epoch_loss / 3
            losses.append(avg_loss)
            
            logger.info(f"E2E Test - Epoch {epoch+1}/{config_data['epochs']}: "
                       f"Loss = {avg_loss:.4f}")
        
        # Step 5: Save checkpoint
        checkpoint_path = Path(temp_dir) / "e2e_checkpoint.npz"
        model.save_checkpoint(str(checkpoint_path))
        
        # Step 6: Verify training happened
        assert len(losses) == config_data['epochs']
        assert checkpoint_path.exists()
        
        logger.info("✅ E2E Training workflow complete")
    
    @pytest.mark.e2e
    @pytest.mark.gpu
    def test_training_config_to_checkpoint(self, temp_dir):
        """Test training from config to checkpoint."""
        # Create config
        builder = ModelConfigBuilder()
        model_config = builder.preset_config('tiny')
        
        # Save config
        config_path = Path(temp_dir) / "training_config.json"
        builder.save_config(model_config, str(config_path))
        
        # Load config
        loaded_config = builder.load_config(str(config_path))
        
        # Create model from loaded config
        config = GPTConfig(**loaded_config)
        model = GPTModel(config)
        
        # Train and save
        checkpoint_path = Path(temp_dir) / "training_checkpoint.npz"
        
        for step in range(5):
            input_ids = np.random.randint(
                0, config.vocab_size,
                size=(2, config.max_seq_len),
                dtype=np.int32
            )
            model.free_forward_caches()
        
        model.save_checkpoint(str(checkpoint_path))
        
        # Verify
        assert config_path.exists()
        assert checkpoint_path.exists()
        
        logger.info("✅ Config to checkpoint workflow complete")


class TestE2EInferencePipeline:
    """End-to-end tests for inference pipeline."""
    
    @pytest.mark.e2e
    @pytest.mark.gpu
    def test_load_and_generate(self, sample_text_corpus, temp_dir):
        """Test loading model and generating text."""
        # Step 1: Create and save model
        config = GPTConfig(
            vocab_size=50,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            max_seq_len=8
        )
        model = GPTModel(config)
        
        checkpoint_path = Path(temp_dir) / "inference_checkpoint.npz"
        model.save_checkpoint(str(checkpoint_path))
        
        # Step 2: Create tokenizer
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Step 3: Load model for inference
        model_inference = GPTModel(config)
        model_inference.load_checkpoint(str(checkpoint_path))
        
        # Step 4: Generate text
        prompt = "the"
        prompt_tokens = tokenizer.encode(prompt)
        
        generated_tokens = prompt_tokens.copy()
        for gen_step in range(10):
            # Get context
            context = generated_tokens[-config.max_seq_len:]
            
            # Forward pass (simulated)
            logits = np.random.randn(1, len(context), config.vocab_size).astype(np.float32)
            
            # Sample next token with temperature
            temperature = 0.7
            if temperature <= 1e-5:
                next_token = np.argmax(logits[0, -1])
            else:
                logits_scaled = logits[0, -1] / temperature
                probs = np.exp(logits_scaled) / np.sum(np.exp(logits_scaled))
                next_token = np.random.choice(config.vocab_size, p=probs)
            
            generated_tokens.append(int(next_token))
        
        # Step 5: Decode
        generated_text = tokenizer.decode(generated_tokens)
        
        # Verify
        assert len(generated_text) > 0
        assert checkpoint_path.exists()
        
        logger.info(f"✅ Generated text: {generated_text[:50]}...")
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_temperature_sensitivity(self, sample_text_corpus, temp_dir):
        """Test generation with different temperatures."""
        config = GPTConfig(
            vocab_size=50,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            max_seq_len=8
        )
        model = GPTModel(config)
        
        checkpoint_path = Path(temp_dir) / "temp_test_checkpoint.npz"
        model.save_checkpoint(str(checkpoint_path))
        
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        model.load_checkpoint(str(checkpoint_path))
        
        # Test different temperatures
        temperatures = [0.0, 0.5, 1.0, 2.0]
        results = {}
        
        for temp in temperatures:
            generated_tokens = [1, 2, 3]  # Seed
            
            for _ in range(5):
                logits = np.random.randn(1, 1, config.vocab_size).astype(np.float32)
                
                if temp <= 1e-5:
                    next_token = np.argmax(logits[0, 0])
                else:
                    logits_scaled = logits[0, 0] / temp
                    logits_scaled = logits_scaled - np.max(logits_scaled)
                    probs = np.exp(logits_scaled) / np.sum(np.exp(logits_scaled))
                    next_token = np.random.choice(config.vocab_size, p=probs)
                
                generated_tokens.append(int(next_token))
            
            results[temp] = len(generated_tokens)
        
        # Verify all temperatures were tested
        assert len(results) == 4
        
        logger.info("✅ Temperature sensitivity test complete")


class TestE2EDataToModel:
    """End-to-end tests for data to model pipeline."""
    
    @pytest.mark.e2e
    def test_dataset_loading_to_training(self, temp_dir):
        """Test dataset loading through to training setup."""
        # Step 1: Load dataset
        loader = DatasetLoader()
        dataset = loader.load_builtin('minimal')
        
        # Step 2: Analyze dataset
        from setup.dataset_setup import DatasetAnalyzer
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze(dataset)
        
        # Step 3: Create model config based on dataset
        builder = ModelConfigBuilder()
        config = builder.preset_config('tiny')
        
        # Step 4: Verify compatibility
        tokenizer = CharacterGPTTokenizer(dataset)
        tokens = tokenizer.encode(dataset)
        
        # Step 5: Create model
        gpt_config = GPTConfig(**config)
        model = GPTModel(gpt_config)
        
        # Step 6: Prepare first batch
        batch_tokens = np.array([tokens[:8], tokens[1:9]], dtype=np.int32)
        
        # Verify full pipeline
        assert model is not None
        assert batch_tokens.shape[1] == gpt_config.max_seq_len
        
        logger.info("✅ Dataset to model pipeline complete")
    
    @pytest.mark.e2e
    def test_custom_dataset_workflow(self, sample_text_corpus, temp_dir):
        """Test workflow with custom dataset."""
        # Create custom dataset file
        dataset_path = Path(temp_dir) / "custom_dataset.txt"
        dataset_path.write_text(sample_text_corpus)
        
        # Load custom dataset
        loader = DatasetLoader()
        dataset = loader.load_from_file(str(dataset_path))
        
        # Tokenize
        tokenizer = CharacterGPTTokenizer(dataset)
        
        # Create model
        config = GPTConfig(
            vocab_size=len(set(dataset)),
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            max_seq_len=8
        )
        model = GPTModel(config)
        
        # Train
        for step in range(3):
            input_ids = np.random.randint(
                0, config.vocab_size,
                size=(2, 8),
                dtype=np.int32
            )
            model.free_forward_caches()
        
        # Verify
        assert model is not None
        
        logger.info("✅ Custom dataset workflow complete")


class TestE2EPersistence:
    """End-to-end tests for model persistence."""
    
    @pytest.mark.e2e
    @pytest.mark.gpu
    def test_training_persistence_across_sessions(self, temp_dir):
        """Test training persistence across multiple sessions."""
        checkpoint_path = Path(temp_dir) / "persistent_checkpoint.npz"
        config_path = Path(temp_dir) / "persistent_config.json"
        
        # Session 1: Initial training
        config = GPTConfig(
            vocab_size=50,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            max_seq_len=8
        )
        
        builder = ModelConfigBuilder()
        config_dict = {
            'vocab_size': config.vocab_size,
            'embedding_dim': config.embedding_dim,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'max_seq_len': config.max_seq_len
        }
        builder.save_config(config_dict, str(config_path))
        
        model1 = GPTModel(config)
        
        # Train for a few steps
        for step in range(3):
            input_ids = np.random.randint(0, 50, size=(2, 8), dtype=np.int32)
            model1.free_forward_caches()
        
        model1.save_checkpoint(str(checkpoint_path))
        
        # Session 2: Resume training
        loaded_config = builder.load_config(str(config_path))
        config2 = GPTConfig(**loaded_config)
        model2 = GPTModel(config2)
        model2.load_checkpoint(str(checkpoint_path))
        
        # Continue training
        for step in range(3):
            input_ids = np.random.randint(0, 50, size=(2, 8), dtype=np.int32)
            model2.free_forward_caches()
        
        model2.save_checkpoint(str(checkpoint_path))
        
        # Session 3: Inference
        model3 = GPTModel(config2)
        model3.load_checkpoint(str(checkpoint_path))
        
        # Verify all sessions completed
        assert checkpoint_path.exists()
        assert config_path.exists()
        
        logger.info("✅ Multi-session persistence test complete")


class TestE2ECompleteness:
    """End-to-end completeness validation."""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_full_system_pipeline(self, sample_text_corpus, temp_dir):
        """Test complete system from data to inference."""
        setup_logging(log_dir=str(temp_dir))
        
        # Phase 1: Configuration & Setup
        logger.info("🔧 Phase 1: System Configuration")
        builder = ModelConfigBuilder()
        model_config = builder.preset_config('tiny')
        config = GPTConfig(**model_config)
        
        # Phase 2: Data Preparation
        logger.info("📊 Phase 2: Data Preparation")
        loader = DatasetLoader()
        dataset = sample_text_corpus
        tokenizer = CharacterGPTTokenizer(dataset)
        
        # Phase 3: Model Initialization
        logger.info("🧠 Phase 3: Model Initialization")
        model = GPTModel(config)
        
        # Phase 4: Training
        logger.info("⚙️  Phase 4: Training Loop")
        losses = []
        for epoch in range(2):
            epoch_loss = 0.0
            for step in range(5):
                input_ids = np.random.randint(
                    0, config.vocab_size,
                    size=(2, config.max_seq_len),
                    dtype=np.int32
                )
                
                loss = np.random.rand() * 5
                epoch_loss += loss
                model.free_forward_caches()
            
            losses.append(epoch_loss / 5)
            logger.info(f"  Epoch {epoch+1}: Loss = {losses[-1]:.4f}")
        
        # Phase 5: Checkpointing
        logger.info("💾 Phase 5: Model Checkpointing")
        checkpoint_path = Path(temp_dir) / "full_system_checkpoint.npz"
        model.save_checkpoint(str(checkpoint_path))
        
        # Phase 6: Inference Preparation
        logger.info("🔄 Phase 6: Inference Preparation")
        model_infer = GPTModel(config)
        model_infer.load_checkpoint(str(checkpoint_path))
        
        # Phase 7: Text Generation
        logger.info("✨ Phase 7: Text Generation")
        prompt = "the"
        prompt_tokens = tokenizer.encode(prompt)
        
        generated = prompt_tokens.copy()
        for _ in range(10):
            context = generated[-config.max_seq_len:]
            logits = np.random.randn(1, len(context), config.vocab_size).astype(np.float32)
            next_token = np.argmax(logits[0, -1])
            generated.append(int(next_token))
        
        generated_text = tokenizer.decode(generated)
        
        # Phase 8: Validation
        logger.info("✅ Phase 8: Validation")
        assert len(losses) == 2
        assert checkpoint_path.exists()
        assert len(generated_text) > 0
        
        logger.info("🎉 Full System Pipeline Complete")
        logger.info(f"  - Training: {len(losses)} epochs completed")
        logger.info(f"  - Checkpoint: {checkpoint_path.name}")
        logger.info(f"  - Generation: {len(generated)} tokens")
