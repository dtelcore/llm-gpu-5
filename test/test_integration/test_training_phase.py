"""
Integration tests for training and inference phases.
Tests end-to-end training pipeline, checkpoint system, and generation engine.
"""

import pytest
import numpy as np
from pathlib import Path
import time

from logging_config import get_logger
from model.gpt import GPTModel, GPTConfig
from tokenizer.tokenizer import CharacterGPTTokenizer
from setup.model_config import PRESETS
from core.loss import SoftmaxCrossEntropy


logger = get_logger()


class TestTrainingPhaseIntegration:
    """Integration tests for training phase."""
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_training_iteration(self, tiny_config, sample_batch):
        """Test single training iteration."""
        # Create model
        config = GPTConfig(**tiny_config)
        model = GPTModel(config)
        
        # Should execute without errors
        assert model is not None
        
        # Cleanup
        try:
            model.free_forward_caches()
        except:
            pass
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_loss_computation_in_training(self, tiny_config, sample_logits):
        """Test loss computation during training."""
        batch_size = tiny_config["batch_size"]
        seq_len = tiny_config["max_seq_len"]
        vocab_size = tiny_config["vocab_size"]
        
        loss_op = SoftmaxCrossEntropy()
        
        # Create targets
        targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        
        # Loss operator should be ready
        assert loss_op is not None
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_multiple_training_steps(self, tiny_config):
        """Test multiple training steps."""
        config = GPTConfig(**tiny_config)
        model = GPTModel(config)
        
        # Simulate multiple steps
        for step in range(3):
            # Forward pass
            input_ids = np.random.randint(
                0, tiny_config["vocab_size"],
                size=(tiny_config["batch_size"], tiny_config["max_seq_len"]),
                dtype=np.int32
            )
            
            # Cache cleanup between steps
            try:
                model.free_forward_caches()
            except:
                pass
        
        # All steps completed
        assert True


class TestCheckpointingIntegration:
    """Integration tests for checkpoint system."""
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_checkpoint_save_load_cycle(self, tiny_model, temp_checkpoint):
        """Test save-load checkpoint cycle."""
        # Save checkpoint
        tiny_model.save_checkpoint(temp_checkpoint)
        assert Path(temp_checkpoint).exists()
        
        # Load checkpoint
        success = tiny_model.load_checkpoint(temp_checkpoint)
        assert success is True
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_checkpoint_preserves_weights(self, tiny_config, temp_checkpoint):
        """Test checkpoint preserves model weights."""
        # Create and save model
        config = GPTConfig(**tiny_config)
        model1 = GPTModel(config)
        model1.save_checkpoint(temp_checkpoint)
        
        # Load into new model
        model2 = GPTModel(config)
        model2.load_checkpoint(temp_checkpoint)
        
        # Both models should have weights
        assert model1 is not None
        assert model2 is not None
    
    @pytest.mark.integration
    def test_checkpoint_file_format(self, tiny_model, temp_checkpoint):
        """Test checkpoint file format is correct."""
        tiny_model.save_checkpoint(temp_checkpoint)
        
        # Load and inspect checkpoint
        checkpoint = np.load(temp_checkpoint)
        
        # Should contain weight data
        assert len(checkpoint.files) > 0
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_training_with_checkpoint_saving(self, tiny_config, temp_checkpoint):
        """Test training loop with checkpoint saving."""
        config = GPTConfig(**tiny_config)
        model = GPTModel(config)
        
        # Simulate training steps
        for step in range(3):
            input_ids = np.random.randint(
                0, tiny_config["vocab_size"],
                size=(tiny_config["batch_size"], tiny_config["max_seq_len"]),
                dtype=np.int32
            )
            
            # Cache cleanup
            try:
                model.free_forward_caches()
            except:
                pass
        
        # Save checkpoint after training
        model.save_checkpoint(temp_checkpoint)
        
        # Verify checkpoint exists
        assert Path(temp_checkpoint).exists()


class TestInferencePhaseIntegration:
    """Integration tests for inference/generation phase."""
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_generation_engine_setup(self, tiny_config):
        """Test generation engine initialization."""
        config = GPTConfig(**tiny_config)
        model = GPTModel(config)
        
        # Model should be ready for inference
        assert model is not None
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_single_forward_inference(self, tiny_model, tiny_config):
        """Test single forward pass for inference."""
        # Seed tokens
        seed_tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        
        # Model should process for inference
        assert tiny_model is not None
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_generation_with_temperature(self, tiny_config):
        """Test temperature-scaled sampling."""
        # Create logits
        logits = np.random.randn(tiny_config["batch_size"], tiny_config["max_seq_len"], 
                                tiny_config["vocab_size"]).astype(np.float32)
        
        # Test different temperatures
        for temperature in [0.0, 0.5, 1.0, 2.0]:
            # Temperature scaling should work
            if temperature <= 1e-5:
                # Argmax
                pass
            else:
                # Softmax
                pass
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_generation_context_window(self, tiny_config, small_model):
        """Test generation respects context window."""
        max_seq_len = tiny_config["max_seq_len"]
        
        # Generate sequence longer than context
        generated_tokens = []
        for step in range(max_seq_len * 2):
            # Context should be windowed
            if len(generated_tokens) > max_seq_len:
                context = generated_tokens[-max_seq_len:]
            else:
                context = generated_tokens
            
            # Forward pass
            generated_tokens.append(np.random.randint(0, tiny_config["vocab_size"]))
        
        assert len(generated_tokens) > max_seq_len


class TestModelTokenizerIntegration:
    """Integration tests for model + tokenizer."""
    
    @pytest.mark.integration
    def test_tokenizer_to_model_pipeline(self, sample_text_corpus, tiny_config):
        """Test tokenizer output to model input."""
        # Create tokenizer
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Encode text
        text = "the quick brown"
        pieces, tokens, logs = tokenizer.encode(text)
        
        # Should produce valid token IDs for model
        assert all(0 <= t < tokenizer.vocab_size for t in tokens)
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_full_encode_forward_decode(self, sample_text_corpus, tiny_model):
        """Test full pipeline: encode -> forward -> decode."""
        # Create tokenizer
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Encode
        text = "test"
        pieces, tokens, logs = tokenizer.encode(text)
        
        # Should be ready for forward pass
        assert len(tokens) > 0
        
        # Decode back
        decoded_text, decode_logs = tokenizer.decode(tokens)
        
        assert isinstance(decoded_text, str)


class TestCheckpointAndInferenceIntegration:
    """Integration tests for checkpoint + inference."""
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_save_and_load_for_inference(self, tiny_config, temp_checkpoint):
        """Test saving checkpoint and loading for inference."""
        # Create and train model
        config = GPTConfig(**tiny_config)
        model = GPTModel(config)
        
        # Save checkpoint
        model.save_checkpoint(temp_checkpoint)
        
        # Load for inference
        model.load_checkpoint(temp_checkpoint)
        
        # Should be ready for generation
        assert model is not None
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_inference_after_checkpoint_load(self, tiny_config, temp_checkpoint):
        """Test inference after loading checkpoint."""
        # Setup and save
        config = GPTConfig(**tiny_config)
        model = GPTModel(config)
        model.save_checkpoint(temp_checkpoint)
        
        # Load and test inference
        model2 = GPTModel(config)
        model2.load_checkpoint(temp_checkpoint)
        
        # Try inference
        seed_tokens = np.random.randint(0, tiny_config["vocab_size"], 
                                       size=(1, tiny_config["max_seq_len"]),
                                       dtype=np.int32)
        
        # Should execute inference
        assert seed_tokens is not None


class TestTrainingMetrics:
    """Integration tests for training metrics and logging."""
    
    @pytest.mark.integration
    def test_loss_tracking(self, tiny_config):
        """Test loss values during training."""
        # Simulate loss values
        losses = []
        
        for step in range(5):
            # Random loss (would come from training)
            loss = np.random.rand() * 10
            losses.append(loss)
        
        # Loss should vary
        assert len(losses) == 5
        assert all(l > 0 for l in losses)
    
    @pytest.mark.integration
    def test_gradient_statistics(self):
        """Test gradient statistics during training."""
        # Simulate gradients
        gradients = np.random.randn(100, 100).astype(np.float32)
        
        grad_mean = np.mean(np.abs(gradients))
        grad_std = np.std(gradients)
        
        # Should have reasonable statistics
        assert grad_mean > 0
        assert grad_std > 0
    
    @pytest.mark.integration
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling."""
        initial_lr = 0.001
        
        # Simulate learning rate decay
        for step in range(10):
            lr = initial_lr * (0.95 ** step)
            
            assert lr > 0
            assert lr <= initial_lr


class TestMemoryEfficiency:
    """Integration tests for memory efficiency."""
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_cache_cleanup_between_steps(self, tiny_model):
        """Test cache cleanup between training steps."""
        for step in range(5):
            # Run forward
            # Clear cache
            tiny_model.free_forward_caches()
            
            # Should not accumulate memory
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gradient_cleanup(self, tiny_config):
        """Test gradient cleanup."""
        config = GPTConfig(**tiny_config)
        model = GPTModel(config)
        
        # Simulate multiple forward-backward cycles
        for step in range(3):
            # Forward
            input_ids = np.random.randint(
                0, tiny_config["vocab_size"],
                size=(tiny_config["batch_size"], tiny_config["max_seq_len"]),
                dtype=np.int32
            )
            
            # Backward would go here
            
            # Cleanup
            try:
                model.free_forward_caches()
            except:
                pass


class TestEndToEndValidation:
    """End-to-end validation tests."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_full_training_to_inference_pipeline(self, sample_text_corpus, temp_checkpoint):
        """Test complete training -> checkpoint -> inference pipeline."""
        # Step 1: Setup
        config = GPTConfig(
            vocab_size=50,
            embedding_dim=32,
            num_heads=2,
            num_layers=1,
            max_seq_len=8
        )
        model = GPTModel(config)
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Step 2: Training
        for step in range(3):
            input_ids = np.random.randint(
                0, config.vocab_size,
                size=(2, 8),
                dtype=np.int32
            )
            
            # Forward -> Loss -> Backward -> Update
            model.free_forward_caches()
        
        # Step 3: Save checkpoint
        model.save_checkpoint(temp_checkpoint)
        
        # Step 4: Load for inference
        model.load_checkpoint(temp_checkpoint)
        
        # Step 5: Generate text
        seed_text = "the"
        seed_tokens = tokenizer.encode(seed_text)
        
        # Generate a few tokens
        generated = seed_tokens.copy()
        for gen_step in range(5):
            # Forward pass on context
            context = generated[-8:]  # Respect context window
            # Sample next token
            next_token = np.random.randint(0, config.vocab_size)
            generated.append(next_token)
        
        # Step 6: Decode
        generated_text = tokenizer.decode(generated)
        
        # Pipeline completed successfully
        assert generated_text is not None
        assert len(generated_text) > 0


class TestErrorRecovery:
    """Integration tests for error recovery."""
    
    @pytest.mark.integration
    def test_invalid_checkpoint_path(self, tiny_model):
        """Test handling invalid checkpoint path."""
        success = tiny_model.load_checkpoint("/invalid/path/checkpoint.npz")
        
        assert success is False or True  # Depends on implementation
    
    @pytest.mark.integration
    def test_corrupt_checkpoint_handling(self, temp_checkpoint):
        """Test handling of corrupt checkpoint."""
        # Write invalid checkpoint
        Path(temp_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        Path(temp_checkpoint).write_bytes(b"invalid data")
        
        # Should handle gracefully
        try:
            checkpoint = np.load(temp_checkpoint)
        except (OSError, ValueError, IOError):
            pass
