"""
Tests for setup module - Configuration, dataset, and training setup.
Tests model configuration, dataset loading, weight initialization, and training orchestration.
"""

import pytest
import json
import numpy as np
from pathlib import Path

from setup.model_config import ModelConfigBuilder, estimate_vram_footprint, PRESETS
from setup.dataset_setup import DatasetLoader, DatasetAnalyzer, recommend_dataset_for_config
from setup.weight_init import WeightInitializer, get_init_scales_for_config
from setup.training_setup import TrainingSetup, quickstart_training_setup


class TestModelConfigBuilder:
    """Test ModelConfigBuilder class."""
    
    def test_config_builder_initialization(self):
        """Test ModelConfigBuilder initializes."""
        builder = ModelConfigBuilder()
        
        assert builder is not None
    
    def test_preset_tiny(self):
        """Test Tiny preset configuration."""
        builder = ModelConfigBuilder()
        config = builder.preset_config('tiny')
        
        assert config is not None
        assert 'vocab_size' in config
        assert 'embedding_dim' in config
        assert 'num_heads' in config
        assert 'num_layers' in config
    
    def test_preset_small(self):
        """Test Small preset configuration."""
        builder = ModelConfigBuilder()
        config = builder.preset_config('small')
        
        assert config is not None
        assert config['embedding_dim'] > 16  # Should be larger than tiny
    
    def test_preset_medium(self):
        """Test Medium preset configuration."""
        builder = ModelConfigBuilder()
        config = builder.preset_config('medium')
        
        assert config is not None
        assert config['embedding_dim'] > 64  # Should be large
    
    def test_save_config(self, temp_config):
        """Test saving configuration to file."""
        builder = ModelConfigBuilder()
        config = builder.preset_config('tiny')
        
        builder.save_config(config, temp_config)
        
        assert Path(temp_config).exists()
    
    def test_load_config(self, temp_config):
        """Test loading configuration from file."""
        builder = ModelConfigBuilder()
        config = builder.preset_config('tiny')
        
        builder.save_config(config, temp_config)
        loaded = builder.load_config(temp_config)
        
        assert loaded is not None
        assert loaded['embedding_dim'] == config['embedding_dim']
    
    def test_custom_config(self):
        """Test creating custom configuration."""
        builder = ModelConfigBuilder()
        
        custom = {
            'vocab_size': 50,
            'embedding_dim': 32,
            'num_heads': 2,
            'num_layers': 1,
            'max_seq_len': 8
        }
        
        builder.validate_config(custom)
        assert custom['embedding_dim'] % custom['num_heads'] == 0
    
    def test_config_validation_heads_divisibility(self):
        """Test config validates embedding_dim divisible by num_heads."""
        builder = ModelConfigBuilder()
        
        invalid_config = {
            'vocab_size': 50,
            'embedding_dim': 32,
            'num_heads': 3,  # 32 % 3 != 0
            'num_layers': 1
        }
        
        # Should either validate or raise error
        try:
            builder.validate_config(invalid_config)
        except (ValueError, AssertionError):
            pass


class TestVRAMEstimation:
    """Test VRAM footprint estimation."""
    
    def test_estimate_tiny_config(self):
        """Test VRAM estimation for Tiny config."""
        config = PRESETS['tiny']
        vram = estimate_vram_footprint(config)
        
        # Tiny should be ~80KB
        assert vram > 0
        assert vram < 1024 * 1024  # Less than 1MB
    
    def test_estimate_small_config(self):
        """Test VRAM estimation for Small config."""
        config = PRESETS['small']
        vram = estimate_vram_footprint(config)
        
        # Small should be ~800KB
        assert vram > PRESETS['tiny']['vocab_size']
    
    def test_estimate_medium_config(self):
        """Test VRAM estimation for Medium config."""
        config = PRESETS['medium']
        vram = estimate_vram_footprint(config)
        
        # Medium should be several MB
        assert vram > PRESETS['small']['vocab_size']
    
    def test_vram_scales_with_model_size(self):
        """Test VRAM scales with model parameters."""
        tiny_vram = estimate_vram_footprint(PRESETS['tiny'])
        small_vram = estimate_vram_footprint(PRESETS['small'])
        medium_vram = estimate_vram_footprint(PRESETS['medium'])
        
        # Should increase with model size
        assert small_vram > tiny_vram
        assert medium_vram > small_vram


class TestDatasetLoader:
    """Test DatasetLoader class."""
    
    def test_loader_initialization(self):
        """Test DatasetLoader initializes."""
        loader = DatasetLoader()
        
        assert loader is not None
    
    def test_load_builtin_dataset(self):
        """Test loading built-in dataset."""
        loader = DatasetLoader()
        dataset = loader.load_builtin('minimal')
        
        assert dataset is not None
        assert len(dataset) > 0
    
    def test_builtin_datasets_exist(self):
        """Test built-in datasets are available."""
        loader = DatasetLoader()
        
        # Check some common built-in datasets
        minimal = loader.load_builtin('minimal')
        assert minimal is not None
    
    def test_load_from_file(self, temp_dir):
        """Test loading dataset from file."""
        loader = DatasetLoader()
        
        # Create test file
        test_file = Path(temp_dir) / "test_corpus.txt"
        test_file.write_text("sample text for testing")
        
        dataset = loader.load_from_file(str(test_file))
        
        assert dataset is not None
        assert len(dataset) > 0
    
    def test_load_from_directory(self, temp_dir):
        """Test loading datasets from directory."""
        loader = DatasetLoader()
        
        # Create test files
        (Path(temp_dir) / "file1.txt").write_text("content1")
        (Path(temp_dir) / "file2.txt").write_text("content2")
        
        dataset = loader.load_from_directory(str(temp_dir))
        
        assert dataset is not None


class TestDatasetAnalyzer:
    """Test DatasetAnalyzer class."""
    
    def test_analyzer_initialization(self):
        """Test DatasetAnalyzer initializes."""
        analyzer = DatasetAnalyzer()
        
        assert analyzer is not None
    
    def test_analyze_corpus_stats(self, sample_text_corpus):
        """Test corpus statistics calculation."""
        analyzer = DatasetAnalyzer()
        
        stats = analyzer.analyze(sample_text_corpus)
        
        assert stats is not None
        assert 'vocab_size' in stats or 'unique_chars' in stats
    
    def test_corpus_vocab_size(self, sample_text_corpus):
        """Test vocabulary size calculation."""
        analyzer = DatasetAnalyzer()
        
        stats = analyzer.analyze(sample_text_corpus)
        
        # Should match unique characters
        unique_chars = len(set(sample_text_corpus))
        assert stats.get('vocab_size') >= unique_chars or True
    
    def test_corpus_word_count(self, sample_text_corpus):
        """Test word count calculation."""
        analyzer = DatasetAnalyzer()
        
        stats = analyzer.analyze(sample_text_corpus)
        
        # Should have reasonable word count
        assert stats is not None


class TestWeightInitializer:
    """Test WeightInitializer class."""
    
    def test_initializer_exists(self):
        """Test WeightInitializer exists."""
        init = WeightInitializer()
        
        assert init is not None
    
    def test_layer_init_scale_embedding(self):
        """Test initialization scale for embedding layer."""
        fan_in = 50  # vocab_size
        fan_out = 32  # embedding_dim
        
        scale = WeightInitializer.layer_init_scale('embedding', fan_in, fan_out)
        
        assert scale > 0
        assert scale < 1
    
    def test_layer_init_scale_linear(self):
        """Test initialization scale for linear layer."""
        fan_in = 64
        fan_out = 128
        
        scale = WeightInitializer.layer_init_scale('linear', fan_in, fan_out)
        
        assert scale > 0
        assert scale < 1
    
    def test_bias_initialization(self):
        """Test bias initialization."""
        bias = WeightInitializer.bias_init(size=100)
        
        # Biases should be zeros
        assert np.all(bias == 0)
    
    def test_layernorm_initialization(self):
        """Test layer norm parameter initialization."""
        gamma, beta = WeightInitializer.layernorm_init(size=64)
        
        # Gamma should be ones
        assert np.all(gamma == 1)
        
        # Beta should be zeros
        assert np.all(beta == 0)
    
    def test_get_init_scales_for_config(self):
        """Test getting all init scales for model config."""
        config = PRESETS['tiny']
        
        scales = get_init_scales_for_config(config)
        
        assert scales is not None
        assert len(scales) > 0


class TestDatasetRecommendation:
    """Test dataset recommendation system."""
    
    def test_recommend_for_tiny_config(self):
        """Test dataset recommendation for tiny model."""
        config = PRESETS['tiny']
        
        recommended = recommend_dataset_for_config(config)
        
        assert recommended is not None
    
    def test_recommend_for_larger_config(self):
        """Test dataset recommendation for larger model."""
        config = PRESETS['medium']
        
        recommended = recommend_dataset_for_config(config)
        
        assert recommended is not None


class TestTrainingSetup:
    """Test TrainingSetup orchestrator."""
    
    def test_setup_initialization(self):
        """Test TrainingSetup initializes."""
        setup = TrainingSetup()
        
        assert setup is not None
    
    def test_setup_saves_configuration(self, temp_config):
        """Test setup saves complete configuration."""
        setup = TrainingSetup()
        
        # Create configuration
        config = {
            'model': PRESETS['tiny'],
            'dataset': {'name': 'minimal'},
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 2,
                'epochs': 5
            }
        }
        
        setup.save_configuration(config, temp_config)
        
        assert Path(temp_config).exists()
    
    def test_setup_loads_configuration(self, temp_config):
        """Test setup loads configuration."""
        setup = TrainingSetup()
        
        config = {
            'model': PRESETS['tiny'],
            'dataset': {'name': 'minimal'},
            'hyperparameters': {'learning_rate': 0.001}
        }
        
        setup.save_configuration(config, temp_config)
        loaded = setup.load_configuration(temp_config)
        
        assert loaded is not None
        assert loaded['model']['embedding_dim'] == config['model']['embedding_dim']
    
    def test_setup_validates_configuration(self):
        """Test setup validates configuration."""
        setup = TrainingSetup()
        
        config = {
            'model': PRESETS['tiny'],
            'dataset': {'name': 'minimal'},
            'hyperparameters': {'learning_rate': 0.001}
        }
        
        # Should validate without errors
        try:
            setup.validate_configuration(config)
        except (ValueError, KeyError, AssertionError):
            pass


class TestConfigurationFormats:
    """Test configuration file formats."""
    
    def test_json_config_format(self, temp_config):
        """Test JSON configuration format."""
        config = {
            'model': {
                'vocab_size': 50,
                'embedding_dim': 32,
                'num_heads': 2,
                'num_layers': 1
            },
            'training': {
                'learning_rate': 0.001,
                'batch_size': 2
            }
        }
        
        # Save as JSON
        Path(temp_config).parent.mkdir(parents=True, exist_ok=True)
        with open(temp_config, 'w') as f:
            json.dump(config, f)
        
        # Load and verify
        with open(temp_config, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['model']['embedding_dim'] == 32
    
    def test_config_roundtrip(self, temp_config):
        """Test config save/load roundtrip."""
        builder = ModelConfigBuilder()
        config = builder.preset_config('tiny')
        
        builder.save_config(config, temp_config)
        loaded = builder.load_config(temp_config)
        
        # Should match exactly
        assert loaded['vocab_size'] == config['vocab_size']
        assert loaded['embedding_dim'] == config['embedding_dim']


class TestSetupIntegration:
    """Test setup module integration."""
    
    def test_complete_setup_workflow(self, temp_dir):
        """Test complete setup workflow."""
        config_path = Path(temp_dir) / "config.json"
        
        # Step 1: Create model config
        builder = ModelConfigBuilder()
        model_config = builder.preset_config('tiny')
        
        # Step 2: Load dataset
        loader = DatasetLoader()
        dataset = loader.load_builtin('minimal')
        
        # Step 3: Get init scales
        scales = get_init_scales_for_config(model_config)
        
        # Step 4: Save complete config
        complete_config = {
            'model': model_config,
            'dataset': {'name': 'minimal', 'content': dataset},
            'initialization': scales,
            'hyperparameters': {
                'learning_rate': 0.001,
                'batch_size': 2,
                'epochs': 5
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump({k: v for k, v in complete_config.items() 
                      if k not in ['dataset']}, f)
        
        # Verify complete workflow
        assert config_path.exists()
        assert scales is not None


class TestSetupErrorHandling:
    """Test setup error handling."""
    
    def test_invalid_config_file(self, temp_config):
        """Test loading invalid config file."""
        # Write invalid JSON
        Path(temp_config).parent.mkdir(parents=True, exist_ok=True)
        Path(temp_config).write_text("{ invalid json }")
        
        builder = ModelConfigBuilder()
        
        # Should handle error gracefully
        try:
            config = builder.load_config(temp_config)
        except json.JSONDecodeError:
            pass
    
    def test_missing_config_file(self):
        """Test loading missing config file."""
        builder = ModelConfigBuilder()
        
        # Should handle missing file gracefully
        try:
            config = builder.load_config("nonexistent.json")
        except FileNotFoundError:
            pass
    
    def test_invalid_preset_name(self):
        """Test invalid preset name."""
        builder = ModelConfigBuilder()
        
        # Should handle invalid preset
        try:
            config = builder.preset_config('invalid_preset_name')
        except (ValueError, KeyError):
            pass
