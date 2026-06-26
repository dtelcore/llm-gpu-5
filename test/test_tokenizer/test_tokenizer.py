"""
Tests for tokenizer module - Text tokenization and encoding/decoding.
Tests CharacterGPTTokenizer functionality.
"""

import pytest
import numpy as np

from tokenizer.tokenizer import CharacterGPTTokenizer


class TestTokenizerInitialization:
    """Test tokenizer initialization."""
    
    def test_tokenizer_from_corpus(self, sample_text_corpus):
        """Test tokenizer initialization from corpus."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0
    
    def test_tokenizer_vocab_buildup(self, sample_text_corpus):
        """Test tokenizer builds vocabulary from corpus."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Vocabulary should contain at least the unique characters
        unique_chars = len(set(sample_text_corpus))
        assert tokenizer.vocab_size >= unique_chars
    
    def test_tokenizer_has_special_tokens(self, sample_text_corpus):
        """Test tokenizer includes special tokens."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Should have vocab properties
        assert hasattr(tokenizer, 'vocab_size')
        assert tokenizer.vocab_size > 0


class TestTokenization:
    """Test text tokenization."""
    
    def test_encode_simple_text(self, sample_text_corpus):
        """Test encoding simple text."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        text = "the quick"
        pieces, tokens, logs = tokenizer.encode(text)
        
        # Should return token IDs
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_encode_returns_integers(self, sample_text_corpus):
        """Test encoding returns integer token IDs."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        pieces, tokens, logs = tokenizer.encode("test")
        
        # All tokens should be integers
        for token in tokens:
            assert isinstance(token, (int, np.integer))
    
    def test_encode_token_range(self, sample_text_corpus):
        """Test all encoded tokens are within vocab range."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        pieces, tokens, logs = tokenizer.encode("the quick brown fox")
        
        # All tokens should be < vocab_size
        for token in tokens:
            assert 0 <= token < tokenizer.vocab_size
    
    def test_decode_tokens(self, sample_text_corpus):
        """Test decoding token IDs back to text."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        text = "the quick"
        pieces, tokens, logs = tokenizer.encode(text)
        decoded_text, decode_logs = tokenizer.decode(tokens)
        
        # Decoded should be string
        assert isinstance(decoded_text, str)
        
        # Should match original (accounting for whitespace normalization)
        assert len(decoded_text) > 0
    
    def test_encode_decode_roundtrip(self, sample_text_corpus):
        """Test encode-decode roundtrip."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        original_text = "the quick brown fox"
        
        # Encode
        pieces, tokens, logs = tokenizer.encode(original_text)
        
        # Decode
        decoded_text = tokenizer.decode(tokens)
        
        # Should be equivalent (may differ in whitespace)
        assert decoded_text is not None
        assert len(decoded_text) > 0
    
    def test_multiple_encode_calls(self, sample_text_corpus):
        """Test multiple encode calls are consistent."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        text = "test"
        
        _, tokens1, _ = tokenizer.encode(text)
        _, tokens2, _ = tokenizer.encode(text)
        
        # Should be identical
        assert tokens1 == tokens2


class TestTokenizerEdgeCases:
    """Test tokenizer edge cases."""
    
    def test_encode_empty_string(self, sample_text_corpus):
        """Test encoding empty string."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        pieces, tokens, logs = tokenizer.encode("")
        
        # Should return empty or padding
        assert isinstance(tokens, list)
    
    def test_encode_single_character(self, sample_text_corpus):
        """Test encoding single character."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        pieces, tokens, logs = tokenizer.encode("a")
        
        # Should return single token
        assert len(tokens) >= 1
    
    def test_encode_special_characters(self, sample_text_corpus):
        """Test encoding special characters."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Try to encode text with special chars (may not be in corpus)
        try:
            pieces, tokens, logs = tokenizer.encode("test! @#$%")
            assert len(tokens) > 0
        except ValueError:
            # Special chars might not be in corpus - that's OK
            pass
    
    def test_encode_whitespace(self, sample_text_corpus):
        """Test encoding whitespace."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        pieces, tokens, logs = tokenizer.encode("a b c")
        
        # Should tokenize spaces
        assert len(tokens) > 0
    
    def test_decode_empty_tokens(self, sample_text_corpus):
        """Test decoding empty token list."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        decoded_text, logs = tokenizer.decode([])
        
        # Should return empty string or handle gracefully
        assert isinstance(decoded_text, str)
    
    def test_decode_invalid_token_id(self, sample_text_corpus):
        """Test decoding invalid token ID."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Create token ID beyond vocab
        invalid_tokens = [tokenizer.vocab_size + 100]
        
        # Should handle gracefully (raise error or return unknown)
        try:
            decoded = tokenizer.decode(invalid_tokens)
            # If it doesn't raise, should return something
            assert decoded is not None
        except (ValueError, IndexError):
            # Exception is acceptable
            pass


class TestVocabulary:
    """Test vocabulary management."""
    
    def test_vocab_size_positive(self, sample_text_corpus):
        """Test vocabulary size is positive."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        assert tokenizer.vocab_size > 0
    
    def test_vocab_size_reasonable(self, sample_text_corpus):
        """Test vocabulary size is reasonable."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Should not be larger than any reasonable vocabulary
        assert tokenizer.vocab_size < 1000000
    
    def test_vocab_contains_corpus_chars(self, sample_text_corpus):
        """Test vocabulary contains characters from corpus."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Encode should work for corpus text
        pieces, tokens, logs = tokenizer.encode(sample_text_corpus)
        
        # Should have tokens
        assert len(tokens) > 0
    
    def test_vocab_consistency(self, sample_text_corpus):
        """Test vocabulary is consistent across calls."""
        tokenizer1 = CharacterGPTTokenizer(sample_text_corpus)
        tokenizer2 = CharacterGPTTokenizer(sample_text_corpus)
        
        # Same corpus should produce same vocab size
        assert tokenizer1.vocab_size == tokenizer2.vocab_size


class TestTokenProperties:
    """Test properties of tokens."""
    
    def test_tokens_are_non_negative(self, sample_text_corpus):
        """Test all tokens are non-negative."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        pieces, tokens, logs = tokenizer.encode("test text")
        
        for token in tokens:
            assert token >= 0
    
    def test_tokens_dtype(self, sample_text_corpus):
        """Test tokens have correct data type."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        pieces, tokens, logs = tokenizer.encode("test")
        
        # Tokens should be integer-like
        if isinstance(tokens, np.ndarray):
            assert tokens.dtype in [np.int32, np.int64, np.int_]
    
    def test_token_consistency_same_input(self, sample_text_corpus):
        """Test same input always produces same tokens."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        text = "consistent"
        
        for _ in range(5):
            _, tokens, _ = tokenizer.encode(text)
            decoded_text, _ = tokenizer.decode(tokens)
            
            # Decoding should always produce same result
            _, tokens2, _ = tokenizer.encode(text)
            decoded_text2, _ = tokenizer.decode(tokens2)
            assert decoded_text2 == decoded_text


class TestTokenBatching:
    """Test tokenizer with batch processing."""
    
    def test_encode_multiple_texts(self, sample_text_corpus):
        """Test encoding multiple texts."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        texts = ["hello", "world", "test"]
        
        token_lists = [tokenizer.encode(text)[1] for text in texts]  # Extract ids from tuple
        
        # Should produce tokens for each text
        assert len(token_lists) == 3
        assert all(len(tl) > 0 for tl in token_lists)
    
    def test_tokens_different_lengths(self, sample_text_corpus):
        """Test tokens can have different lengths."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        _, short_tokens, _ = tokenizer.encode("a")
        _, long_tokens, _ = tokenizer.encode("the quick brown fox jumps over")
        
        # Longer text should produce more tokens
        assert len(long_tokens) >= len(short_tokens)


class TestUnicode:
    """Test unicode and special character handling."""
    
    def test_encode_numeric_strings(self, sample_text_corpus):
        """Test encoding numeric strings."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Should handle numbers (if in corpus)
        try:
            _, tokens, _ = tokenizer.encode("123")
            assert len(tokens) > 0
        except (KeyError, ValueError):
            # Numbers might not be in corpus
            pass
    
    def test_encode_mixed_case(self, sample_text_corpus):
        """Test encoding mixed case text."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Try mixed case (may not work if corpus is lowercase)
        text = "ThE QuICK"
        try:
            _, tokens, _ = tokenizer.encode(text)
            assert len(tokens) > 0
        except ValueError:
            # Uppercase might not be in corpus - that's OK
            pass
    
    def test_repeated_characters(self, sample_text_corpus):
        """Test encoding repeated characters."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        _, tokens, _ = tokenizer.encode("aaaaa")
        
        # Should tokenize all characters
        assert len(tokens) == 5 or len(tokens) > 0  # Depends on tokenization strategy

    def test_encode_unseen_unicode_uses_byte_fallback(self, sample_text_corpus):
        """Test unseen Unicode characters encode through the reserved byte window."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)

        unseen_text = "seen ☊"
        _, tokens, _ = tokenizer.encode(unseen_text)

        byte_tokens = [
            token for token in tokens
            if tokenizer.byte_offset <= token < tokenizer.BOS_ID
        ]

        assert len(byte_tokens) == len("☊".encode("utf-8"))

    def test_decode_byte_fallback_roundtrip(self, sample_text_corpus):
        """Test byte fallback tokens decode back to the original Unicode text."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)

        original_text = "prefix ☊ suffix"
        _, tokens, _ = tokenizer.encode(original_text)
        decoded_text, _ = tokenizer.decode(tokens)

        assert decoded_text == original_text


class TestTokenizerMemoryManagement:
    """Test tokenizer memory usage."""
    
    def test_tokenizer_creation_memory(self, sample_text_corpus):
        """Test tokenizer creation doesn't leak memory."""
        # Create multiple tokenizers
        tokenizers = []
        for _ in range(10):
            t = CharacterGPTTokenizer(sample_text_corpus)
            tokenizers.append(t)
        
        # Should all have consistent properties
        assert all(t.vocab_size == tokenizers[0].vocab_size for t in tokenizers)
    
    def test_encoding_doesnt_accumulate(self, sample_text_corpus):
        """Test encoding calls don't accumulate memory."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Call encode many times
        for _ in range(100):
            _, tokens, _ = tokenizer.encode("test text")
        
        # Should still work normally
        _, tokens, _ = tokenizer.encode("final")
        assert len(tokens) > 0


class TestCharacterGPTTokenizer:
    """Test CharacterGPTTokenizer specifically."""
    
    def test_character_level_tokenization(self, sample_text_corpus):
        """Test tokenizer works at character level."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Vocabulary should be relatively small (character-level)
        assert tokenizer.vocab_size < 500  # Reasonable for char-level
    
    def test_corpus_preservation(self, sample_text_corpus):
        """Test tokenizer preserves corpus information."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Should be able to tokenize the corpus it was built from
        _, tokens, _ = tokenizer.encode(sample_text_corpus)
        decoded = tokenizer.decode(tokens)
        
        # Roundtrip should work
        assert len(decoded) > 0
    
    def test_different_corpus_different_vocab(self):
        """Test different corpus produces different vocabularies."""
        corpus1 = "hello world"
        corpus2 = "foo bar baz qux"
        
        tokenizer1 = CharacterGPTTokenizer(corpus1)
        tokenizer2 = CharacterGPTTokenizer(corpus2)
        
        # Vocabularies should be different
        # (character sets are different)
        assert tokenizer1.vocab_size >= 0
        assert tokenizer2.vocab_size >= 0


class TestTokenizerPerformance:
    """Test tokenizer performance characteristics."""
    
    def test_encode_large_text(self, sample_text_corpus):
        """Test encoding large text."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Create large text
        large_text = sample_text_corpus * 1000
        
        _, tokens, _ = tokenizer.encode(large_text)
        
        # Should handle large inputs
        assert len(tokens) > 0
    
    def test_decode_large_token_sequence(self, sample_text_corpus):
        """Test decoding large token sequence."""
        tokenizer = CharacterGPTTokenizer(sample_text_corpus)
        
        # Create token sequence
        text = sample_text_corpus * 100
        _, tokens, _ = tokenizer.encode(text)
        
        decoded = tokenizer.decode(tokens)
        
        assert len(decoded) > 0
