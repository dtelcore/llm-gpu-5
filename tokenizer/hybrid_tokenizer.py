import os
import re
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is an optional UX dependency
    def tqdm(iterable, **kwargs):
        return iterable

# Keep Python 3.8 isolated DLL environment mapping active for Windows GPU runtimes
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin"
if hasattr(os, "add_dll_directory") and os.path.exists(CUDA_BIN):
    os.add_dll_directory(CUDA_BIN)


class TokenizerRunResults:
    """Container for the one-step end_encoder pipeline results."""

    def __init__(self, encoded_pieces, encoded_ids, gpu_aligned_matrix, decoded_text, encode_logs, decode_logs):
        self.encoded_pieces = encoded_pieces
        self.encoded_ids = encoded_ids
        self.gpu_aligned_matrix = gpu_aligned_matrix
        self.decoded_text = decoded_text
        self.encode_logs = encode_logs
        self.decode_logs = decode_logs


class CharacterGPTTokenizer:
    """Dependency-free hybrid tokenizer with character fallback.

    The tokenizer keeps exact character coverage for robustness while also
    adding frequent word and punctuation pieces up to a configurable vocab cap.
    That raises the effective vocab beyond pure character tokenization without
    introducing external dependencies.
    """

    PIECE_PATTERN = re.compile(r"\s+|[\w]+|[^\w\s]", re.UNICODE)
    BYTE_FALLBACK_WINDOW = 256
    TOKENIZER_VERSION = 2

    def __init__(self, docs: list, max_vocab_size: int = 4096, min_piece_frequency: int = 2) -> None:
        self.max_vocab_size = max(self.BYTE_FALLBACK_WINDOW + 2, int(max_vocab_size))
        self.min_piece_frequency = max(1, int(min_piece_frequency))

        normalized_docs = [str(doc) for doc in docs]
        char_counts = {}
        for doc in tqdm(normalized_docs, desc="Tokenizer: counting chars", unit="doc"):
            for char in doc:
                char_counts[char] = char_counts.get(char, 0) + 1

        base_chars = list(char_counts)
        if not base_chars:
            base_chars = [" "]
            char_counts[" "] = 1

        base_chars.sort(key=lambda char: (-char_counts[char], char))

        piece_counts = {}
        for doc in tqdm(normalized_docs, desc="Tokenizer: counting pieces", unit="doc"):
            for piece in self.PIECE_PATTERN.findall(doc):
                if len(piece) <= 1:
                    continue
                piece_counts[piece] = piece_counts.get(piece, 0) + 1

        candidate_pieces = [
            piece for piece, count in piece_counts.items()
            if count >= self.min_piece_frequency and piece not in base_chars
        ]
        candidate_pieces.sort(key=lambda piece: (-piece_counts[piece], -len(piece), piece))

        reserved_tokens = self.BYTE_FALLBACK_WINDOW + 1
        base_vocab_limit = max(1, self.max_vocab_size - reserved_tokens)

        vocab_tokens = []
        for char in base_chars:
            if len(vocab_tokens) >= base_vocab_limit:
                break
            vocab_tokens.append(char)

        for piece in candidate_pieces:
            if len(vocab_tokens) >= base_vocab_limit:
                break
            vocab_tokens.append(piece)

        self.uchars = vocab_tokens
        self.char_to_id = {token: idx for idx, token in enumerate(self.uchars)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.uchars)}

        self.byte_offset = len(self.uchars)
        self.byte_token_ids = set(range(self.byte_offset, self.byte_offset + self.BYTE_FALLBACK_WINDOW))

        for byte_value in range(self.BYTE_FALLBACK_WINDOW):
            self.id_to_token[self.byte_offset + byte_value] = f"<byte_{byte_value}>"

        # Define a special Beginning of Sequence (BOS) token at the end of the vocabulary
        self.BOS_ID = self.byte_offset + self.BYTE_FALLBACK_WINDOW
        self.vocab_size = self.BOS_ID + 1

        # Use the BOS_ID as the padding element when formatting fixed matrices for PyCUDA
        self.PAD_ID = self.BOS_ID

    def _encode_char_with_fallback(self, char: str, pieces: list, ids: list) -> None:
        if char in self.char_to_id:
            pieces.append(char)
            ids.append(self.char_to_id[char])
            return

        for byte_value in char.encode('utf-8'):
            pieces.append(f"<byte_{byte_value}>")
            ids.append(self.byte_offset + byte_value)

    def _flush_byte_buffer(self, byte_buffer: bytearray, decoded_parts: list) -> None:
        if not byte_buffer:
            return
        decoded_parts.append(byte_buffer.decode('utf-8', errors='replace'))
        byte_buffer.clear()

    def encode(self, text: str, verbose: bool = False) -> tuple:
        """Converts a raw text string into token pieces and integer IDs."""
        pieces = []
        ids = []

        for chunk in self.PIECE_PATTERN.findall(str(text)):
            if chunk in self.char_to_id:
                pieces.append(chunk)
                ids.append(self.char_to_id[chunk])
                continue

            for char in chunk:
                self._encode_char_with_fallback(char, pieces, ids)

        log_lines = [
            f"[encode] raw_text='{text}'",
            f"[encode] pieces={pieces}",
            f"[encode] ids={ids}",
        ]
        if verbose:
            for line in log_lines:
                print(line)
        return pieces, ids, log_lines

    def encode_batch_gpu_aligned(self, batch_texts: list, max_sequence_length: int) -> np.ndarray:
        """
        Packs a batch of strings into a structured 2D NumPy array wrapped with
        BOS tokens and padded to a uniform sequence length, ready for GPU memory blocks.
        """
        batch_size = len(batch_texts)
        tensor_matrix = np.full((batch_size, max_sequence_length), fill_value=self.PAD_ID, dtype=np.int32)

        print(f"  [Encoding] Processing {batch_size:,} documents...")
        for idx, text in enumerate(batch_texts):
            if (idx + 1) % 10000 == 0:
                print(f"    → Encoded {(idx + 1):,} / {batch_size:,} documents... ({100 * (idx + 1) / batch_size:.1f}%)")

            _, core_ids, _ = self.encode(text, verbose=False)
            full_tokens = [self.BOS_ID] + core_ids
            if len(full_tokens) > max_sequence_length:
                full_tokens = full_tokens[:max_sequence_length]

            tensor_matrix[idx, :len(full_tokens)] = full_tokens

        print(f"  [OK] All {batch_size:,} documents encoded successfully")
        return tensor_matrix

    def decode(self, ids: list, verbose: bool = False) -> tuple:
        """Converts an integer token stream or matrix slice back into a readable string."""
        sanitized_ids = [int(token_id) for token_id in ids if int(token_id) != self.BOS_ID]
        decoded_parts = []
        byte_buffer = bytearray()

        for token_id in sanitized_ids:
            if self.byte_offset <= token_id < self.BOS_ID:
                byte_buffer.append(token_id - self.byte_offset)
                continue

            self._flush_byte_buffer(byte_buffer, decoded_parts)
            decoded_parts.append(self.uchars[token_id])

        self._flush_byte_buffer(byte_buffer, decoded_parts)
        text = ''.join(decoded_parts)

        log_lines = [
            f"[decode] input_ids={list(ids)}",
            f"[decode] sanitized_ids={sanitized_ids}",
            f"[decode] decoded_text='{text}'",
        ]
        if verbose:
            for line in log_lines:
                print(line)
        return text, log_lines

    def end_encoder(self, text_or_batch, max_sequence_length: int = 16, verbose: bool = False) -> TokenizerRunResults:
        """
        Executes an isolated one-step validation cycle.
        Runs encoding pipelines, generates GPU-ready matrices, runs decoding loops,
        and aggregates all cross-sectional outputs and metadata.
        """
        batch = [text_or_batch] if isinstance(text_or_batch, str) else list(text_or_batch)

        if verbose:
            print(f"\n⚡ Starting One-Step Pipeline (`end_encoder`) for Batch Size: {len(batch)}")
            print("-" * 70)

        all_pieces = []
        all_ids = []
        combined_encode_logs = []

        for text in batch:
            pieces, ids, e_logs = self.encode(text, verbose=verbose)
            all_pieces.extend(pieces)
            all_ids.extend(ids)
            combined_encode_logs.extend(e_logs)

        gpu_matrix = self.encode_batch_gpu_aligned(batch, max_sequence_length=max_sequence_length)
        decoded_text_string, combined_decode_logs = self.decode(all_ids, verbose=verbose)

        if verbose:
            print("-" * 70)
            print("🚀 One-step pipeline verification complete.")

        return TokenizerRunResults(
            encoded_pieces=all_pieces,
            encoded_ids=all_ids,
            gpu_aligned_matrix=gpu_matrix,
            decoded_text=decoded_text_string,
            encode_logs=combined_encode_logs,
            decode_logs=combined_decode_logs,
        )

    def save_vocab(self, filepath: str) -> None:
        """Export token mappings and metadata to JSON."""
        import json
        from pathlib import Path
        payload = {
            "vocab_size": getattr(self, "vocab_size", len(self.uchars)),
            "uchars": self.uchars,
            "char_to_id": self.char_to_id,
            "id_to_token": self.id_to_token
        }
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath_obj, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Tokenizer state frozen and saved to {filepath_obj}")

    @classmethod
    def load_vocab(cls, filepath: str) -> "CharacterGPTTokenizer":
        """Instantiate tokenizer directly from a saved JSON state."""
        import json
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
            
        instance = cls.__new__(cls)
        
        instance.uchars = payload["uchars"]
        instance.char_to_id = payload["char_to_id"]
        instance.id_to_token = {int(k): v for k, v in payload["id_to_token"].items()}
        instance.vocab_size = payload["vocab_size"]
        
        instance.BYTE_FALLBACK_WINDOW = 256
        instance.byte_offset = len(instance.uchars)
        instance.byte_token_ids = set(range(instance.byte_offset, instance.byte_offset + instance.BYTE_FALLBACK_WINDOW))
        instance.BOS_ID = instance.byte_offset + instance.BYTE_FALLBACK_WINDOW
        instance.PAD_ID = instance.BOS_ID
        
        print(f"[INFO] Tokenizer successfully hydrated from static map: {filepath}")
        return instance
