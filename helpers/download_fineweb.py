#!/usr/bin/env python3
"""
Stream TinyStories dataset directly to file - download ~100MB shard
Uses streaming=True to avoid loading the full dataset into memory.
"""

import os
from pathlib import Path


def stream_tinystories_subset():
    """Stream TinyStories and write about 500MB of text."""
    output_dir = "data"
    output_file = os.path.join(output_dir, "tinystories_500mb.txt")
    target_bytes = 500 * 1024 * 1024  # 500 MiB

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("TINYSTORIES STREAMING DOWNLOADER")
    print("=" * 80)

    print("\n🌐 Connecting to Hugging Face and opening stream for TinyStories...")
    print("   Dataset: roneneldan/TinyStories")
    print("   Split: train")

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n[ERROR] datasets library not found")
        print("Install with: pip install datasets")
        return False

    try:
        print("\n📥 Opening streaming connection...")
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            streaming=True
        )

        print("✓ Stream opened successfully")
        print(f"📥 Writing text shard to: {output_file}\n")

        bytes_written = 0
        document_count = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for row in dataset:
                text = (row.get("text") or "").strip()
                if not text:
                    continue

                text_chunk = text + "\n\n"
                chunk_bytes = len(text_chunk.encode("utf-8"))

                # Optional: avoid overshooting too much on the last write
                if bytes_written + chunk_bytes > target_bytes:
                    remaining = target_bytes - bytes_written
                    if remaining > 0:
                        encoded = text_chunk.encode("utf-8")
                        truncated = encoded[:remaining]

                        # Keep valid UTF-8 if final cut lands mid-character
                        while truncated:
                            try:
                                final_text = truncated.decode("utf-8")
                                break
                            except UnicodeDecodeError:
                                truncated = truncated[:-1]
                        else:
                            final_text = ""

                        if final_text:
                            f.write(final_text)
                            bytes_written += len(final_text.encode("utf-8"))
                    break

                f.write(text_chunk)
                bytes_written += chunk_bytes
                document_count += 1

                if document_count % 1000 == 0:
                    current_mb = bytes_written / (1024 * 1024)
                    print(f"  → {current_mb:.2f} MiB captured ({document_count:,} stories)...")

        final_mb = bytes_written / (1024 * 1024)
        print("\n✅ Download complete!")
        print(f"   Size: {final_mb:.2f} MiB")
        print(f"   Stories: {document_count:,}")
        print(f"   Location: {Path(output_file).absolute()}")

        print("\n📄 Sample content (first 3 lines):")
        print("-" * 80)
        with open(output_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                sample = line.strip()
                if len(sample) > 100:
                    sample = sample[:100] + "..."
                print(f"   {sample}")
        print("-" * 80)

        return True

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nTroubleshooting:")
        print("  - Check internet connection")
        print("  - Try: pip install --upgrade datasets huggingface-hub")
        print("  - If streaming fails, try without streaming on a smaller split")
        return False


def verify_dataset():
    """Verify downloaded TinyStories shard."""
    output_file = Path("data/tinystories_500mb.txt")

    if not output_file.exists():
        print(f"\n[ERROR] File not found: {output_file}")
        return False

    print(f"\n✓ File verified: {output_file}")

    size_mb = output_file.stat().st_size / (1024 * 1024)

    total_lines = 0
    total_chars = 0
    non_empty_lines = 0

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            total_chars += len(line)
            if line.strip():
                non_empty_lines += 1

    avg_line_len = total_chars / max(1, total_lines)

    print("\n📊 Dataset Statistics:")
    print(f"   Total lines: {total_lines:,}")
    print(f"   Non-empty lines: {non_empty_lines:,}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Average line length: {avg_line_len:.0f} chars")
    print(f"   File size: {size_mb:.2f} MiB")

    return True


def main():
    print()

    output_file = Path("data/tinystories_500mb.txt")
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"Dataset already exists: {output_file}")
        print(f"Size: {size_mb:.2f} MiB\n")

        choice = input("Re-download? [y/n] (default: n): ").strip().lower() or "n"
        if choice != "y":
            print("\nVerifying existing dataset...")
            verify_dataset()
            return

    print("\nStarting download...\n")
    success = stream_tinystories_subset()

    if success:
        print("\n" + "=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        verify_dataset()

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\nNow you can train with TinyStories shard:")
        print("  python pipeline.py")
        print("  # Select: tinystories_500mb")
        print("\nOr start training directly:")
        print("  python train.py")


if __name__ == "__main__":
    main()