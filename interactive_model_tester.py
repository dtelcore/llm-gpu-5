"""Interactive CLI for testing text generation against a checkpoint."""

import argparse
import os
import time

import numpy as np

from corpus_utils import load_dataset_corpus
from generate import GenerationSession, load_matching_run_config, resolve_recommended_checkpoint


def print_help():
    print("Commands:")
    print("  /help                Show this help")
    print("  /settings            Show current generation settings")
    print("  /pick                Pick and load a different checkpoint")
    print("  /checkpoints         List available checkpoints")
    print("  /temperature <value> Set sampling temperature")
    print("  /max_tokens <value>  Set max generated tokens per turn")
    print("  /delay <value>       Set animation delay in seconds (e.g. 0.01)")
    print("  /checkpoint          Show active checkpoint")
    print("  /exit                Quit the tester")


def list_checkpoint_paths(checkpoint_dir=os.path.join("output", "checkpoints")):
    """Return checkpoint paths sorted by last-modified time, newest first."""
    if not os.path.isdir(checkpoint_dir):
        return []

    candidates = []
    for name in os.listdir(checkpoint_dir):
        if name.endswith(".npz") and not name.endswith(".tmp"):
            candidates.append(os.path.join(checkpoint_dir, name))

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates


def print_checkpoint_list(paths, active_path=None):
    """Print a numbered checkpoint list for manual selection."""
    if not paths:
        print("No checkpoints found in output/checkpoints")
        return

    for index, path in enumerate(paths, start=1):
        marker = "*" if active_path and os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(active_path)) else " "
        print(f"{index:>2}. [{marker}] {path}")


def pick_checkpoint_interactively(current_path=None):
    """Prompt user to choose a checkpoint by number."""
    checkpoint_paths = list_checkpoint_paths()
    if not checkpoint_paths:
        return current_path

    print("Available checkpoints:")
    print_checkpoint_list(checkpoint_paths, active_path=current_path)
    prompt = "Select checkpoint number"
    if current_path:
        prompt += " (Enter to keep current)"
    prompt += ": "

    while True:
        raw_value = input(prompt).strip()
        if not raw_value and current_path:
            return current_path
        if not raw_value:
            print("Please enter a checkpoint number.")
            continue
        try:
            selected_index = int(raw_value)
        except ValueError:
            print("Please enter a valid integer index.")
            continue
        if 1 <= selected_index <= len(checkpoint_paths):
            return checkpoint_paths[selected_index - 1]
        print(f"Please choose a number between 1 and {len(checkpoint_paths)}.")


def animate_print(prefix, text, delay_seconds):
    """Render generated text with a simple typewriter animation."""
    print(prefix, end="", flush=True)
    for ch in text:
        print(ch, end="", flush=True)
        if delay_seconds > 0:
            time.sleep(delay_seconds)
    print()


def resolve_source_docs_for_checkpoint(checkpoint_path):
    """Best-effort source-doc reconstruction for tokenizer stability at inference."""
    run_config = load_matching_run_config(checkpoint_path)
    if run_config is None:
        return None, "run_config_unavailable"

    dataset_name = run_config.get("dataset")
    corpus_size = run_config.get("corpus_size")
    if not dataset_name or not corpus_size:
        return None, "run_config_missing_dataset_or_corpus_size"

    try:
        source_docs, source = load_dataset_corpus(dataset_name, limit=int(corpus_size))
    except Exception:
        return None, "dataset_load_failed"

    if not source_docs:
        return None, "dataset_empty"

    return source_docs, f"{len(source_docs):,} docs ({source})"


def describe_checkpoint_load_failure(checkpoint_path, exc):
    """Build a short user-facing explanation for incompatible checkpoints."""
    checkpoint_vocab_size = None
    try:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint_data:
            if "wte" in checkpoint_data.files:
                checkpoint_vocab_size = int(checkpoint_data["wte"].shape[0])
    except Exception:
        pass

    reason = str(exc)
    if checkpoint_vocab_size is not None:
        reason += f" | checkpoint vocab_size={checkpoint_vocab_size}"

    if load_matching_run_config(checkpoint_path) is None:
        reason += " | no matching run config/tokenizer metadata was found for this checkpoint"

    return reason


def open_generation_session(checkpoint_path, num_heads=None):
    """Open a checkpoint session plus matching source-doc metadata."""
    source_docs, source_docs_label = resolve_source_docs_for_checkpoint(checkpoint_path)
    session = GenerationSession(checkpoint_path, num_heads=num_heads, source_docs=source_docs)
    return session, source_docs_label


def try_open_generation_session(checkpoint_path, num_heads=None):
    """Open a session and return a readable error instead of throwing to the shell."""
    try:
        session, source_docs_label = open_generation_session(checkpoint_path, num_heads=num_heads)
        return session, source_docs_label, None
    except Exception as exc:
        return None, None, describe_checkpoint_load_failure(checkpoint_path, exc)


def main():
    parser = argparse.ArgumentParser(description="Interactive GPT checkpoint tester")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .npz to load; defaults to the best available checkpoint")
    parser.add_argument("--pick", action="store_true", help="Show checkpoint picker on startup")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Default number of tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.6, help="Default sampling temperature")
    parser.add_argument("--animation_delay", type=float, default=0.01, help="Delay per character in animated output")
    parser.add_argument("--num_heads", type=int, default=None, help="Override attention head count if checkpoint metadata is incomplete")
    args = parser.parse_args()

    checkpoint_path = resolve_recommended_checkpoint(args.checkpoint)
    if args.pick or args.checkpoint is None:
        checkpoint_path = pick_checkpoint_interactively(current_path=checkpoint_path)

    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    animation_delay = max(0.0, float(args.animation_delay))

    print("=" * 72)
    print("Interactive Model Tester")
    print("=" * 72)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens : {max_new_tokens}")
    print(f"Anim delay : {animation_delay}")
    print("Type a prompt and press Enter. Use /help for commands.")
    print("=" * 72)

    session = None
    source_docs_label = "unavailable"
    try:
        while session is None:
            session, source_docs_label, open_error = try_open_generation_session(checkpoint_path, num_heads=args.num_heads)
            if session is not None:
                break

            print(f"Unable to load checkpoint: {checkpoint_path}")
            print(f"Reason: {open_error}")
            replacement_checkpoint = pick_checkpoint_interactively(current_path=None)
            if not replacement_checkpoint:
                print("No compatible checkpoint selected. Exiting interactive tester.")
                return
            checkpoint_path = replacement_checkpoint

        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting interactive tester.")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                command, _, value = user_input.partition(" ")
                command = command.lower()
                value = value.strip()

                if command in {"/exit", "/quit"}:
                    print("Exiting interactive tester.")
                    break
                if command == "/help":
                    print_help()
                    continue
                if command == "/settings":
                    print(f"Checkpoint : {checkpoint_path}")
                    print(f"Temperature: {temperature}")
                    print(f"Max tokens : {max_new_tokens}")
                    print(f"Anim delay : {animation_delay}")
                    print(f"Source docs: {source_docs_label}")
                    continue
                if command == "/checkpoints":
                    print_checkpoint_list(list_checkpoint_paths(), active_path=checkpoint_path)
                    continue
                if command == "/pick":
                    selected_checkpoint = pick_checkpoint_interactively(current_path=checkpoint_path)
                    if selected_checkpoint != checkpoint_path:
                        new_session, new_source_docs_label, open_error = try_open_generation_session(
                            selected_checkpoint,
                            num_heads=args.num_heads,
                        )
                        if new_session is None:
                            print(f"Unable to load checkpoint: {selected_checkpoint}")
                            print(f"Reason: {open_error}")
                            print(f"Keeping current checkpoint: {checkpoint_path}")
                            continue

                        if session is not None:
                            session.close()
                        session = new_session
                        checkpoint_path = selected_checkpoint
                        source_docs_label = new_source_docs_label
                        print(f"Loaded checkpoint: {checkpoint_path}")
                        print(f"Source docs: {source_docs_label}")
                    continue
                if command == "/checkpoint":
                    print(checkpoint_path)
                    continue
                if command == "/temperature":
                    if not value:
                        print("Usage: /temperature <value>")
                        continue
                    try:
                        temperature = float(value)
                    except ValueError:
                        print("Temperature must be a number.")
                        continue
                    print(f"Temperature set to {temperature}")
                    continue
                if command == "/max_tokens":
                    if not value:
                        print("Usage: /max_tokens <value>")
                        continue
                    try:
                        max_new_tokens = int(value)
                    except ValueError:
                        print("Max tokens must be an integer.")
                        continue
                    print(f"Max tokens set to {max_new_tokens}")
                    continue
                if command == "/delay":
                    if not value:
                        print("Usage: /delay <value>")
                        continue
                    try:
                        animation_delay = max(0.0, float(value))
                    except ValueError:
                        print("Delay must be a number.")
                        continue
                    print(f"Animation delay set to {animation_delay}")
                    continue

                print("Unknown command. Use /help.")
                continue

            result = session.generate(user_input, max_new_tokens=max_new_tokens, temperature=temperature, stream=False)
            animate_print("model> ", result["completion"], animation_delay)
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    main()