import os
import sys

# Ensure the parent workspace directory is in the system path to handle local imports cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to import your custom tokenizer class
try:
    from tokenizer import CharacterGPTTokenizer
except ImportError:
    # Fallback if executing directly inside the folder without package structures initialized
    from tokenizer import CharacterGPTTokenizer

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # 1. Initialize with a foundational corpus so you can type common test phrases
    # without running into immediate out-of-vocabulary ValueErrors.
    seed_corpus = [
        "cuda", "kepler", "gt730", "gpu", "matrix", 
        "abcdefghijklmnopqrstuvwxyz", 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 
        "0123456789 _,.!?"
    ]
    
    tokenizer = CharacterGPTTokenizer(seed_corpus)
    
    clear_screen()
    print("======================================================================")
    print("⚡ Legacy Kepler GPT - Interactive Tokenizer CLI Helper ⚡")
    print("======================================================================")
    print(f"Vocabulary Size: {tokenizer.vocab_size} unique elements")
    print(f"BOS / PAD Structural Identifier: {tokenizer.BOS_ID}")
    print("======================================================================")

    while True:
        print("\n--- Available Testing Modes ---")
        print("1. [Encode] Text ➔ Token IDs")
        print("2. [Decode] Token IDs ➔ Text")
        print("3. [Full Run] Verbose Pipeline Execution")
        print("4. Exit CLI")
        
        choice = input("\nSelect a mode (1-4): ").strip()
        
        if choice == '1':
            print("\n--- Mode 1: Encode Text ---")
            text_input = input("Enter text string to encode: ")
            if not text_input:
                print("[!] Empty string provided.")
                continue
            try:
                pieces, ids, _ = tokenizer.encode(text_input, verbose=False)
                print(f"\n➔ Input Text: '{text_input}'")
                print(f"➔ Token IDs:  {ids}")
                print(f"➔ Character Pieces: {pieces}")
            except ValueError as e:
                print(f"\n[ERROR] Character mismatch: {e}")
                print("Tip: Stick to alphanumeric characters and basic punctuation present in seed_corpus.")
                
        elif choice == '2':
            print("\n--- Mode 2: Decode Token IDs ---")
            id_input = input("Enter integer token IDs (separated by spaces or commas): ")
            if not id_input:
                print("[!] No tokens provided.")
                continue
            try:
                # Normalize commas to spaces and parse out list integers
                sanitized_input = id_input.replace(",", " ")
                token_ids = [int(tid) for tid in sanitized_input.split()]
                
                decoded_text, _ = tokenizer.decode(token_ids, verbose=False)
                print(f"\n➔ Input IDs:    {token_ids}")
                print(f"➔ Decoded Text: '{decoded_text}'")
            except ValueError:
                print("\n[ERROR] Could not parse inputs. Ensure you are only inputting integers.")
            except IndexError:
                print(f"\n[ERROR] Token ID out of vocabulary range (0-{tokenizer.vocab_size - 1}).")

        elif choice == '3':
            print("\n--- Mode 3: Full Pipeline Execution (Verbose) ---")
            text_input = input("Enter text string for pipeline validation: ")
            if not text_input:
                print("[!] Empty string provided.")
                continue
            try:
                # Executes the one-step isolated verification loop inside tokenizer.py
                results = tokenizer.end_encoder(text_input, max_sequence_length=16, verbose=True)
                
                print("\n--- Final Aggregated Outputs ---")
                print(f"➔ Extracted Token String List: {results.encoded_pieces}")
                print(f"➔ Extracted Token ID Layout:   {results.encoded_ids}")
                print(f"➔ Final Decoded Output Text:   '{results.decoded_text}'")
                print("\n➔ GPU Aligned Input Matrix Layer (Max Len: 16):")
                print(results.gpu_aligned_matrix)
                
            except ValueError as e:
                print(f"\n[ERROR] Pipeline dropped: {e}")
                
        elif choice == '4':
            print("\nExiting interactive verification environment. Back to work!\n")
            break
        else:
            print("\n[!] Invalid choice. Please pick an option between 1 and 4.")

if __name__ == "__main__":
    main()