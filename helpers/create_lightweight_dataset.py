#!/usr/bin/env python3
"""
Lightweight FineWeb Alternative - Uses minimal samples for quick testing
Instead of downloading from HuggingFace, this creates a small realistic dataset locally
"""

import os
from pathlib import Path

def create_lightweight_sample():
    """Create a lightweight ~10MB text file with diverse content."""
    
    print("="*80)
    print("LIGHTWEIGHT SAMPLE DATASET CREATOR")
    print("="*80)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "fineweb_lightweight_10mb.txt"
    
    print(f"\nCreating lightweight sample to: {output_file.absolute()}")
    print("This will create ~10MB of diverse text content...\n")
    
    # Sample texts representing different domains from FineWeb
    samples = [
        # Technology
        "Machine learning is a subset of artificial intelligence that focuses on enabling systems to learn from data.",
        "Python has become the lingua franca of data science and machine learning development.",
        "Deep neural networks have revolutionized computer vision and natural language processing.",
        "CUDA programming allows developers to harness the power of NVIDIA GPUs.",
        "Cloud computing has transformed how businesses deploy and scale applications.",
        
        # Science
        "Quantum mechanics describes the behavior of matter at the molecular, atomic, and subatomic levels.",
        "The Standard Model of particle physics describes three of the four known fundamental forces.",
        "DNA is the molecule that carries genetic instructions for the development and functioning of all living things.",
        "Climate change is altering weather patterns, sea levels, and ecosystems worldwide.",
        "Black holes are regions of space where gravity is so strong that nothing can escape.",
        
        # History
        "The Renaissance was a period of European cultural movement spanning the 14th to 17th century.",
        "The Industrial Revolution transformed agrarian societies into industrial ones.",
        "World War II was the deadliest conflict in human history, lasting from 1939 to 1945.",
        "The American Revolution established the United States as an independent nation.",
        "The fall of the Berlin Wall marked the beginning of the end of the Cold War.",
        
        # Literature
        "The works of William Shakespeare have profoundly influenced literature and theater.",
        "Jane Austen wrote novels that critiqued social norms and women's limited options.",
        "Leo Tolstoy explored themes of war, peace, and the human condition.",
        "The Great Gatsby captures the essence of the Jazz Age in America.",
        "Science fiction explores possible futures and alternative worlds.",
        
        # Business
        "Entrepreneurship involves identifying opportunities and creating value through innovation.",
        "Effective leadership requires vision, integrity, and the ability to inspire teams.",
        "Supply chain management coordinates the flow of goods from production to consumption.",
        "Digital marketing leverages online channels to reach and engage customers.",
        "Blockchain technology enables secure and decentralized transactions.",
        
        # Health
        "Regular exercise reduces the risk of cardiovascular disease and obesity.",
        "Nutrition plays a crucial role in maintaining health and preventing disease.",
        "Mental health is as important as physical health for overall well-being.",
        "Sleep deprivation impairs cognitive function and increases health risks.",
        "Vaccines are one of the most effective public health interventions.",
        
        # Environment
        "Renewable energy sources like solar and wind are becoming increasingly cost-effective.",
        "Deforestation contributes to habitat loss and increases carbon dioxide in the atmosphere.",
        "Plastic pollution affects marine ecosystems and wildlife worldwide.",
        "Water scarcity affects billions of people globally due to climate change and overconsumption.",
        "Conservation efforts are critical to protecting endangered species and ecosystems.",
        
        # Philosophy
        "Ethics is the branch of philosophy concerned with questions of right and wrong.",
        "Existentialism emphasizes individual freedom and responsibility.",
        "Utilitarianism judges actions based on their consequences for overall happiness.",
        "Epistemology examines the nature and limits of human knowledge.",
        "Metaphysics explores the fundamental nature of reality.",
    ]
    
    print("[1/2] Generating diverse text samples...")
    
    # Write samples multiple times to reach ~10MB
    with open(output_file, 'w', encoding='utf-8') as f:
        # Each sample is ~100-150 characters
        # To get 10MB (10,485,760 bytes), we need ~70,000-100,000 samples
        target_size_mb = 10
        target_size_bytes = target_size_mb * 1024 * 1024
        
        written_bytes = 0
        sample_count = 0
        
        while written_bytes < target_size_bytes:
            for sample in samples:
                f.write(sample + '\n')
                written_bytes += len(sample.encode('utf-8')) + 1
                sample_count += 1
                
                if written_bytes >= target_size_bytes:
                    break
        
        print(f"  Wrote {sample_count:,} samples")
    
    # Get actual file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    
    print(f"\n[2/2] File statistics:")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Path: {output_file.absolute()}")
    
    # Count lines
    with open(output_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f)
    print(f"  Lines: {num_lines:,}")
    
    print(f"\n[OK] Lightweight dataset created!")
    print("\nThis dataset includes diverse content from:")
    print("  - Technology and AI")
    print("  - Science")
    print("  - History")
    print("  - Literature")
    print("  - Business")
    print("  - Health")
    print("  - Environment")
    print("  - Philosophy")
    
    return True


def main():
    print("\n")
    
    # Check if already exists
    output_file = Path("data/fineweb_lightweight_10mb.txt")
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"Dataset already exists: {output_file}")
        print(f"Size: {size_mb:.2f} MB\n")
        
        choice = input("Recreate? [y/n] (default: n): ").strip().lower() or "n"
        if choice != 'y':
            print("\nUsing existing dataset.")
            return
    
    # Create
    success = create_lightweight_sample()
    
    if success:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\nNow you can train with lightweight dataset:")
        print("  python pipeline.py")
        print("  # Select: fineweb_lightweight_10mb")
        print("\nOr use directly in train.py:")
        print("  corpus = open('data/fineweb_lightweight_10mb.txt').readlines()")


if __name__ == '__main__':
    main()
