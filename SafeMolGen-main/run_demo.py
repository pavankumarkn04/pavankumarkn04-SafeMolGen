#!/usr/bin/env python
"""Simple demo to test SafeMolGen API locally without server."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles, calculate_properties

def main():
    print("=" * 60)
    print("SafeMolGen Demo - Testing Model Loading")
    print("=" * 60)
    
    # Test 1: Load generator model
    print("\n[1] Loading generator model...")
    try:
        generator_path = _PROJECT_ROOT / "checkpoints" / "generator"
        model_file = generator_path / "model.pt"
        
        if not model_file.exists():
            print(f"❌ Model not found at {model_file}")
            print("\n📋 SOLUTION:")
            print("   Your friend's laptop has the real models.")
            print("   Ask them to copy the entire 'checkpoints/' folder to you.")
            print("   Then paste it into: SafeMolGen-main/checkpoints/")
            print(f"\n   See MODELS_NEEDED.txt for detailed instructions")
            return
        
        # Check file size
        file_size = model_file.stat().st_size
        if file_size < 10_000_000:  # Less than 10MB = corrupted/placeholder
            print(f"❌ Model file is corrupted (only {file_size} bytes, should be 100+ MB)")
            print("\n📋 SOLUTION:")
            print("   The model.pt file is incomplete or placeholder.")
            print("   Ask your friend to copy the real model from their laptop:")
            print(f"   - Their SafeMolGen/checkpoints/generator/model.pt")
            print(f"   - To your SafeMolGen/checkpoints/generator/model.pt")
            print(f"\n   See MODELS_NEEDED.txt for detailed instructions")
            return
        
        generator = SafeMolGen.from_pretrained(str(generator_path), device="cpu")
        print("✅ Generator loaded successfully!")
        print(f"   Model: {type(generator).__name__}")
    except Exception as e:
        print(f"❌ Failed to load generator: {e}")
        print("\n📋 SOLUTION:")
        print("   See MODELS_NEEDED.txt for how to get the models from your friend")
        return
    
    # Test 2: Generate molecules
    print("\n[2] Generating 5 molecules...")
    try:
        samples = generator.generate(n=5, temperature=0.8, top_k=40, device="cpu")
        print("✅ Generated molecules:")
        for i, smi in enumerate(samples, 1):
            valid = validate_smiles(smi)
            status = "✓" if valid else "✗"
            print(f"   {i}. {status} {smi}")
    except Exception as e:
        print(f"❌ Failed to generate: {e}")
        return
    
    # Test 3: Analyze molecules
    print("\n[3] Analyzing molecules...")
    try:
        for smi in samples[:2]:
            if validate_smiles(smi):
                props = calculate_properties(smi)
                if props:
                    print(f"\n   SMILES: {smi}")
                    print(f"   LogP: {props.get('logp', 'N/A'):.2f}")
                    print(f"   MW: {props.get('mw', 'N/A'):.1f}")
                    print(f"   HBA: {props.get('hba', 'N/A')}")
                    print(f"   HBD: {props.get('hbd', 'N/A')}")
                    print(f"   TPSA: {props.get('tpsa', 'N/A'):.1f}")
    except Exception as e:
        print(f"❌ Failed to analyze: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)
    print("\nNow you can run the API server:")
    print("  uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000")
    print("\nOr visit: http://127.0.0.1:8000/docs")
    print("=" * 60)

if __name__ == "__main__":
    main()
