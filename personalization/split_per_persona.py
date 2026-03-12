#!/usr/bin/env python3
"""
Split per-persona datasets into train/val/test sets.

This script iterates through all persona directories in the synthesis output
and applies the dataset splitting logic to each persona's dataset individually.
The split datasets are saved to [workspaceroot]/dataset/[datasetname]/persona_X/
"""

import sys
from pathlib import Path

# Import the split_dataset function directly
sys.path.insert(0, str(Path(__file__).parent / "dataset"))
from split import split_dataset  # type: ignore


def split_per_persona(
    synthesis_output_dir,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42
):
    """
    Split datasets for each persona in the synthesis output directory.
    
    Args:
        synthesis_output_dir: Path to synthesis output directory (e.g., synthesis/output/creative_writing)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)
    """
    synthesis_path = Path(synthesis_output_dir).resolve()
    
    if not synthesis_path.exists():
        print(f"Error: Directory {synthesis_output_dir} does not exist")
        sys.exit(1)
    
    # Get the workspace root and dataset name
    script_dir = Path(__file__).parent
    workspace_root = script_dir
    
    # Extract dataset name from synthesis path (e.g., "creative_writing" from "synthesis/output/creative_writing")
    dataset_name = synthesis_path.name
    
    # Create output directory structure
    output_base = workspace_root / "dataset" / dataset_name
    output_base.mkdir(parents=True, exist_ok=True)
    
    print(f"Workspace root: {workspace_root}")
    print(f"Dataset name: {dataset_name}")
    print(f"Output directory: {output_base}")
    
    # Find all persona directories
    persona_dirs = sorted([d for d in synthesis_path.iterdir() 
                          if d.is_dir() and d.name.startswith('persona_')])
    
    if not persona_dirs:
        print(f"Warning: No persona directories found in {synthesis_output_dir}")
        return
    
    print(f"Found {len(persona_dirs)} persona directories")
    print(f"Split ratios: train={1-val_ratio-test_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    print(f"Random seed: {random_seed}")
    print("-" * 80)
    
    successful = 0
    failed = 0
    
    for persona_dir in persona_dirs:
        persona_dataset = persona_dir / "dataset.json"
        
        if not persona_dataset.exists():
            print(f"⚠️  Skipping {persona_dir.name}: dataset.json not found")
            failed += 1
            continue
        
        print(f"\n📊 Processing {persona_dir.name}...")
        
        # Create output directory for this persona
        persona_output_dir = output_base / persona_dir.name
        persona_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file paths
        train_file = persona_output_dir / "train.json"
        val_file = persona_output_dir / "val.json"
        test_file = persona_output_dir / "test.json"
        
        try:
            # Call split_dataset function directly
            split_dataset(
                input_file=str(persona_dataset),
                train_file=str(train_file),
                val_file=str(val_file),
                test_file=str(test_file),
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed
            )
            
            successful += 1
            print(f"✅ {persona_dir.name} split complete → {persona_output_dir}")
            
        except Exception as e:
            print(f"❌ Error processing {persona_dir.name}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total personas: {len(persona_dirs)}")
    print(f"Successfully split: {successful}")
    print(f"Failed: {failed}")
    
    if successful > 0:
        print(f"\n✅ All datasets split into train.json, val.json, and test.json")
        print(f"   Output location: {output_base}/")
        print(f"   Structure: {output_base}/persona_X/{{train,val,test}}.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_per_persona.py <synthesis_output_directory>")
        print("Example: python split_per_persona.py synthesis/output/creative_writing")
        sys.exit(1)
    
    synthesis_output_dir = sys.argv[1]
    
    # Optional arguments
    val_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    test_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    random_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    
    split_per_persona(
        synthesis_output_dir=synthesis_output_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

