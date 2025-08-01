#!/usr/bin/env python3
"""
Sequential Ablation Study Runner for TomNetCausal

This script runs all 4 ablation configurations sequentially:
1. none: No VAE components - direct graph encoder to goal predictor
2. belief: Only belief VAE active
3. belief_desire: Belief and desire VAEs active  
4. full: All three VAEs active (belief + desire + intention)

Each ablation gets its own wandb run for comparison.
"""

import os
import sys
import time
import torch
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from real_world_src.models.tomnet_causal_ablation_trainer_base import train_pipeline

def run_ablation_study(
    epochs=50,
    batch_size=1024,
    max_seq_len=100,
    top_k=5,
    gpu=0,
    data_dir="./data/1k/",
    node_mapping_path=None,
    save_node_mapping_path=None,
    model_base_name="/goal_loss_30ep/tomnet_causal_g_ablation",
    log_wandb=True
):
    """
    Run all ablation studies sequentially.
    
    Args:
        epochs: Number of training epochs for each ablation
        batch_size: Batch size for training
        max_seq_len: Maximum sequence length for trajectories  
        top_k: Top-k accuracy to compute
        gpu: GPU device to use
        data_dir: Directory containing the dataset
        node_mapping_path: Path to existing node mapping
        save_node_mapping_path: Path to save node mapping
        model_base_name: Base name for saved models
        log_wandb: Whether to log to wandb
    """
    
    # Define ablation modes in order of complexity
    ablation_modes = [
        ('none', 'No VAE - Direct graph encoder to goal predictor'),
        ('belief', 'Belief VAE only'),
        ('belief_desire', 'Belief + Desire VAEs'),
        ('full', 'Full hierarchical VAE (Belief + Desire + Intention)')
    ]
    
    print("="*80)
    print("🧠 TOMNET CAUSAL ABLATION STUDY")
    print("="*80)
    print(f"📊 Total ablations to run: {len(ablation_modes)}")
    print(f"⏱️  Epochs per ablation: {epochs}")
    print(f"🔥 Batch size: {batch_size}")
    print(f"🎯 GPU: {gpu}")
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Models will be saved to: ./ablation_trained_models/")
    print("="*80)
    
    # Create ablation results directory
    os.makedirs("./ablation_trained_models/goal_loss_30ep", exist_ok=True)
    
    # Track timing and results
    results_summary = []
    total_start_time = time.time()
    
    for i, (mode, description) in enumerate(ablation_modes, 1):
        print(f"\n🚀 STARTING ABLATION {i}/{len(ablation_modes)}: {mode.upper()}")
        print(f"📝 Description: {description}")
        print("-" * 60)
        
        ablation_start_time = time.time()
        
        try:
            # Run training pipeline for this ablation
            result = train_pipeline(
                epochs=epochs,
                batch_size=batch_size,
                log_wandb=log_wandb,
                max_seq_len=max_seq_len,
                top_k=top_k,
                gpu=gpu,
                data_dir=data_dir,
                node_mapping_path=node_mapping_path,
                save_node_mapping_path=save_node_mapping_path,
                save_model=True,
                model_save_name=model_base_name,
                ablation_mode=mode
            )
            
            ablation_duration = time.time() - ablation_start_time
            status = "✅ SUCCESS"
            
            print(f"\n{status} Ablation '{mode}' completed in {ablation_duration/60:.1f} minutes")
            if result and 'best_val_acc' in result:
                print(f"🏆 Best validation accuracy: {result['best_val_acc']:.4f} at epoch {result['best_epoch']}")
            
        except Exception as e:
            ablation_duration = time.time() - ablation_start_time  
            status = "❌ FAILED"
            print(f"\n{status} Ablation '{mode}' failed after {ablation_duration/60:.1f} minutes")
            print(f"Error: {str(e)}")
        
        # Record results
        results_summary.append({
            'mode': mode,
            'description': description,
            'status': status,
            'duration_minutes': ablation_duration / 60,
            'final_model_path': f"./ablation_trained_models/{model_base_name}_{mode}.pth",
            'best_model_path': f"./ablation_trained_models/{model_base_name}_{mode}_best.pth",
            'best_val_acc': result.get('best_val_acc', 0.0) if 'result' in locals() and result else 0.0,
            'best_epoch': result.get('best_epoch', 0) if 'result' in locals() and result else 0
        })
        
        print("-" * 60)
        
        # Optional: Add delay between ablations to prevent resource conflicts
        if i < len(ablation_modes):
            print("⏳ Waiting 30 seconds before next ablation...")
            time.sleep(30)
    
    # Print final summary
    total_duration = time.time() - total_start_time
    
    print("\n" + "="*80)
    print("📋 ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"🕐 Total time: {total_duration/3600:.1f} hours ({total_duration/60:.1f} minutes)")
    print()
    
    for result in results_summary:
        print(f"{result['status']} {result['mode']:15} | {result['duration_minutes']:6.1f} min | {result['description']}")
        if result['status'] == "✅ SUCCESS":
            print(f"    � Final model: {result['final_model_path']}")
            if result['best_val_acc'] > 0:
                print(f"    🏆 Best model: {result['best_model_path']} (acc: {result['best_val_acc']:.4f} @ epoch {result['best_epoch']})")
        print()
    
    # Count successes and failures
    successful = sum(1 for r in results_summary if "SUCCESS" in r['status'])
    failed = len(results_summary) - successful
    
    print(f"📊 Results: {successful}/{len(results_summary)} successful, {failed} failed")
    
    if log_wandb:
        print(f"📈 View results in wandb project: tom-graph-causalnet-distributional")
        print(f"🔗 Each ablation has its own run: ablation_none, ablation_belief, etc.")
    
    print("="*80)
    print("🎉 Ablation study complete!")
    print("="*80)
    
    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Run all TomNet Causal ablation studies sequentially")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per ablation")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--max_seq_len", type=int, default=100, help="Maximum sequence length for trajectories")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k accuracy to compute")
    
    # Hardware and data
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use (e.g., 0 for cuda:0)")
    parser.add_argument("--data_dir", type=str, default="./data/1k/", help="Directory containing the dataset")
    parser.add_argument("--node_mapping_path", type=str, help="Path to existing node mapping")
    parser.add_argument("--save_node_mapping_path", type=str, default="./data/1k/node_mapping.pkl", help="Path to save node mapping")
    
    # Model and logging
    parser.add_argument("--model_base_name", type=str, default="/goal_loss/tomnet_causal_ablation_30ep", help="Base name for saved models")
    parser.add_argument("--no_wandb", action='store_true', help="Disable wandb logging")
    
    # Quick test mode
    parser.add_argument("--quick_test", action='store_true', help="Run with minimal epochs for testing (5 epochs)")
    
    parser.add_argument("--log_wandb", action='store_true', help="Enable wandb logging (default: True)")
    if not parser.get_default('log_wandb'):
        print("⚠️  Wandb logging is disabled. Results will not be logged to wandb.")

    args = parser.parse_args()
    
    # Adjust epochs for quick test
    if args.quick_test:
        args.epochs = 5
        print("🧪 Quick test mode: Using 5 epochs per ablation")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🔥 CUDA available: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("⚠️  CUDA not available, using CPU")
        args.gpu = None
    
    # Run the ablation study
    results = run_ablation_study(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        top_k=args.top_k,
        gpu=args.gpu,
        data_dir=args.data_dir,
        node_mapping_path=args.node_mapping_path,
        save_node_mapping_path=args.save_node_mapping_path,
        model_base_name=args.model_base_name,
        log_wandb=not args.no_wandb
    )
    
    return results

if __name__ == "__main__":
    main()
