"""
Summarize training results from CV results JSON.
"""
import json
import os
import sys

def summarize_results(results_file):
    """Summarize CV results."""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        cv_results = json.load(f)
    
    print('='*80)
    print('CROSS-VALIDATION RESULTS SUMMARY')
    print('='*80)
    print()
    
    for fold_result in cv_results:
        fold = fold_result['fold']
        val_loss = fold_result['val_loss']
        train_loss = fold_result.get('train_loss', None)
        epochs = fold_result.get('epochs', None)
        train_losses = fold_result.get('train_losses', [])
        val_losses = fold_result.get('val_losses', [])
        
        print(f'Fold {fold + 1}:')
        if train_loss is not None:
            print(f'  Final Train Loss: {train_loss:.4f}')
        print(f'  Final Val Loss: {val_loss:.4f}')
        if epochs is not None:
            print(f'  Epochs Trained: {epochs}')
        
        # Show loss progression if available
        if train_losses and val_losses:
            print(f'  Loss Progression:')
            num_epochs = min(len(train_losses), len(val_losses))
            # Show first 3, last 3, and best
            if num_epochs > 0:
                print(f'    Epoch 1: Train={train_losses[0]:.4f}, Val={val_losses[0]:.4f}')
                if num_epochs > 1:
                    best_val_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
                    print(f'    Epoch {best_val_idx+1} (best): Train={train_losses[best_val_idx]:.4f}, Val={val_losses[best_val_idx]:.4f}')
                if num_epochs > 1:
                    print(f'    Epoch {num_epochs}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}')
        print()
    
    # Find best fold
    best_fold = min(cv_results, key=lambda x: x['val_loss'])
    print(f'Best Fold: {best_fold["fold"] + 1} (Val Loss: {best_fold["val_loss"]:.4f})')
    print('='*80)
    
    # Summary statistics
    val_losses = [r['val_loss'] for r in cv_results]
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f'\nAverage Validation Loss: {avg_val_loss:.4f}')
    print(f'Std Dev: {(sum((x - avg_val_loss)**2 for x in val_losses) / len(val_losses))**0.5:.4f}')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = 'training/vit_results_synthetic/cv_results.json'
    
    summarize_results(results_file)
