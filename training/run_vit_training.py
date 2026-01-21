"""
Main script to run the complete ViT training pipeline:
1. Create patient stratification
2. Extract ViT features
3. Train with cross-validation
4. Test on held-out set
"""
import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run complete ViT training pipeline')
    parser.add_argument('--annotation_csv', type=str,
                       default='evaluation/dataset/90_FeatureAnnotation.csv',
                       help='Path to annotation CSV')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--splits_dir', type=str, default='training/splits',
                       help='Directory for train/test splits')
    parser.add_argument('--output_dir', type=str, default='training/vit_results',
                       help='Output directory for results')
    parser.add_argument('--vit_model', type=str, default='google/vit-base-patch16-224',
                       help='ViT model name')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--skip_stratification', action='store_true',
                       help='Skip stratification if already done')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training if already done')
    
    args = parser.parse_args()
    
    # Step 1: Create patient stratification
    if not args.skip_stratification:
        print("="*80)
        print("Step 1: Creating patient stratification (60 train, 30 test)")
        print("="*80)
        
        from patient_stratification import create_patient_stratification
        create_patient_stratification(args.annotation_csv, args.splits_dir)
    else:
        print("Skipping stratification (already done)")
    
    # Step 2: Train with cross-validation
    if not args.skip_training:
        print("\n" + "="*80)
        print("Step 2: Training ViT classifier with 3-fold cross-validation")
        print("="*80)
        
        train_cmd = [
            sys.executable, 'train_vit_classifier.py',
            '--annotation_csv', args.annotation_csv,
            '--video_dir', args.video_dir,
            '--splits_dir', args.splits_dir,
            '--output_dir', args.output_dir,
            '--vit_model', args.vit_model,
            '--batch_size', str(args.batch_size),
            '--learning_rate', str(args.learning_rate),
            '--max_epochs', str(args.max_epochs),
            '--patience', str(args.patience),
            '--device', args.device
        ]
        
        subprocess.run(train_cmd, cwd=os.path.dirname(__file__))
    else:
        print("Skipping training (already done)")
    
    # Step 3: Test on held-out set
    print("\n" + "="*80)
    print("Step 3: Testing on held-out test set (30 patients)")
    print("="*80)
    
    test_cmd = [
        sys.executable, 'test_vit_classifier.py',
        '--annotation_csv', args.annotation_csv,
        '--video_dir', args.video_dir,
        '--splits_dir', args.splits_dir,
        '--model_dir', args.output_dir,
        '--output_csv', os.path.join(args.output_dir, 'test_results.csv'),
        '--vit_model', args.vit_model,
        '--batch_size', str(args.batch_size),
        '--device', args.device
    ]
    
    subprocess.run(test_cmd, cwd=os.path.dirname(__file__))
    
    print("\n" + "="*80)
    print("Pipeline complete!")
    print(f"Test results saved to: {os.path.join(args.output_dir, 'test_results.csv')}")
    print("="*80)

if __name__ == '__main__':
    main()
