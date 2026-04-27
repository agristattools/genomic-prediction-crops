# main.py
"""
MASTER PIPELINE — Genomic Prediction System for Crop Breeding
=============================================================
Runs the complete pipeline:
  1. Data Generation (or loading)
  2. Data Preprocessing
  3. Feature Engineering
  4. GBLUP Baseline Model
  5. Machine Learning Models (RF + XGBoost)
  6. Deep Learning Model
  7. Genotype Ranking & Breeding Recommendations

Usage:
  python main.py                    # Run complete pipeline
  python main.py --skip-ml          # Skip RF/XGBoost (faster)
  python main.py --data-only        # Only generate and preprocess data
"""
import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "█" * 70)
    print(f"█  {title}")
    print("█" * 70)


def print_step(step_num, step_name):
    """Print step information."""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}: {step_name}")
    print(f"{'='*70}")


def run_pipeline(skip_ml=False, data_only=False, skip_dl=False):
    """
    Run the complete genomic prediction pipeline.
    """
    start_time = time.time()
    
    print_header("GENOMIC PREDICTION SYSTEM FOR CROP BREEDING")
    print(f"  Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project: Multi-modal G×E Genomic Prediction")
    print(f"  Options: skip_ml={skip_ml}, data_only={data_only}, skip_dl={skip_dl}")
    
    # ================================================================
    # STEP 0: Verify Environment
    # ================================================================
    print_step(0, "Environment Verification")
    
    try:
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        import sklearn
        import xgboost
        
        print(f"  ✓ Python: {sys.version.split()[0]}")
        print(f"  ✓ NumPy: {np.__version__}")
        print(f"  ✓ Pandas: {pd.__version__}")
        print(f"  ✓ Scikit-learn: {sklearn.__version__}")
        print(f"  ✓ XGBoost: {xgboost.__version__}")
        print(f"  ✓ TensorFlow: {tf.__version__}")
        print(f"  ✓ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except Exception as e:
        print(f"  ✗ Environment Error: {e}")
        return
    
    # ================================================================
    # STEP 1: Data Generation
    # ================================================================
    print_step(1, "Data Generation")
    
    data_files_exist = all([
        os.path.exists("data/raw/genotypes.csv"),
        os.path.exists("data/raw/phenotypes.csv"),
        os.path.exists("data/raw/weather.csv"),
    ])
    
    if data_files_exist:
        print("  ✓ Data files already exist. Skipping generation.")
        print("    To regenerate, delete data/raw/ and rerun.")
    else:
        print("  Generating synthetic G2F-style datasets...")
        from src.data_processing.generate_synthetic_data import main as gen_data
        gen_data()
        print("  ✓ Data generation complete.")
    
    # ================================================================
    # STEP 2: Data Preprocessing
    # ================================================================
    print_step(2, "Data Preprocessing")
    
    processed_exists = all([
        os.path.exists("data/processed/genotypes_imputed.csv"),
        os.path.exists("data/processed/train_data.csv"),
        os.path.exists("data/processed/test_data.csv"),
    ])
    
    if processed_exists:
        print("  ✓ Processed data already exists. Skipping.")
        print("    To reprocess, delete data/processed/ and rerun.")
    else:
        print("  Running preprocessing pipeline...")
        from src.data_processing.preprocess import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.run_full_pipeline()
        print("  ✓ Preprocessing complete.")
    
    if data_only:
        print_header("DATA PREPARATION COMPLETE (--data-only mode)")
        elapsed = time.time() - start_time
        print(f"  Total time: {elapsed:.1f} seconds")
        return
    
    # ================================================================
    # STEP 3: Feature Engineering
    # ================================================================
    print_step(3, "Genomic Feature Engineering")
    
    features_exist = all([
        os.path.exists("data/processed/pca_features.csv"),
        os.path.exists("data/processed/genomic_relationship_matrix.csv"),
        os.path.exists("data/processed/blup_estimates.csv"),
    ])
    
    if features_exist:
        print("  ✓ Features already exist. Skipping.")
    else:
        print("  Computing PCA, GRM, and BLUPs...")
        from src.feature_engineering.genomic_features import GenomicFeatureEngineer
        engineer = GenomicFeatureEngineer(random_seed=42)
        engineer.run_full_feature_engineering(n_pca_components=10)
        print("  ✓ Feature engineering complete.")
    
    # ================================================================
    # STEP 4: GBLUP Baseline Model
    # ================================================================
    print_step(4, "GBLUP Baseline Model")
    
    gblup_exists = os.path.exists("outputs/gblup_predictions.csv")
    
    if gblup_exists:
        print("  ✓ GBLUP results already exist. Skipping.")
        # Load metrics
        gblup_results = pd.read_csv("outputs/gblup_predictions.csv")
        from sklearn.metrics import mean_squared_error
        gblup_rmse = np.sqrt(mean_squared_error(
            gblup_results['Yield_bu_ac_norm'], 
            gblup_results['GBLUP_Prediction']
        ))
        gblup_corr = np.corrcoef(
            gblup_results['Yield_bu_ac_norm'], 
            gblup_results['GBLUP_Prediction']
        )[0, 1]
    else:
        print("  Training GBLUP model...")
        from src.models.gblup_model import GBLUP
        train_df = pd.read_csv("data/processed/train_data.csv")
        test_df = pd.read_csv("data/processed/test_data.csv")
        grm_df = pd.read_csv("data/processed/genomic_relationship_matrix.csv", index_col=0)
        
        gblup = GBLUP()
        gblup.fit(train_df, grm_df, trait_col='Yield_bu_ac_norm')
        predictions = gblup.predict(test_df, grm_df)
        metrics = gblup.evaluate(predictions)
        
        gblup_rmse = metrics['RMSE']
        gblup_corr = metrics['Correlation']
        print("  ✓ GBLUP training complete.")
    
    print(f"  GBLUP RMSE: {gblup_rmse:.4f}, Correlation: {gblup_corr:.4f}")
    
    # ================================================================
    # STEP 5: Machine Learning Models (Optional)
    # ================================================================
    if not skip_ml:
        print_step(5, "Machine Learning Models (RF + XGBoost)")
        
        ml_exists = os.path.exists("outputs/ml_predictions.csv")
        
        if ml_exists:
            print("  ✓ ML predictions already exist. Skipping.")
        else:
            print("  Training Random Forest and XGBoost...")
            print("  (This may take 2-5 minutes)")
            from src.models.ml_models import GenomicMLModels
            
            train_df = pd.read_csv("data/processed/train_data.csv")
            test_df = pd.read_csv("data/processed/test_data.csv")
            pca_df = pd.read_csv("data/processed/pca_features.csv", index_col=0)
            
            ml = GenomicMLModels(random_seed=42)
            
            X_train, y_train, _ = ml.prepare_features(train_df, None, pca_df, None)
            X_test, y_test, _ = ml.prepare_features(test_df, None, pca_df, None)
            
            ml.train_random_forest(X_train, y_train, X_test, y_test)
            ml.train_xgboost(X_train, y_train, X_test, y_test)
            ml.compare_models()
            print("  ✓ ML training complete.")
    else:
        print_step(5, "Machine Learning Models — SKIPPED")
    
    # ================================================================
    # STEP 6: Deep Learning Model
    # ================================================================
    if not skip_dl:
        print_step(6, "Multi-Modal Deep Learning Model")
        
        dl_exists = os.path.exists("outputs/dl_predictions.csv")
        
        if dl_exists:
            print("  ✓ DL predictions already exist. Skipping.")
        else:
            print("  Building and training multi-modal neural network...")
            print("  (This may take 5-10 minutes)")
            from src.models.dl_model import MultiModalGenomicModel
            
            train_df = pd.read_csv("data/processed/train_data.csv")
            test_df = pd.read_csv("data/processed/test_data.csv")
            geno_df = pd.read_csv("data/processed/genotypes_imputed.csv", index_col=0)
            
            env_cols = [
                'GDD', 'HeatStress_Days', 'MaxDrySpell_Days',
                'TotalPrecip_mm', 'MeanTemp_C', 'DiurnalRange_C',
                'MeanHumidity_pct', 'MeanSolarRadiation'
            ]
            
            dl = MultiModalGenomicModel(
                n_snps=geno_df.shape[1],
                n_env_features=len(env_cols),
                random_seed=42
            )
            dl.build_model(learning_rate=0.001)
            dl.train(train_df, test_df, geno_df, env_cols, batch_size=64, epochs=50)
            y_pred, y_std = dl.evaluate(test_df, geno_df, env_cols)
            
            # Save predictions
            test_results = test_df[['Genotype_ID', 'Year', 'Location', 'Yield_bu_ac_norm']].copy()
            test_results['DL_Prediction'] = y_pred
            test_results['DL_Uncertainty'] = y_std
            test_results.to_csv("outputs/dl_predictions.csv", index=False)
            
            dl.save_model("models_saved/dl_model.keras")
            print("  ✓ DL training complete.")
    else:
        print_step(6, "Multi-Modal Deep Learning Model — SKIPPED")
    
    # ================================================================
    # STEP 7: Genotype Ranking & Recommendations
    # ================================================================
    print_step(7, "Genotype Ranking & Breeding Recommendations")
    
    ranking_exists = os.path.exists("outputs/final_genotype_ranking.csv")
    
    if ranking_exists:
        print("  ✓ Rankings already exist. Loading...")
        ranking = pd.read_csv("outputs/final_genotype_ranking.csv")
    else:
        print("  Computing rankings and recommendations...")
        from src.evaluation.prediction_system import BreedingPredictionSystem
        
        system = BreedingPredictionSystem()
        ranking, stability, recommendations = system.run_full_system()
        print("  ✓ Ranking complete.")
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - start_time
    
    print_header("PIPELINE COMPLETE!")
    
    print(f"\n  ⏱️  Total Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\n  📊 FINAL RESULTS:")
    print(f"  {'─'*50}")
    print(f"  {'Model':<25} {'RMSE':>10} {'Correlation':>12}")
    print(f"  {'─'*50}")
    print(f"  {'GBLUP (Baseline)':<25} {gblup_rmse:>10.4f} {gblup_corr:>12.4f}")
    
    # Check for ML results
    if os.path.exists("outputs/ml_predictions.csv"):
        ml_preds = pd.read_csv("outputs/ml_predictions.csv")
        from sklearn.metrics import mean_squared_error
        
        rf_rmse = np.sqrt(mean_squared_error(ml_preds['Yield_bu_ac_norm'], ml_preds['RF_Prediction']))
        rf_corr = np.corrcoef(ml_preds['Yield_bu_ac_norm'], ml_preds['RF_Prediction'])[0, 1]
        xgb_rmse = np.sqrt(mean_squared_error(ml_preds['Yield_bu_ac_norm'], ml_preds['XGB_Prediction']))
        xgb_corr = np.corrcoef(ml_preds['Yield_bu_ac_norm'], ml_preds['XGB_Prediction'])[0, 1]
        
        print(f"  {'Random Forest':<25} {rf_rmse:>10.4f} {rf_corr:>12.4f}")
        print(f"  {'XGBoost':<25} {xgb_rmse:>10.4f} {xgb_corr:>12.4f}")
    
    if os.path.exists("outputs/dl_predictions.csv"):
        dl_preds = pd.read_csv("outputs/dl_predictions.csv")
        dl_rmse = np.sqrt(mean_squared_error(dl_preds['Yield_bu_ac_norm'], dl_preds['DL_Prediction']))
        dl_corr = np.corrcoef(dl_preds['Yield_bu_ac_norm'], dl_preds['DL_Prediction'])[0, 1]
        print(f"  {'Deep Learning (Ours)':<25} {dl_rmse:>10.4f} {dl_corr:>12.4f}")
    
    print(f"  {'─'*50}")
    
    # Show top genotypes
    if os.path.exists("outputs/final_genotype_ranking.csv"):
        ranking = pd.read_csv("outputs/final_genotype_ranking.csv")
        print(f"\n  🏆 TOP 5 RECOMMENDED GENOTYPES:")
        for _, row in ranking.head(5).iterrows():
            print(f"    Rank {int(row['Final_Rank'])}: {row['Genotype_ID']} "
                  f"(Score: {row['Final_Score']:.3f})")
    
    print(f"\n  📁 OUTPUT FILES:")
    output_files = [
        "outputs/gblup_predictions.csv",
        "outputs/gblup_breeding_values.csv",
        "outputs/ml_predictions.csv",
        "outputs/dl_predictions.csv",
        "outputs/final_genotype_ranking.csv",
        "outputs/stability_scores.csv",
        "reports/figures/genotype_ranking.png",
        "models_saved/dl_model.keras",
    ]
    for f in output_files:
        if os.path.exists(f):
            print(f"    ✓ {f}")
    
    print_header("SYSTEM READY FOR PRODUCTION USE")
    print(f"  Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genomic Prediction System for Crop Breeding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    Run complete pipeline
  python main.py --skip-ml          Skip RF/XGBoost (faster)
  python main.py --skip-dl          Skip deep learning
  python main.py --data-only        Only generate and preprocess data
        """
    )
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip Random Forest and XGBoost models')
    parser.add_argument('--skip-dl', action='store_true',
                       help='Skip Deep Learning model')
    parser.add_argument('--data-only', action='store_true',
                       help='Only generate and preprocess data')
    
    args = parser.parse_args()
    
    run_pipeline(
        skip_ml=args.skip_ml,
        data_only=args.data_only,
        skip_dl=args.skip_dl
    )