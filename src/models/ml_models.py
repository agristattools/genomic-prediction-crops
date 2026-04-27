# src/models/ml_models.py
"""
Machine Learning models for genomic prediction.
1. Random Forest Regressor
2. XGBoost Regressor
Both use: Genomic features (SNPs/PCA) + Environmental features
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')


class GenomicMLModels:
    """
    ML models for genomic prediction with G×E interactions.
    """
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.rf_model = None
        self.xgb_model = None
        self.feature_names = None
        self.results = {}
        
    # ================================================================
    # PREPARE FEATURES
    # ================================================================
    def prepare_features(self, data_df, geno_df, pca_df, env_df):
        """
        Merge genomic and environmental features into a single feature matrix.
        
        Features used:
        - PCA components (10) — dimensionality-reduced SNP information
        - Environmental variables (8) — GDD, precipitation, humidity, etc.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            Phenotype data (train or test)
        geno_df : pd.DataFrame
            Genotype matrix or PCA features
        pca_df : pd.DataFrame
            PCA scores per genotype
        env_df : pd.DataFrame
            Environmental features per environment
            
        Returns:
        --------
        np.array : Feature matrix X
        np.array : Target vector y
        list : Feature names
        """
        # Merge PCA features with phenotype data
        data_with_pca = data_df.merge(
            pca_df, left_on='Genotype_ID', right_index=True, how='left'
        )
        
        # Feature columns
        pca_cols = [f"PC{i+1}" for i in range(10)]
        env_cols = [
            'GDD', 'HeatStress_Days', 'MaxDrySpell_Days',
            'TotalPrecip_mm', 'MeanTemp_C', 'DiurnalRange_C',
            'MeanHumidity_pct', 'MeanSolarRadiation'
        ]
        
        feature_cols = pca_cols + env_cols
        self.feature_names = feature_cols
        
        # Create feature matrix
        X = data_with_pca[feature_cols].values
        
        # Target
        y = data_with_pca['Yield_bu_ac_norm'].values
        
        return X, y, feature_cols
    
    # ================================================================
    # 1. RANDOM FOREST
    # ================================================================
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Train Random Forest with hyperparameter tuning.
        """
        print("\n" + "=" * 50)
        print("RANDOM FOREST MODEL")
        print("=" * 50)
        
        # Hyperparameter grid
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
        }
        
        # Base model
        rf = RandomForestRegressor(
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=0
        )
        
        # Randomized search
        print("  Running hyperparameter tuning...")
        rf_search = RandomizedSearchCV(
            rf, param_dist,
            n_iter=20,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=0
        )
        
        rf_search.fit(X_train, y_train)
        
        self.rf_model = rf_search.best_estimator_
        
        print(f"  Best parameters: {rf_search.best_params_}")
        
        # Predict
        y_pred_train = self.rf_model.predict(X_train)
        y_pred_test = self.rf_model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
        test_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
        
        print(f"\n  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE:     {test_rmse:.4f}")
        print(f"  Training Corr: {train_corr:.4f}")
        print(f"  Test Corr:     {test_corr:.4f}")
        
        # Feature importance
        importances = self.rf_model.feature_importances_
        top_features = np.argsort(importances)[-5:][::-1]
        print(f"\n  Top 5 features:")
        for idx in top_features:
            print(f"    {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        self.results['RandomForest'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_corr': train_corr,
            'test_corr': test_corr,
            'best_params': rf_search.best_params_,
        }
        
        return y_pred_test
    
    # ================================================================
    # 2. XGBOOST
    # ================================================================
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        Train XGBoost with hyperparameter tuning.
        """
        print("\n" + "=" * 50)
        print("XGBOOST MODEL")
        print("=" * 50)
        
        # Hyperparameter grid
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
        }
        
        # Base model
        xgb = XGBRegressor(
            random_state=self.random_seed,
            n_jobs=-1,
            verbosity=0
        )
        
        # Randomized search
        print("  Running hyperparameter tuning...")
        xgb_search = RandomizedSearchCV(
            xgb, param_dist,
            n_iter=20,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=0
        )
        
        xgb_search.fit(X_train, y_train)
        
        self.xgb_model = xgb_search.best_estimator_
        
        print(f"  Best parameters: {xgb_search.best_params_}")
        
        # Predict
        y_pred_train = self.xgb_model.predict(X_train)
        y_pred_test = self.xgb_model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
        test_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
        
        print(f"\n  Training RMSE: {train_rmse:.4f}")
        print(f"  Test RMSE:     {test_rmse:.4f}")
        print(f"  Training Corr: {train_corr:.4f}")
        print(f"  Test Corr:     {test_corr:.4f}")
        
        # Feature importance
        importances = self.xgb_model.feature_importances_
        top_features = np.argsort(importances)[-5:][::-1]
        print(f"\n  Top 5 features:")
        for idx in top_features:
            print(f"    {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        self.results['XGBoost'] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_corr': train_corr,
            'test_corr': test_corr,
            'best_params': xgb_search.best_params_,
        }
        
        return y_pred_test
    
    # ================================================================
    # COMPARE MODELS
    # ================================================================
    def compare_models(self):
        """
        Compare all trained models against GBLUP baseline.
        """
        print("\n" + "█" * 60)
        print("█  MODEL COMPARISON")
        print("█" * 60)
        
        print(f"\n{'Model':<20} {'RMSE':>8} {'Correlation':>12} {'vs GBLUP':>12}")
        print("-" * 55)
        
        gblup_rmse = 0.8218  # From previous step
        gblup_corr = 0.5760  # From previous step
        
        print(f"{'GBLUP (Baseline)':<20} {gblup_rmse:>8.4f} {gblup_corr:>12.4f} {'—':>12}")
        
        for model_name, res in self.results.items():
            rmse_improve = (gblup_rmse - res['test_rmse']) / gblup_rmse * 100
            corr_improve = (res['test_corr'] - gblup_corr) / gblup_corr * 100
            
            print(f"{model_name:<20} {res['test_rmse']:>8.4f} {res['test_corr']:>12.4f} "
                  f"{corr_improve:>+8.1f}% corr")
        
        print("-" * 55)
        print("  Positive % = improvement over GBLUP")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("█" * 60)
    print("█  MACHINE LEARNING MODELS FOR GENOMIC PREDICTION")
    print("█" * 60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv("data/processed/train_data.csv")
    test_df = pd.read_csv("data/processed/test_data.csv")
    geno_df = pd.read_csv("data/processed/genotypes_imputed.csv", index_col=0)
    pca_df = pd.read_csv("data/processed/pca_features.csv", index_col=0)
    env_df = pd.read_csv("data/processed/environment_features.csv")
    
    print(f"  Train: {len(train_df)} records")
    print(f"  Test:  {len(test_df)} records")
    print(f"  PCA features: {pca_df.shape}")
    
    # Initialize models
    ml_models = GenomicMLModels(random_seed=42)
    
    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, feature_names = ml_models.prepare_features(
        train_df, geno_df, pca_df, env_df
    )
    X_test, y_test, _ = ml_models.prepare_features(
        test_df, geno_df, pca_df, env_df
    )
    
    print(f"  Training features: {X_train.shape}")
    print(f"  Test features: {X_test.shape}")
    print(f"  Features: {feature_names}")
    
    # Train Random Forest
    rf_preds = ml_models.train_random_forest(X_train, y_train, X_test, y_test)
    
    # Train XGBoost
    xgb_preds = ml_models.train_xgboost(X_train, y_train, X_test, y_test)
    
    # Compare models
    ml_models.compare_models()
    
    # Save predictions
    os.makedirs("outputs", exist_ok=True)
    
    test_results = test_df[['Genotype_ID', 'Year', 'Location', 'Yield_bu_ac_norm']].copy()
    test_results['RF_Prediction'] = rf_preds
    test_results['XGB_Prediction'] = xgb_preds
    test_results.to_csv("outputs/ml_predictions.csv", index=False)
    
    print(f"\n✓ ML predictions saved to outputs/ml_predictions.csv")