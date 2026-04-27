# src/models/dl_model.py
"""
Multi-Modal Deep Learning Model for Genomic Prediction.
Architecture:
  Genotype Branch  (SNPs) ──→ Dense ──→ Dense ──┐
                                                   ├──→ Fusion ──→ Output (Yield)
  Environment Branch (Weather) ──→ Dense ──→ Dense┘

Includes:
- Monte Carlo Dropout for uncertainty estimation
- Early stopping to prevent overfitting
- Learning rate scheduling
"""
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class MultiModalGenomicModel:
    """
    Multi-modal deep learning model for genomic prediction with G×E.
    """
    
    def __init__(self, n_snps, n_env_features, random_seed=42):
        self.n_snps = n_snps
        self.n_env_features = n_env_features
        self.random_seed = random_seed
        self.model = None
        self.history = None
        self.geno_scaler = StandardScaler()
        self.env_scaler = StandardScaler()
        
        # Set seeds for reproducibility
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
    # ================================================================
    # BUILD MODEL ARCHITECTURE
    # ================================================================
    def build_model(self, learning_rate=0.001):
        """
        Build the multi-modal neural network architecture.
        
        Genotype Branch:  SNPs → Dense(256) → Dropout → Dense(128) → Dropout
        Environment Branch: Weather → Dense(64) → Dropout → Dense(32)
        Fusion: Concatenate → Dense(64) → Dense(32) → Output(1)
        """
        print("=" * 50)
        print("BUILDING MULTI-MODAL DL ARCHITECTURE")
        print("=" * 50)
        
        # --- GENOTYPE BRANCH ---
        geno_input = Input(shape=(self.n_snps,), name='Genotype_Input')
        
        geno = layers.Dense(512, activation='relu', kernel_regularizer=l2(1e-4),
                           name='Geno_Dense1')(geno_input)
        geno = layers.BatchNormalization(name='Geno_BN1')(geno)
        geno = layers.Dropout(0.3, name='Geno_Dropout1')(geno)
        
        geno = layers.Dense(256, activation='relu', kernel_regularizer=l2(1e-4),
                           name='Geno_Dense2')(geno)
        geno = layers.BatchNormalization(name='Geno_BN2')(geno)
        geno = layers.Dropout(0.3, name='Geno_Dropout2')(geno)
        
        geno = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4),
                           name='Geno_Dense3')(geno)
        geno = layers.BatchNormalization(name='Geno_BN3')(geno)
        geno = layers.Dropout(0.3, name='Geno_Dropout3')(geno)
        
        # --- ENVIRONMENT BRANCH ---
        env_input = Input(shape=(self.n_env_features,), name='Environment_Input')
        
        env = layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4),
                          name='Env_Dense1')(env_input)
        env = layers.BatchNormalization(name='Env_BN1')(env)
        env = layers.Dropout(0.2, name='Env_Dropout1')(env)
        
        env = layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4),
                          name='Env_Dense2')(env)
        env = layers.BatchNormalization(name='Env_BN2')(env)
        env = layers.Dropout(0.2, name='Env_Dropout2')(env)
        
        # --- FUSION LAYER ---
        fusion = layers.Concatenate(name='Fusion_Concat')([geno, env])
        
        fusion = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4),
                             name='Fusion_Dense1')(fusion)
        fusion = layers.BatchNormalization(name='Fusion_BN1')(fusion)
        fusion = layers.Dropout(0.2, name='Fusion_Dropout1')(fusion)
        
        fusion = layers.Dense(64, activation='relu', kernel_regularizer=l2(1e-4),
                             name='Fusion_Dense2')(fusion)
        fusion = layers.BatchNormalization(name='Fusion_BN2')(fusion)
        fusion = layers.Dropout(0.2, name='Fusion_Dropout2')(fusion)
        
        fusion = layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-4),
                             name='Fusion_Dense3')(fusion)
        
        # --- OUTPUT ---
        output = layers.Dense(1, name='Yield_Prediction')(fusion)
        
        # Build model
        self.model = Model(
            inputs=[geno_input, env_input],
            outputs=output,
            name='MultiModal_Genomic_Predictor'
        )
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Print summary
        print("\nModel Architecture:")
        print(f"  Genotype branch:  {self.n_snps} SNPs → 512 → 256 → 128")
        print(f"  Environment branch: {self.n_env_features} features → 64 → 32")
        print(f"  Fusion: 128 → 64 → 32 → 1 (Yield)")
        print(f"  Total parameters: {self.model.count_params():,}")
        print(f"  Regularization: L2 + BatchNorm + Dropout (MC Dropout)")
        
        return self.model
    
    # ================================================================
    # PREPARE DATA FOR DL MODEL
    # ================================================================
    def prepare_data(self, data_df, geno_df, env_cols, fit_scaler=False):
        """
        Prepare genotype and environment inputs for the DL model.
        
        Returns:
        --------
        X_geno : np.array (n_samples, n_snps)
        X_env : np.array (n_samples, n_env_features)
        y : np.array (n_samples,)
        """
        # Get genotype matrix for samples
        genotype_ids = data_df['Genotype_ID'].values
        X_geno = geno_df.loc[genotype_ids].values.astype(np.float32)
        
        # Get environmental features
        X_env = data_df[env_cols].values.astype(np.float32)
        
        # Scale features
        if fit_scaler:
            X_geno = self.geno_scaler.fit_transform(X_geno)
            X_env = self.env_scaler.fit_transform(X_env)
        else:
            X_geno = self.geno_scaler.transform(X_geno)
            X_env = self.env_scaler.transform(X_env)
        
        # Target
        y = data_df['Yield_bu_ac_norm'].values.astype(np.float32)
        
        return X_geno, X_env, y
    
    # ================================================================
    # TRAIN MODEL
    # ================================================================
    def train(self, train_df, test_df, geno_df, env_cols,
              batch_size=64, epochs=50):
        """
        Train the multi-modal DL model.
        """
        print("\n" + "=" * 50)
        print("TRAINING MULTI-MODAL DL MODEL")
        print("=" * 50)
        
        # Prepare data
        print("\nPreparing data...")
        X_geno_train, X_env_train, y_train = self.prepare_data(
            train_df, geno_df, env_cols, fit_scaler=True
        )
        X_geno_test, X_env_test, y_test = self.prepare_data(
            test_df, geno_df, env_cols, fit_scaler=False
        )
        
        print(f"  Training samples: {len(y_train)}")
        print(f"  Test samples: {len(y_test)}")
        print(f"  Genotype features: {X_geno_train.shape[1]}")
        print(f"  Environment features: {X_env_train.shape[1]}")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train
        print(f"\n  Training for up to {epochs} epochs...")
        print(f"  Batch size: {batch_size}")
        print("  (This may take a few minutes)\n")
        
        self.history = self.model.fit(
            [X_geno_train, X_env_train],
            y_train,
            validation_data=([X_geno_test, X_env_test], y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=2
        )
        
        # Final evaluation
        train_loss = self.history.history['loss'][-1]
        val_loss = self.history.history['val_loss'][-1]
        
        print(f"\n  Final Training Loss (MSE): {train_loss:.4f}")
        print(f"  Final Validation Loss (MSE): {val_loss:.4f}")
        
        return self.history
    
    # ================================================================
    # PREDICT WITH UNCERTAINTY (Monte Carlo Dropout)
    # ================================================================
    def predict_with_uncertainty(self, X_geno, X_env, n_iterations=50):
        """
        Predict with uncertainty using Monte Carlo Dropout.
        Dropout is kept ON during inference to generate multiple predictions.
        
        Parameters:
        -----------
        n_iterations : int
            Number of MC samples for uncertainty estimation
            
        Returns:
        --------
        mean : np.array — Mean prediction
        std : np.array — Prediction uncertainty (standard deviation)
        """
        # Enable dropout during inference
        # We use the functional API to run predictions with training=True
        
        predictions = []
        
        for _ in range(n_iterations):
            pred = self.model([X_geno, X_env], training=True)
            predictions.append(pred.numpy().flatten())
        
        predictions = np.array(predictions)
        
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred
    
    # ================================================================
    # EVALUATE
    # ================================================================
    def evaluate(self, test_df, geno_df, env_cols):
        """
        Evaluate the DL model with uncertainty.
        """
        print("\n" + "=" * 50)
        print("DL MODEL EVALUATION (with Uncertainty)")
        print("=" * 50)
        
        # Prepare test data
        X_geno_test, X_env_test, y_test = self.prepare_data(
            test_df, geno_df, env_cols, fit_scaler=False
        )
        
        # Standard prediction
        y_pred = self.model.predict(
            [X_geno_test, X_env_test], verbose=0
        ).flatten()
        
        # Prediction with uncertainty (MC Dropout)
        print("  Computing uncertainty (MC Dropout, 50 iterations)...")
        y_pred_mc, y_std_mc = self.predict_with_uncertainty(
            X_geno_test, X_env_test, n_iterations=50
        )
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        rmse_mc = np.sqrt(mean_squared_error(y_test, y_pred_mc))
        corr_mc = np.corrcoef(y_test, y_pred_mc)[0, 1]
        
        print(f"\n  Standard Prediction:")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    Correlation: {corr:.4f}")
        print(f"\n  MC Dropout Prediction:")
        print(f"    RMSE: {rmse_mc:.4f}")
        print(f"    Correlation: {corr_mc:.4f}")
        print(f"    Mean Uncertainty (std): {y_std_mc.mean():.4f}")
        
        self.metrics = {
            'RMSE': rmse_mc,
            'Correlation': corr_mc,
            'Mean_Uncertainty': float(y_std_mc.mean()),
        }
        
        return y_pred_mc, y_std_mc
    
    # ================================================================
    # SAVE MODEL
    # ================================================================
    def save_model(self, path="models_saved/dl_model.keras"):
        """Save the trained model."""
        os.makedirs("models_saved", exist_ok=True)
        self.model.save(path)
        print(f"\n✓ Model saved to {path}")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("█" * 60)
    print("█  MULTI-MODAL DEEP LEARNING FOR GENOMIC PREDICTION")
    print("█" * 60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv("data/processed/train_data.csv")
    test_df = pd.read_csv("data/processed/test_data.csv")
    geno_df = pd.read_csv("data/processed/genotypes_imputed.csv", index_col=0)
    
    # Environment feature columns
    env_cols = [
        'GDD', 'HeatStress_Days', 'MaxDrySpell_Days',
        'TotalPrecip_mm', 'MeanTemp_C', 'DiurnalRange_C',
        'MeanHumidity_pct', 'MeanSolarRadiation'
    ]
    
    print(f"  Genotype matrix: {geno_df.shape}")
    
    # Initialize model
    dl_model = MultiModalGenomicModel(
        n_snps=geno_df.shape[1],
        n_env_features=len(env_cols),
        random_seed=42
    )
    
    # Build model
    dl_model.build_model(learning_rate=0.001)
    
    # Train model
    dl_model.train(
        train_df, test_df, geno_df, env_cols,
        batch_size=64,
        epochs=50
    )
    
    # Evaluate
    y_pred, y_std = dl_model.evaluate(test_df, geno_df, env_cols)
    
    # Save model
    dl_model.save_model("models_saved/dl_model.keras")
    
    # Save predictions
    test_results = test_df[['Genotype_ID', 'Year', 'Location', 'Yield_bu_ac_norm']].copy()
    test_results['DL_Prediction'] = y_pred
    test_results['DL_Uncertainty'] = y_std
    test_results.to_csv("outputs/dl_predictions.csv", index=False)
    
    print(f"\n✓ DL predictions saved to outputs/dl_predictions.csv")
    
    # Final comparison
    print("\n" + "█" * 60)
    print("█  FINAL COMPARISON (ALL MODELS)")
    print("█" * 60)
    print(f"\n{'Model':<25} {'RMSE':>8} {'Correlation':>12}")
    print("-" * 48)
    print(f"{'GBLUP (Baseline)':<25} {0.8218:>8.4f} {0.5760:>12.4f}")
    print(f"{'Random Forest':<25} {0.9176:>8.4f} {0.4526:>12.4f}")
    print(f"{'XGBoost':<25} {0.9687:>8.4f} {0.3580:>12.4f}")
    print(f"{'Deep Learning (Ours)':<25} {dl_model.metrics['RMSE']:>8.4f} "
          f"{dl_model.metrics['Correlation']:>12.4f}")
    print("-" * 48)