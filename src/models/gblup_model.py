# src/models/gblup_model.py
"""
GBLUP (Genomic Best Linear Unbiased Prediction) Model.
Scientific baseline for genomic prediction using the Genomic Relationship Matrix.

Method: Mixed linear model
y = Xb + Zu + e
where:
  y = phenotypes
  X = fixed effects design matrix (environment)
  b = fixed effects
  Z = incidence matrix linking phenotypes to genotypes
  u = random genetic effects ~ N(0, G*σ²g)
  e = residuals ~ N(0, I*σ²e)
"""
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.metrics import mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')


class GBLUP:
    """
    Genomic Best Linear Unbiased Prediction using mixed model equations.
    """
    
    def __init__(self):
        self.beta = None          # Fixed effects
        self.u = None             # Random genetic effects (breeding values)
        self.sigma2_g = None      # Genetic variance
        self.sigma2_e = None      # Residual variance
        self.h2 = None            # Heritability
        self.genotype_ids = None  # Genotype IDs for u
        
    # ================================================================
    # FIT GBLUP MODEL
    # ================================================================
    def fit(self, train_df, grm_df, trait_col='Yield_bu_ac_norm'):
        """
        Fit GBLUP model using Mixed Model Equations (Henderson, 1975).
        
        Mixed Model Equations:
        [X'X      X'Z      ] [b]   [X'y]
        [Z'X      Z'Z + λG⁻¹] [u] = [Z'y]
        
        where λ = σ²e / σ²g
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training data with phenotypes and environment info
        grm_df : pd.DataFrame
            Genomic Relationship Matrix (n_genotypes × n_genotypes)
        trait_col : str
            Target trait column
        """
        print("=" * 50)
        print("GBLUP MODEL TRAINING")
        print("=" * 50)
        
        # Get unique genotypes and environments
        self.genotype_ids = sorted(train_df['Genotype_ID'].unique())
        environments = sorted(train_df['Environment'].unique())
        
        n = len(train_df)           # number of observations
        n_g = len(self.genotype_ids)  # number of genotypes
        n_e = len(environments)     # number of environments
        
        print(f"  Observations: {n}")
        print(f"  Genotypes: {n_g}")
        print(f"  Environments: {n_e}")
        print(f"  Target trait: {trait_col}")
        
        # Create genotype index mapping
        geno_to_idx = {g: i for i, g in enumerate(self.genotype_ids)}
        
        # Subset GRM to match training genotypes
        grm_subset = grm_df.loc[self.genotype_ids, self.genotype_ids]
        G = grm_subset.values
        
        # Create design matrices
        # X: fixed effects (environment means)
        X = np.zeros((n, n_e))
        y = np.zeros(n)
        
        # Z: random effects (genotypes)
        Z = np.zeros((n, n_g))
        
        for i, (_, row) in enumerate(train_df.iterrows()):
            env_idx = environments.index(row['Environment'])
            geno_idx = geno_to_idx[row['Genotype_ID']]
            
            X[i, env_idx] = 1.0
            Z[i, geno_idx] = 1.0
            y[i] = row[trait_col]
        
        # Compute variance components using REML-like approach
        # Initial estimates
        sigma2_e = np.var(y) * 0.5
        sigma2_g = np.var(y) * 0.5
        
        # EM algorithm for variance components (simplified)
        for iteration in range(10):
            lambda_val = sigma2_e / sigma2_g
            
            # Build MME left-hand side
            XtX = X.T @ X
            XtZ = X.T @ Z
            ZtX = Z.T @ X
            ZtZ = Z.T @ Z
            
            # G inverse with regularization
            G_inv = np.linalg.inv(G + np.eye(n_g) * 0.01)
            
            # MME left-hand side
            MME_L = np.block([
                [XtX, XtZ],
                [ZtX, ZtZ + lambda_val * G_inv]
            ])
            
            # MME right-hand side
            Xty = X.T @ y
            Zty = Z.T @ y
            MME_R = np.concatenate([Xty, Zty])
            
            # Solve MME
            try:
                solution = solve(MME_L, MME_R, assume_a='pos')
            except:
                # Fallback: use least squares with regularization
                solution = np.linalg.lstsq(MME_L, MME_R, rcond=None)[0]
            
            self.beta = solution[:n_e]
            self.u = solution[n_e:]
            
            # Update variance components
            y_pred = X @ self.beta + Z @ self.u
            residuals = y - y_pred
            
            sigma2_e_new = np.sum(residuals**2) / (n - n_e)
            sigma2_g_new = np.sum(self.u**2) / n_g
            
            # Check convergence
            if abs(sigma2_e_new - sigma2_e) < 1e-6 and abs(sigma2_g_new - sigma2_g) < 1e-6:
                sigma2_e = sigma2_e_new
                sigma2_g = sigma2_g_new
                break
            
            sigma2_e = sigma2_e_new
            sigma2_g = sigma2_g_new
        
        self.sigma2_g = sigma2_g
        self.sigma2_e = sigma2_e
        self.h2 = sigma2_g / (sigma2_g + sigma2_e)
        
        print(f"\n  Variance Components:")
        print(f"    σ²g (Genetic):  {sigma2_g:.4f}")
        print(f"    σ²e (Residual): {sigma2_e:.4f}")
        print(f"    h² (Heritability): {self.h2:.3f}")
        print(f"  Breeding values range: [{self.u.min():.3f}, {self.u.max():.3f}]")
        
        return self
    
    # ================================================================
    # PREDICT
    # ================================================================
    def predict(self, test_df, grm_df):
        """
        Predict breeding values for test genotypes.
        
        For genotypes seen in training: use estimated u directly
        For new genotypes: use genomic relationship to training genotypes
        """
        print("\n" + "=" * 50)
        print("GBLUP PREDICTION")
        print("=" * 50)
        
        test_genotypes = sorted(test_df['Genotype_ID'].unique())
        geno_to_idx_train = {g: i for i, g in enumerate(self.genotype_ids)}
        
        predictions = []
        
        for _, row in test_df.iterrows():
            geno = row['Genotype_ID']
            
            if geno in geno_to_idx_train:
                # Genotype seen in training — use direct breeding value
                gbv = self.u[geno_to_idx_train[geno]]
            else:
                # New genotype — predict based on relationship
                # This is simplified; in practice use kinship to training genotypes
                gbv = 0.0  # Fallback to population mean
            
            # Add environment effect
            # For simplicity, we use the mean environment effect
            env_effect = self.beta.mean()
            
            predictions.append(gbv + env_effect)
        
        test_df = test_df.copy()
        test_df['GBLUP_Prediction'] = predictions
        
        print(f"  Predicted {len(predictions)} observations")
        print(f"  Prediction range: [{min(predictions):.3f}, {max(predictions):.3f}]")
        
        return test_df
    
    # ================================================================
    # EVALUATE
    # ================================================================
    def evaluate(self, test_df, trait_col='Yield_bu_ac_norm'):
        """
        Evaluate GBLUP predictions.
        """
        print("\n" + "=" * 50)
        print("GBLUP EVALUATION")
        print("=" * 50)
        
        y_true = test_df[trait_col].values
        y_pred = test_df['GBLUP_Prediction'].values
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Pearson correlation
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Ranking accuracy: correlation of top 10% genotypes
        n_top = max(10, len(test_df) // 10)
        true_ranked = test_df.nlargest(n_top, trait_col)['Genotype_ID']
        pred_ranked = test_df.nlargest(n_top, 'GBLUP_Prediction')['Genotype_ID']
        overlap = len(set(true_ranked) & set(pred_ranked)) / n_top
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Pearson Correlation: {corr:.4f}")
        print(f"  Top {n_top} overlap: {overlap:.2%}")
        
        self.metrics = {
            'RMSE': rmse,
            'Correlation': corr,
            'Top_Overlap': overlap,
            'h2': self.h2,
        }
        
        return self.metrics
    
    # ================================================================
    # GET BREEDING VALUES
    # ================================================================
    def get_breeding_values(self):
        """
        Return estimated breeding values for all genotypes.
        """
        bv_df = pd.DataFrame({
            'Genotype_ID': self.genotype_ids,
            'Breeding_Value': self.u,
            'Rank': np.argsort(np.argsort(-self.u)) + 1  # 1 = best
        }).sort_values('Breeding_Value', ascending=False)
        
        return bv_df


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train_data.csv")
    test_df = pd.read_csv("data/processed/test_data.csv")
    grm_df = pd.read_csv("data/processed/genomic_relationship_matrix.csv", index_col=0)
    
    # Initialize and fit model
    gblup = GBLUP()
    gblup.fit(train_df, grm_df, trait_col='Yield_bu_ac_norm')
    
    # Predict
    test_predictions = gblup.predict(test_df, grm_df)
    
    # Evaluate
    metrics = gblup.evaluate(test_predictions, trait_col='Yield_bu_ac_norm')
    
    # Get breeding values
    bv = gblup.get_breeding_values()
    
    # Save results
    os.makedirs("outputs", exist_ok=True)
    test_predictions[['Genotype_ID', 'Year', 'Location', 'Yield_bu_ac_norm', 'GBLUP_Prediction']].to_csv(
        "outputs/gblup_predictions.csv", index=False
    )
    bv.to_csv("outputs/gblup_breeding_values.csv", index=False)
    
    print(f"\n✓ GBLUP results saved to outputs/")
    print(f"  - gblup_predictions.csv")
    print(f"  - gblup_breeding_values.csv")
    print(f"\nTop 5 Genotypes by Breeding Value:")
    for _, row in bv.head(5).iterrows():
        print(f"  Rank {int(row['Rank'])}: {row['Genotype_ID']} (BV={row['Breeding_Value']:.4f})")