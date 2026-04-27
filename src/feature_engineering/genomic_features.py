# src/feature_engineering/genomic_features.py
"""
Genomic feature engineering for genomic prediction.
1. PCA on SNP matrix (dimensionality reduction)
2. Genomic Relationship Matrix (GRM) - VanRaden method
3. BLUP estimation for breeding values
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


class GenomicFeatureEngineer:
    """
    Creates genomic features from SNP data.
    """
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.pca_model = None
        self.pca_components = None
        self.GRM = None
        self.genotype_ids = None
        
    # ================================================================
    # 1. PRINCIPAL COMPONENT ANALYSIS ON SNPs
    # ================================================================
    def compute_pca(self, geno_df, n_components=10):
        """
        Compute PCA on the SNP genotype matrix.
        This reduces dimensionality from ~5000 SNPs to 10-50 principal components.
        
        Parameters:
        -----------
        geno_df : pd.DataFrame
            Imputed genotype matrix (Genotype_ID × SNPs)
        n_components : int
            Number of principal components to retain
            
        Returns:
        --------
        pd.DataFrame : PCA scores for each genotype
        np.array : Explained variance ratio
        """
        print("=" * 50)
        print("1. PCA ON SNP MATRIX")
        print("=" * 50)
        
        print(f"  Input: {geno_df.shape[0]} genotypes × {geno_df.shape[1]} SNPs")
        
        # Standardize genotypes (mean=0, std=1 per SNP)
        scaler = StandardScaler()
        geno_scaled = scaler.fit_transform(geno_df.values)
        
        # Fit PCA
        self.pca_model = PCA(n_components=n_components, random_state=self.random_seed)
        pca_scores = self.pca_model.fit_transform(geno_scaled)
        
        # Create DataFrame with genotype IDs
        pca_df = pd.DataFrame(
            pca_scores,
            index=geno_df.index,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        
        # Variance explained
        var_explained = self.pca_model.explained_variance_ratio_
        cum_var = np.cumsum(var_explained)
        
        print(f"  Output: {n_components} principal components")
        print(f"  Variance explained by PC1: {var_explained[0]:.2%}")
        print(f"  Cumulative variance (first {min(5, n_components)} PCs):")
        for i in range(min(5, n_components)):
            print(f"    PC{i+1}: {var_explained[i]:.2%} (cumulative: {cum_var[i]:.2%})")
        print(f"  Total variance explained: {cum_var[-1]:.2%}")
        
        self.pca_components = pca_df
        return pca_df, var_explained
    
    # ================================================================
    # 2. GENOMIC RELATIONSHIP MATRIX (GRM) - VanRaden Method
    # ================================================================
    def compute_grm(self, geno_df, method='VanRaden'):
        """
        Compute Genomic Relationship Matrix using VanRaden (2008) method 1.
        G = Z Z' / (2 * sum(p_i * (1-p_i)))
        where Z = M - 2P (centered genotype matrix)
        
        Parameters:
        -----------
        geno_df : pd.DataFrame
            Imputed genotype matrix (Genotype_ID × SNPs)
        method : str
            'VanRaden' (standard method)
            
        Returns:
        --------
        pd.DataFrame : Genomic Relationship Matrix (n × n)
        """
        print("\n" + "=" * 50)
        print("2. GENOMIC RELATIONSHIP MATRIX (GRM)")
        print("=" * 50)
        
        n = geno_df.shape[0]  # number of genotypes
        m = geno_df.shape[1]  # number of SNPs
        
        print(f"  Computing {n}×{n} GRM from {m} SNPs...")
        
        # Get genotype matrix
        M = geno_df.values.astype(float)
        
        # Calculate allele frequencies
        p = M.mean(axis=0) / 2.0  # Allele frequency of the second allele
        
        # Center the genotype matrix: Z = M - 2P
        # P is a matrix where each column is p
        Z = M - 2 * p
        
        # VanRaden method 1
        # G = Z Z' / (2 * sum(p * (1-p)))
        denominator = 2 * np.sum(p * (1 - p))
        G = (Z @ Z.T) / denominator
        
        # Create DataFrame
        grm_df = pd.DataFrame(G, index=geno_df.index, columns=geno_df.index)
        
        # Check matrix properties
        diag_mean = np.mean(np.diag(G))
        off_diag_mean = (G.sum() - np.trace(G)) / (n * (n - 1))
        
        print(f"  ✓ GRM computed: {n}×{n}")
        print(f"  Mean diagonal (self-relatedness): {diag_mean:.3f}")
        print(f"  Mean off-diagonal (pairwise relatedness): {off_diag_mean:.3f}")
        print(f"  Matrix is symmetric: {np.allclose(G, G.T)}")
        print(f"  Matrix is positive semi-definite: {np.all(np.linalg.eigvalsh(G) >= -1e-10)}")
        
        self.GRM = grm_df
        self.genotype_ids = geno_df.index.tolist()
        return grm_df
    
    # ================================================================
    # 3. BLUP ESTIMATION (SIMPLIFIED)
    # ================================================================
    def estimate_blups(self, train_df, trait_col='Yield_bu_ac_norm'):
        """
        Estimate Best Linear Unbiased Predictions for each genotype.
        Simplified BLUP: genotype mean + shrinkage based on heritability.
        
        This gives us an initial estimate of breeding values.
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training data with phenotype values
        trait_col : str
            Trait column to estimate BLUPs for
            
        Returns:
        --------
        pd.DataFrame : BLUP estimates for each genotype
        """
        print("\n" + "=" * 50)
        print("3. BLUP ESTIMATION")
        print("=" * 50)
        
        # Calculate genotype means
        genotype_means = train_df.groupby('Genotype_ID')[trait_col].agg(['mean', 'count', 'std'])
        genotype_means.columns = ['Mean', 'N_obs', 'Std']
        
        # Overall mean
        overall_mean = train_df[trait_col].mean()
        
        # Estimate heritability-like shrinkage
        # Using a simplified approach: h2 ~ var_g / (var_g + var_e/n)
        var_between = genotype_means['Mean'].var()
        var_within = (genotype_means['Std'] ** 2).mean()
        n_avg = genotype_means['N_obs'].mean()
        
        # Shrinkage factor
        shrinkage = var_between / (var_between + var_within / n_avg)
        shrinkage = max(0.1, min(0.9, shrinkage))  # Bound between 0.1 and 0.9
        
        # BLUP = overall_mean + shrinkage * (genotype_mean - overall_mean)
        genotype_means['BLUP'] = overall_mean + shrinkage * (genotype_means['Mean'] - overall_mean)
        
        # Sort by BLUP
        genotype_means = genotype_means.sort_values('BLUP', ascending=False)
        
        print(f"  Trait: {trait_col}")
        print(f"  Number of genotypes: {len(genotype_means)}")
        print(f"  Overall mean: {overall_mean:.3f}")
        print(f"  Shrinkage factor (h2 estimate): {shrinkage:.3f}")
        print(f"  Top 5 genotypes by BLUP:")
        for i, (idx, row) in enumerate(genotype_means.head(5).iterrows()):
            print(f"    {idx}: BLUP={row['BLUP']:.3f} (mean={row['Mean']:.3f}, n={int(row['N_obs'])})")
        
        return genotype_means
    
    # ================================================================
    # 4. FULL FEATURE ENGINEERING PIPELINE
    # ================================================================
    def run_full_feature_engineering(self, n_pca_components=10):
        """
        Run complete feature engineering pipeline.
        """
        print("\n" + "█" * 60)
        print("█  GENOMIC FEATURE ENGINEERING PIPELINE")
        print("█" * 60)
        
        # Load imputed genotypes
        print("\nLoading imputed genotypes...")
        geno_imputed = pd.read_csv("data/processed/genotypes_imputed.csv", index_col=0)
        print(f"  Genotypes: {geno_imputed.shape}")
        
        # Load training data
        train_df = pd.read_csv("data/processed/train_data.csv")
        print(f"  Training records: {len(train_df)}")
        
        # 1. Compute PCA
        pca_df, var_explained = self.compute_pca(
            geno_imputed,
            n_components=n_pca_components
        )
        
        # 2. Compute GRM
        grm_df = self.compute_grm(geno_imputed)
        
        # 3. Estimate BLUPs
        blups = self.estimate_blups(
            train_df,
            trait_col='Yield_bu_ac_norm'
        )
        
        # Save all features
        os.makedirs("data/processed", exist_ok=True)
        pca_df.to_csv("data/processed/pca_features.csv")
        grm_df.to_csv("data/processed/genomic_relationship_matrix.csv")
        blups.to_csv("data/processed/blup_estimates.csv")
        
        print("\n" + "█" * 60)
        print("█  FEATURE ENGINEERING COMPLETE!")
        print("█" * 60)
        print(f"\nSaved to data/processed/:")
        print(f"  - pca_features.csv ({pca_df.shape})")
        print(f"  - genomic_relationship_matrix.csv ({grm_df.shape})")
        print(f"  - blup_estimates.csv ({len(blups)} genotypes)")
        
        return pca_df, grm_df, blups


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    engineer = GenomicFeatureEngineer(random_seed=42)
    pca, grm, blups = engineer.run_full_feature_engineering(n_pca_components=10)