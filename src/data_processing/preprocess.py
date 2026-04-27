# src/data_processing/preprocess.py
"""
Data preprocessing pipeline for genomic prediction.
Steps:
1. SNP quality control (MAF filtering, missing rate)
2. Genotype imputation (mode imputation)
3. Phenotype normalization (within environment)
4. Environmental feature engineering (GDD, heat stress, drought index)
5. Train/test splitting by year (no data leakage!)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import yaml
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Complete preprocessing pipeline for genomic prediction data.
    """
    
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.geno_stats = {}
        self.pheno_stats = {}
        self.env_scaler = None
        self.trait_scalers = {}
        
    # ================================================================
    # 1. SNP QUALITY CONTROL
    # ================================================================
    def filter_snps(self, geno_df, maf_threshold=0.05, call_rate_threshold=0.9):
        """
        Filter SNPs based on Minor Allele Frequency and call rate.
        """
        print("=" * 50)
        print("1. SNP QUALITY CONTROL")
        print("=" * 50)
        
        n_snps_start = geno_df.shape[1]
        
        # Calculate call rate (proportion of non-missing per SNP)
        call_rate = (geno_df != -1).mean(axis=0)
        
        # Filter by call rate
        snps_keep_cr = call_rate[call_rate >= call_rate_threshold].index
        geno_filtered = geno_df[snps_keep_cr]
        n_removed_cr = n_snps_start - len(snps_keep_cr)
        
        # Calculate MAF for each SNP (ignoring missing = -1)
        maf_values = []
        for snp in geno_filtered.columns:
            values = geno_filtered[snp].values
            valid = values[values != -1]
            if len(valid) == 0:
                maf_values.append(0)
            else:
                alt_freq = valid.sum() / (2 * len(valid))
                maf = min(alt_freq, 1 - alt_freq)
                maf_values.append(maf)
        
        maf_series = pd.Series(maf_values, index=geno_filtered.columns)
        
        # Filter by MAF
        snps_keep_maf = maf_series[maf_series >= maf_threshold].index
        geno_final = geno_filtered[snps_keep_maf]
        n_removed_maf = len(geno_filtered.columns) - len(snps_keep_maf)
        
        # Store stats
        self.geno_stats = {
            'n_snps_start': n_snps_start,
            'n_snps_after_call_rate': len(geno_filtered.columns),
            'n_snps_after_maf': len(geno_final),
            'n_removed_call_rate': n_removed_cr,
            'n_removed_maf': n_removed_maf,
            'n_removed_total': n_snps_start - len(geno_final.columns),
            'retention_pct': 100 * len(geno_final.columns) / n_snps_start,
        }
        
        print(f"  Starting SNPs: {n_snps_start}")
        print(f"  Removed (low call rate): {n_removed_cr}")
        print(f"  Removed (low MAF): {n_removed_maf}")
        print(f"  Retained SNPs: {len(geno_final.columns)} ({self.geno_stats['retention_pct']:.1f}%)")
        
        return geno_final
    
    # ================================================================
    # 2. GENOTYPE IMPUTATION (FIXED)
    # ================================================================
    def impute_genotypes(self, geno_df):
        """
        Impute missing genotypes using mode imputation per SNP.
        FIX: Uses iloc to avoid read-only array issue.
        """
        print("\n" + "=" * 50)
        print("2. GENOTYPE IMPUTATION")
        print("=" * 50)
        
        geno_imputed = geno_df.copy()
        total_missing = (geno_imputed == -1).sum().sum()
        
        # Impute each SNP with its mode
        for col_idx, snp in enumerate(geno_imputed.columns):
            # Get values as a fresh numpy array (writable)
            col_values = geno_imputed.iloc[:, col_idx].values.copy()
            valid = col_values[col_values != -1]
            if len(valid) > 0:
                # Find mode value
                mode_val = pd.Series(valid).mode()
                if len(mode_val) > 0:
                    mode_val = int(mode_val.iloc[0])
                    # Replace -1 with mode
                    col_values[col_values == -1] = mode_val
                    # Put back into dataframe
                    geno_imputed.iloc[:, col_idx] = col_values.astype(np.int8)
        
        remaining_missing = (geno_imputed == -1).sum().sum()
        
        print(f"  Missing values imputed: {total_missing}")
        print(f"  Remaining missing: {remaining_missing}")
        print(f"  Imputation method: SNP-wise mode")
        
        return geno_imputed
    
    # ================================================================
    # 3. PHENOTYPE NORMALIZATION
    # ================================================================
    def normalize_phenotypes(self, pheno_df, trait_columns, method='within_environment'):
        """
        Normalize phenotypes. Default: within each environment (Year×Location).
        """
        print("\n" + "=" * 50)
        print("3. PHENOTYPE NORMALIZATION")
        print("=" * 50)
        
        pheno_norm = pheno_df.copy()
        
        for trait in trait_columns:
            norm_col_name = f"{trait}_norm"
            
            if method == 'within_environment':
                # Normalize within each environment
                for env in pheno_norm['Environment'].unique():
                    mask = pheno_norm['Environment'] == env
                    values = pheno_norm.loc[mask, trait].copy()
                    
                    mean = values.mean()
                    std = values.std()
                    
                    if std > 0:
                        pheno_norm.loc[mask, norm_col_name] = (values - mean) / std
                    else:
                        pheno_norm.loc[mask, norm_col_name] = 0
                        
                    # Store stats
                    self.pheno_stats[f"{env}_{trait}"] = {'mean': mean, 'std': std}
            else:
                # Global normalization
                scaler = StandardScaler()
                pheno_norm[norm_col_name] = scaler.fit_transform(
                    pheno_norm[[trait]]
                )
                self.trait_scalers[trait] = scaler
            
            print(f"  ✓ {trait} → {norm_col_name} ({method})")
        
        return pheno_norm
    
    # ================================================================
    # 4. ENVIRONMENTAL FEATURE ENGINEERING
    # ================================================================
    def engineer_environmental_features(self, weather_df):
        """
        Engineer environmental indices from daily weather data.
        
        Features:
        - GDD: Growing Degree Days (base 10°C)
        - Heat stress days: Days with Tmax > 35°C
        - Drought index: Consecutive days without rain
        - Cumulative precipitation
        - Mean temperature
        - Diurnal temperature range
        """
        print("\n" + "=" * 50)
        print("4. ENVIRONMENTAL FEATURE ENGINEERING")
        print("=" * 50)
        
        env_features = []
        
        for (loc, year), group in weather_df.groupby(['Location', 'Year']):
            features = {'Location': loc, 'Year': year, 'Environment': f"{loc}_{year}"}
            
            # GDD (base 10°C)
            tmean = (group['Temp_Max_C'] + group['Temp_Min_C']) / 2
            gdd = np.maximum(tmean - 10, 0).sum()
            features['GDD'] = round(float(gdd), 1)
            
            # Heat stress days (Tmax > 35°C)
            features['HeatStress_Days'] = int((group['Temp_Max_C'] > 35).sum())
            
            # Drought index: max consecutive days with precip < 1mm
            dry_days_list = (group['Precipitation_mm'] < 1).astype(int).tolist()
            consecutive = 0
            max_consecutive = 0
            for d in dry_days_list:
                if d == 1:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            features['MaxDrySpell_Days'] = max_consecutive
            
            # Cumulative precipitation
            features['TotalPrecip_mm'] = round(float(group['Precipitation_mm'].sum()), 1)
            
            # Mean temperature
            features['MeanTemp_C'] = round(float(tmean.mean()), 1)
            
            # Mean diurnal temperature range
            features['DiurnalRange_C'] = round(
                float((group['Temp_Max_C'] - group['Temp_Min_C']).mean()), 1
            )
            
            # Mean humidity
            features['MeanHumidity_pct'] = round(float(group['Humidity_pct'].mean()), 1)
            
            # Mean solar radiation
            features['MeanSolarRadiation'] = round(float(group['SolarRadiation_MJ_m2'].mean()), 1)
            
            env_features.append(features)
        
        env_df = pd.DataFrame(env_features)
        print(f"  ✓ Generated {len(env_df)} environment profiles")
        print(f"  Features: GDD, HeatStress, MaxDrySpell, TotalPrecip, "
              f"MeanTemp, DiurnalRange, MeanHumidity, MeanSolarRadiation")
        
        return env_df
    
    # ================================================================
    # 5. TRAIN/TEST SPLIT (By Year — No Data Leakage!)
    # ================================================================
    def split_by_year(self, pheno_df, train_years, test_years):
        """
        Split data by year to prevent data leakage.
        Train on past years, test on future years.
        """
        print("\n" + "=" * 50)
        print("5. TRAIN/TEST SPLIT")
        print("=" * 50)
        
        train_mask = pheno_df['Year'].isin(train_years)
        test_mask = pheno_df['Year'].isin(test_years)
        
        train_df = pheno_df[train_mask].copy()
        test_df = pheno_df[test_mask].copy()
        
        print(f"  Train: {len(train_df)} records ({min(train_years)}-{max(train_years)})")
        print(f"  Test:  {len(test_df)} records ({min(test_years)}-{max(test_years)})")
        print(f"  Split ratio: {len(train_df) / len(pheno_df):.1%} / {len(test_df) / len(pheno_df):.1%}")
        
        # Verify no data leakage
        train_ids = set(train_df['Genotype_ID'])
        test_ids = set(test_df['Genotype_ID'])
        leakage = train_ids & test_ids
        print(f"  Genotypes in both sets: {len(leakage)} (expected > 0)")
        print(f"  ✓ No year overlap = No temporal leakage")
        
        return train_df, test_df
    
    # ================================================================
    # 6. FULL PIPELINE
    # ================================================================
    def run_full_pipeline(self):
        """
        Execute the complete preprocessing pipeline.
        """
        print("\n" + "█" * 60)
        print("█  FULL PREPROCESSING PIPELINE")
        print("█" * 60)
        
        # Load raw data
        print("\nLoading raw data...")
        geno_raw = pd.read_csv("data/raw/genotypes.csv", index_col=0)
        pheno_raw = pd.read_csv("data/raw/phenotypes.csv")
        weather_raw = pd.read_csv("data/raw/weather.csv")
        print(f"  Genotypes: {geno_raw.shape}")
        print(f"  Phenotypes: {pheno_raw.shape}")
        print(f"  Weather: {weather_raw.shape}")
        
        # Step 1: Filter SNPs
        geno_filtered = self.filter_snps(
            geno_raw,
            maf_threshold=self.config['genomic']['maf_threshold'],
            call_rate_threshold=self.config['genomic']['min_call_rate']
        )
        
        # Step 2: Impute genotypes
        geno_imputed = self.impute_genotypes(geno_filtered)
        
        # Step 3: Normalize phenotypes
        trait_cols = ['Yield_bu_ac', 'PlantHeight_cm', 'DaysToAnthesis', 'GrainMoisture_pct']
        pheno_norm = self.normalize_phenotypes(pheno_raw, trait_cols)
        
        # Step 4: Engineer environmental features
        env_features = self.engineer_environmental_features(weather_raw)
        
        # Step 5: Split by year
        train_df, test_df = self.split_by_year(
            pheno_norm,
            train_years=list(range(
                self.config['training']['train_years_start'],
                self.config['training']['train_years_end'] + 1
            )),
            test_years=list(range(
                self.config['training']['test_years_start'],
                self.config['training']['test_years_end'] + 1
            ))
        )
        
        # Merge with environmental features
        train_df = train_df.merge(env_features, on=['Location', 'Year', 'Environment'])
        test_df = test_df.merge(env_features, on=['Location', 'Year', 'Environment'])
        
        # Save processed data
        os.makedirs("data/processed", exist_ok=True)
        geno_imputed.to_csv("data/processed/genotypes_imputed.csv")
        train_df.to_csv("data/processed/train_data.csv", index=False)
        test_df.to_csv("data/processed/test_data.csv", index=False)
        env_features.to_csv("data/processed/environment_features.csv", index=False)
        
        print("\n" + "█" * 60)
        print("█  PREPROCESSING COMPLETE!")
        print("█" * 60)
        print(f"\nSaved to data/processed/:")
        print(f"  - genotypes_imputed.csv ({geno_imputed.shape})")
        print(f"  - train_data.csv ({len(train_df)} records)")
        print(f"  - test_data.csv ({len(test_df)} records)")
        print(f"  - environment_features.csv ({len(env_features)} records)")
        
        return geno_imputed, train_df, test_df, env_features


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    geno, train, test, env = preprocessor.run_full_pipeline()