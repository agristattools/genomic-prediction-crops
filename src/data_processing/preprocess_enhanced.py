# -*- coding: utf-8 -*-
# src/data_processing/preprocess_enhanced.py
"""
Quick preprocessing of enhanced data for training the multi-pathway predictor.
"""
import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

print("=" * 60)
print("PREPROCESSING ENHANCED DATA")
print("=" * 60)

# Load raw enhanced data
print("\n1. Loading raw enhanced data...")
geno_raw = pd.read_csv("data/raw/genotypes_enhanced.csv", index_col=0)
pheno_raw = pd.read_csv("data/raw/phenotypes_enhanced.csv")
weather_raw = pd.read_csv("data/raw/weather_enhanced.csv")

print(f"   Genotypes: {geno_raw.shape}")
print(f"   Phenotypes: {pheno_raw.shape}")
print(f"   Weather: {weather_raw.shape}")

# Quick SNP filtering (MAF > 1%, call rate > 80%)
print("\n2. Filtering SNPs...")
call_rate = (geno_raw != -1).mean(axis=0)
geno_filt = geno_raw.loc[:, call_rate >= 0.80]

# MAF filter
maf_values = []
for snp in geno_filt.columns:
    values = geno_filt[snp].values
    valid = values[values != -1]
    if len(valid) > 0:
        alt_freq = valid.sum() / (2 * len(valid))
        maf_values.append(min(alt_freq, 1 - alt_freq))
    else:
        maf_values.append(0)

maf_series = pd.Series(maf_values, index=geno_filt.columns)
geno_final = geno_filt.loc[:, maf_series >= 0.01]
print(f"   SNPs retained: {geno_final.shape[1]} / {geno_raw.shape[1]}")

# Quick imputation
print("\n3. Imputing missing genotypes...")
geno_imputed = geno_final.copy()
for col in geno_imputed.columns:
    vals = geno_imputed[col].copy()
    valid = vals[vals != -1]
    if len(valid) > 0:
        mode_val = int(pd.Series(valid).mode().iloc[0])
        vals[vals == -1] = mode_val
        geno_imputed[col] = vals.astype(np.int8)

# Normalize phenotypes within environment
print("\n4. Normalizing phenotypes...")
pheno_norm = pheno_raw.copy()
trait_cols = ['Yield_bu_ac', 'PlantHeight_cm', 'EarHeight_cm', 'DaysToAnthesis',
              'DaysToSilk', 'GrainMoisture_pct', 'KernelRowNumber', 
              'KernelWeight_100_g', 'StayGreen_Score', 'DiseaseResistance_Score']

for trait in trait_cols:
    norm_col = f"{trait}_norm"
    for env in pheno_norm['Environment'].unique():
        mask = pheno_norm['Environment'] == env
        values = pheno_norm.loc[mask, trait]
        mean, std = values.mean(), values.std()
        if std > 0:
            pheno_norm.loc[mask, norm_col] = (values - mean) / std
        else:
            pheno_norm.loc[mask, norm_col] = 0
    print(f"   ✓ {trait} → {norm_col}")

# Environmental features
print("\n5. Computing environmental features...")
env_features = []
for (loc, year), group in weather_raw.groupby(['Location', 'Year']):
    tmean = (group['Temp_Max_C'] + group['Temp_Min_C']) / 2
    gdd = np.maximum(tmean - 10, 0).sum()
    
    dry_days = (group['Precipitation_mm'] < 1).astype(int)
    consecutive = max_consecutive = 0
    for d in dry_days:
        if d == 1:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0
    
    env_features.append({
        'Environment': f"{loc}_{year}",
        'GDD': round(float(gdd), 1),
        'HeatStress_Days': int((group['Temp_Max_C'] > 35).sum()),
        'MaxDrySpell_Days': max_consecutive,
        'TotalPrecip_mm': round(float(group['Precipitation_mm'].sum()), 1),
        'MeanTemp_C': round(float(tmean.mean()), 1),
        'DiurnalRange_C': round(float((group['Temp_Max_C'] - group['Temp_Min_C']).mean()), 1),
        'MeanHumidity_pct': round(float(group['Humidity_pct'].mean()), 1),
        'MeanSolarRadiation': round(float(group['SolarRadiation_MJ_m2'].mean()), 1),
    })

env_df = pd.DataFrame(env_features)
print(f"   ✓ {len(env_df)} environment profiles")

# Train/test split by year
print("\n6. Splitting data...")
train_mask = pheno_norm['Year'].between(2015, 2020)
test_mask = pheno_norm['Year'].between(2021, 2022)

train_df = pheno_norm[train_mask].copy()
test_df = pheno_norm[test_mask].copy()

# Merge with environment features
env_indexed = env_df.set_index('Environment')
train_df = train_df.join(env_indexed, on='Environment')
test_df = test_df.join(env_indexed, on='Environment')

print(f"   Train: {len(train_df)} records (2015-2020)")
print(f"   Test:  {len(test_df)} records (2021-2022)")

# Save processed data
print("\n7. Saving processed data...")
os.makedirs("data/processed", exist_ok=True)
geno_imputed.to_csv("data/processed/genotypes_imputed_enhanced.csv")
train_df.to_csv("data/processed/train_data_enhanced.csv", index=False)
test_df.to_csv("data/processed/test_data_enhanced.csv", index=False)
env_df.to_csv("data/processed/environment_features_enhanced.csv", index=False)

print("\n" + "=" * 60)
print("✓ ENHANCED DATA PREPROCESSING COMPLETE!")
print("=" * 60)
print(f"\nSaved to data/processed/:")
print(f"  - genotypes_imputed_enhanced.csv")
print(f"  - train_data_enhanced.csv ({len(train_df)} records)")
print(f"  - test_data_enhanced.csv ({len(test_df)} records)")
print(f"  - environment_features_enhanced.csv")