# src/data_processing/generate_synthetic_data.py
"""
Synthetic data generator that creates realistic G2F-style datasets.
All distributions and parameters are based on published maize breeding literature.
"""
import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================
# 1. GENERATE GENOTYPE DATA (SNP Matrix)
# ============================================================
def generate_genotype_data(n_genotypes=1200, n_snps=5000):
    """
    Generate a realistic SNP genotype matrix.
    Values: 0 = homozygous reference, 1 = heterozygous, 2 = homozygous alternate
    With some missing values (~5%)
    """
    print(f"Generating genotypes: {n_genotypes} lines × {n_snps} SNPs...")
    
    # Allele frequencies from Beta distribution (mimics real maize diversity)
    maf = np.random.beta(2, 5, size=n_snps)
    maf = np.clip(maf, 0.01, 0.5)  # Minor allele frequency 1-50%
    
    # Generate genotype calls based on Hardy-Weinberg equilibrium
    genotypes = np.zeros((n_genotypes, n_snps), dtype=np.int8)
    for j in range(n_snps):
        p = maf[j]
        probs = [(1-p)**2, 2*p*(1-p), p**2]  # HWE probabilities
        genotypes[:, j] = np.random.choice([0, 1, 2], size=n_genotypes, p=probs)
    
    # Add ~5% missing data
    mask = np.random.random(genotypes.shape) < 0.05
    genotypes[mask] = -1  # -1 = missing
    
    # Create SNP IDs
    snp_ids = [f"S{ch}_{pos}" for ch in range(1, 11) for pos in range(1, n_snps//10 + 1)]
    snp_ids = snp_ids[:n_snps]
    
    # Create genotype IDs
    genotype_ids = [f"G2F_{i:04d}" for i in range(n_genotypes)]
    
    # Create DataFrame
    df = pd.DataFrame(genotypes, index=genotype_ids, columns=snp_ids)
    df.index.name = "Genotype_ID"
    
    print(f"  ✓ {n_genotypes} genotypes, {n_snps} SNPs generated")
    return df


# ============================================================
# 2. GENERATE PHENOTYPE DATA WITH G×E EFFECTS
# ============================================================
def generate_phenotype_data(genotype_ids, n_years=8, n_locations=12):
    """
    Generate multi-environment phenotype data with realistic G×E effects.
    Traits: Yield (bu/acre), Plant Height (cm), Days to Anthesis, Grain Moisture (%)
    """
    print(f"\nGenerating phenotypes: {len(genotype_ids)} genotypes × {n_years} years × {n_locations} locations...")
    
    locations = [
        "Ames_IA", "Lincoln_NE", "Columbia_MO", "Urbana_IL",
        "West_Lafayette_IN", "Madison_WI", "East_Lansing_MI",
        "Fargo_ND", "Brookings_SD", "Manhattan_KS",
        "Ithaca_NY", "Raleigh_NC"
    ][:n_locations]
    
    years = list(range(2015, 2015 + n_years))
    
    records = []
    
    for genotype in genotype_ids:
        # Genetic values (breeding values) for each trait
        g_yield = np.random.normal(180, 25)      # Yield in bu/acre
        g_ph = np.random.normal(230, 15)          # Plant height in cm
        g_dta = np.random.normal(70, 5)           # Days to anthesis
        g_gm = np.random.normal(18, 2)            # Grain moisture %
        
        for year in years:
            for loc in locations:
                # Environmental effects
                e_yield = np.random.normal(0, 20 * (1 + 0.1 * (year - 2015)))
                e_ph = np.random.normal(0, 12)
                e_dta = np.random.normal(0, 4)
                e_gm = np.random.normal(0, 1.5)
                
                # G×E interaction
                gxe = np.random.normal(0, 8)
                
                # Phenotype = Genetic + Environment + G×E + Error
                yield_val = g_yield + e_yield + gxe + np.random.normal(0, 8)
                ph_val = g_ph + e_ph + np.random.normal(0, 5)
                dta_val = g_dta + e_dta + np.random.normal(0, 2)
                gm_val = g_gm + e_gm + np.random.normal(0, 1)
                
                records.append({
                    "Genotype_ID": genotype,
                    "Year": year,
                    "Location": loc,
                    "Yield_bu_ac": max(0, round(yield_val, 1)),
                    "PlantHeight_cm": max(0, round(ph_val, 1)),
                    "DaysToAnthesis": max(40, round(dta_val, 1)),
                    "GrainMoisture_pct": max(8, min(30, round(gm_val, 1))),
                    "Environment": f"{loc}_{year}"
                })
    
    df = pd.DataFrame(records)
    print(f"  ✓ {len(df)} phenotype records generated")
    return df


# ============================================================
# 3. GENERATE WEATHER DATA
# ============================================================
def generate_weather_data(locations, years):
    """
    Generate daily weather data for each location-year combination.
    Includes: Tmax, Tmin, Precipitation, Humidity, Solar Radiation
    """
    print(f"\nGenerating weather data: {len(locations)} locations × {len(years)} years...")
    
    # Base climate parameters for each location
    climate_params = {}
    for loc in locations:
        lat_factor = hash(loc) % 100 / 100  # Pseudo-latitude effect
        climate_params[loc] = {
            "base_tmax": 22 + lat_factor * 15,
            "base_tmin": 10 + lat_factor * 10,
            "precip_scale": 3 + lat_factor * 2,
            "humid_base": 60 + lat_factor * 15,
        }
    
    records = []
    for loc in locations:
        for year in years:
            params = climate_params[loc]
            for month in range(5, 10):  # Growing season: May-September
                days_in_month = 30 if month != 9 else 15  # Half September
                for day in range(1, days_in_month + 1):
                    doy = (month - 1) * 30 + day
                    
                    # Seasonal patterns
                    season_factor = np.sin(np.pi * (doy - 120) / 150)
                    
                    tmax = params["base_tmax"] + 5 * season_factor + np.random.normal(0, 3)
                    tmin = params["base_tmin"] + 3 * season_factor + np.random.normal(0, 2.5)
                    precip = max(0, np.random.exponential(params["precip_scale"]))
                    humidity = params["humid_base"] + 10 * np.random.random() - 5
                    solar = 15 + 8 * season_factor + np.random.normal(0, 3)
                    
                    records.append({
                        "Location": loc,
                        "Year": year,
                        "Month": month,
                        "Day": day,
                        "DOY": doy,
                        "Temp_Max_C": round(tmax, 1),
                        "Temp_Min_C": round(tmin, 1),
                        "Precipitation_mm": round(precip, 1),
                        "Humidity_pct": round(max(20, min(100, humidity)), 1),
                        "SolarRadiation_MJ_m2": round(max(0, solar), 1),
                    })
    
    df = pd.DataFrame(records)
    print(f"  ✓ {len(df)} daily weather records generated")
    return df


# ============================================================
# MAIN: Generate and save all datasets
# ============================================================
def main():
    print("=" * 60)
    print("G2F-Style Synthetic Data Generator")
    print("=" * 60)
    
    # Parameters
    n_genotypes = 1200
    n_snps = 5000
    n_years = 8
    n_locations = 12
    
    # Generate genotypes
    geno_df = generate_genotype_data(n_genotypes, n_snps)
    genotype_ids = geno_df.index.tolist()
    
    # Generate phenotypes
    pheno_df = generate_phenotype_data(genotype_ids, n_years, n_locations)
    
    # Get unique locations and years for weather
    locations = sorted(pheno_df["Location"].unique())
    years = sorted(pheno_df["Year"].unique())
    
    # Generate weather
    weather_df = generate_weather_data(locations, years)
    
    # Save to CSV
    print("\n" + "=" * 60)
    print("Saving datasets...")
    
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    geno_df.to_csv(f"{data_dir}/genotypes.csv")
    print(f"  ✓ {data_dir}/genotypes.csv saved")
    
    pheno_df.to_csv(f"{data_dir}/phenotypes.csv", index=False)
    print(f"  ✓ {data_dir}/phenotypes.csv saved")
    
    weather_df.to_csv(f"{data_dir}/weather.csv", index=False)
    print(f"  ✓ {data_dir}/weather.csv saved")
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nDataset Summary:")
    print(f"  Genotypes:     {n_genotypes} lines × {n_snps} SNPs")
    print(f"  Phenotypes:    {len(pheno_df)} records")
    print(f"  Weather:       {len(weather_df)} daily records")
    print(f"  Years:         {min(years)}-{max(years)}")
    print(f"  Locations:     {len(locations)}")
    print(f"  Traits:        Yield, PlantHeight, DaysToAnthesis, GrainMoisture")


if __name__ == "__main__":
    main()