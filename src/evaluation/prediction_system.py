# src/evaluation/prediction_system.py
"""
Prediction System and Genotype Ranking for Breeding Decisions.
Combines model predictions with:
- Selection Index (multi-trait)
- Stability Analysis (across environments)
- Genotype Ranking
- Breeding Recommendations
"""
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class BreedingPredictionSystem:
    """
    Complete breeding prediction and recommendation system.
    """
    
    def __init__(self):
        self.rankings = {}
        self.stability_scores = {}
        self.recommendations = {}
        
    # ================================================================
    # 1. COMPUTE STABILITY SCORE
    # ================================================================
    def compute_stability(self, predictions_df, genotype_col='Genotype_ID',
                          env_col='Environment', pred_col='DL_Prediction'):
        """
        Compute genotype stability across environments.
        """
        print("=" * 50)
        print("STABILITY ANALYSIS")
        print("=" * 50)
        
        stability_data = []
        
        for genotype, group in predictions_df.groupby(genotype_col):
            n_env = len(group)
            mean_perf = group[pred_col].mean()
            std_perf = group[pred_col].std()
            cv = (std_perf / abs(mean_perf) * 100) if mean_perf != 0 else 0
            static_stability = group[pred_col].var()
            
            max_per_env = predictions_df.groupby(env_col)[pred_col].max()
            superiority = 0
            for env in group[env_col].unique():
                env_data = group[group[env_col] == env]
                env_best = max_per_env[env]
                for _, row in env_data.iterrows():
                    superiority += (row[pred_col] - env_best) ** 2
            superiority /= max(n_env, 1)
            
            stability_data.append({
                'Genotype_ID': genotype,
                'N_Environments': n_env,
                'Mean_Performance': mean_perf,
                'Std_Performance': std_perf,
                'CV_pct': cv,
                'Static_Stability': static_stability,
                'Superiority_Measure': superiority,
            })
        
        stability_df = pd.DataFrame(stability_data)
        
        stability_df['Performance_Rank'] = stability_df['Mean_Performance'].rank(ascending=False)
        stability_df['Stability_Rank'] = stability_df['CV_pct'].rank(ascending=True)
        stability_df['Combined_Score'] = (stability_df['Performance_Rank'] + stability_df['Stability_Rank']) / 2
        stability_df = stability_df.sort_values('Combined_Score')
        
        print(f"  Genotypes analyzed: {len(stability_df)}")
        print(f"  Top 5 stable genotypes:")
        for _, row in stability_df.head(5).iterrows():
            print(f"    {row['Genotype_ID']}: Mean={row['Mean_Performance']:.3f}, "
                  f"CV={row['CV_pct']:.1f}%, Score={row['Combined_Score']:.1f}")
        
        self.stability_scores = stability_df
        return stability_df
    
    # ================================================================
    # 2. COMPUTE SELECTION INDEX (FIXED)
    # ================================================================
    def compute_selection_index(self, predictions_df, 
                                 weights={'DL_Prediction': 1.0},
                                 genotype_col='Genotype_ID'):
        """
        Compute a weighted selection index for multi-trait selection.
        FIXED: Only uses numeric columns for aggregation.
        """
        print("\n" + "=" * 50)
        print("SELECTION INDEX")
        print("=" * 50)
        
        # Select only numeric columns + genotype column for groupby
        numeric_cols = predictions_df.select_dtypes(include=[np.number]).columns.tolist()
        group_cols = [genotype_col] + numeric_cols
        
        # Aggregate predictions per genotype (numeric only)
        geno_means = predictions_df[group_cols].groupby(genotype_col).mean()
        
        # Compute selection index
        geno_means['Selection_Index'] = 0.0
        for trait, weight in weights.items():
            if trait in geno_means.columns:
                trait_min = geno_means[trait].min()
                trait_max = geno_means[trait].max()
                if trait_max > trait_min:
                    trait_norm = (geno_means[trait] - trait_min) / (trait_max - trait_min)
                else:
                    trait_norm = 0.0
                geno_means['Selection_Index'] += weight * trait_norm
                print(f"  {trait}: weight={weight}")
        
        # Rank by selection index
        geno_means['Rank'] = geno_means['Selection_Index'].rank(ascending=False)
        geno_means = geno_means.sort_values('Selection_Index', ascending=False)
        
        print(f"\n  Top 5 genotypes by Selection Index:")
        for idx, (geno, row) in enumerate(geno_means.head(5).iterrows()):
            print(f"    Rank {int(row['Rank'])}: {geno} (Index={row['Selection_Index']:.3f})")
        
        return geno_means
    
    # ================================================================
    # 3. FINAL GENOTYPE RANKING
    # ================================================================
    def rank_genotypes(self, predictions_df, stability_df, 
                       performance_weight=0.6, stability_weight=0.4):
        """
        Combine performance and stability for final genotype ranking.
        """
        print("\n" + "=" * 50)
        print("FINAL GENOTYPE RANKING")
        print("=" * 50)
        
        # Use only numeric columns for aggregation
        perf_summary = predictions_df.groupby('Genotype_ID')['DL_Prediction'].agg(['mean', 'std'])
        perf_summary.columns = ['Mean_Pred', 'Std_Pred']
        
        ranking = perf_summary.merge(
            stability_df[['Genotype_ID', 'CV_pct', 'Static_Stability']],
            on='Genotype_ID'
        )
        
        # Normalize to 0-1 for combining
        ranking['Perf_Norm'] = (ranking['Mean_Pred'] - ranking['Mean_Pred'].min()) / \
                                (ranking['Mean_Pred'].max() - ranking['Mean_Pred'].min())
        ranking['Stab_Norm'] = 1 - (ranking['CV_pct'] - ranking['CV_pct'].min()) / \
                                   (ranking['CV_pct'].max() - ranking['CV_pct'].min())
        
        # Final score
        ranking['Final_Score'] = (performance_weight * ranking['Perf_Norm'] + 
                                   stability_weight * ranking['Stab_Norm'])
        ranking['Final_Rank'] = ranking['Final_Score'].rank(ascending=False)
        ranking = ranking.sort_values('Final_Score', ascending=False)
        
        print(f"  Performance weight: {performance_weight}")
        print(f"  Stability weight: {stability_weight}")
        print(f"\n  🏆 TOP 10 GENOTYPES (Ranked):")
        print(f"  {'Rank':<6} {'Genotype':<12} {'Score':<8} {'Mean':<8} {'CV%':<8}")
        print(f"  {'-'*45}")
        for _, row in ranking.head(10).iterrows():
            print(f"  {int(row['Final_Rank']):<6} {row['Genotype_ID']:<12} "
                  f"{row['Final_Score']:<8.3f} {row['Mean_Pred']:<8.3f} {row['CV_pct']:<8.1f}")
        
        self.rankings = ranking
        return ranking
    
    # ================================================================
    # 4. BREEDING RECOMMENDATIONS
    # ================================================================
    def generate_recommendations(self, ranking, top_k=20, selection_pct=0.05):
        """
        Generate breeding recommendations based on rankings.
        """
        print("\n" + "=" * 50)
        print("BREEDING RECOMMENDATIONS")
        print("=" * 50)
        
        n_genotypes = len(ranking)
        n_elite = max(5, int(n_genotypes * 0.05))
        n_promising = max(10, int(n_genotypes * 0.15))
        
        elite = ranking.head(n_elite)
        promising = ranking.iloc[n_elite:n_promising]
        
        print(f"\n  🌟 ELITE GENOTYPES (Top {n_elite}):")
        print(f"     Advance to multi-location variety trials")
        for _, row in elite.head(5).iterrows():
            print(f"     {row['Genotype_ID']} (Score: {row['Final_Score']:.3f})")
        
        print(f"\n  📈 PROMISING GENOTYPES (Next {n_promising - n_elite}):")
        print(f"     Advance to second-year evaluation")
        for _, row in promising.head(5).iterrows():
            print(f"     {row['Genotype_ID']} (Score: {row['Final_Score']:.3f})")
        
        self.recommendations = {
            'elite': elite,
            'promising': promising,
            'elite_ids': elite['Genotype_ID'].tolist(),
            'n_elite': n_elite,
            'n_promising': n_promising,
            'selection_intensity_pct': n_elite / n_genotypes * 100,
        }
        
        print(f"\n  Selection Intensity: {self.recommendations['selection_intensity_pct']:.1f}%")
        print(f"  Total selected: {n_elite + n_promising} genotypes")
        
        return self.recommendations
    
    # ================================================================
    # 5. VISUALIZATION
    # ================================================================
    def plot_ranking(self, ranking, save_path="reports/figures/"):
        """Generate ranking visualization."""
        os.makedirs(save_path, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Performance vs Stability
        ax = axes[0]
        top20 = ranking.head(20)
        others = ranking.iloc[20:]
        ax.scatter(others['Mean_Pred'], others['CV_pct'], 
                  alpha=0.3, c='gray', s=20, label='Other')
        ax.scatter(top20['Mean_Pred'], top20['CV_pct'], 
                  c='red', s=80, edgecolors='black', linewidth=1, label='Top 20')
        ax.set_xlabel('Mean Performance (normalized yield)', fontsize=12)
        ax.set_ylabel('CV% (lower = more stable)', fontsize=12)
        ax.set_title('Genotype Performance vs Stability', fontsize=14)
        ax.legend()
        
        # 2. Top 10 Bar Chart
        ax = axes[1]
        top10 = ranking.head(10)
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, 10))[::-1]
        bars = ax.barh(range(10), top10['Final_Score'].values, color=colors, edgecolor='black')
        ax.set_yticks(range(10))
        ax.set_yticklabels(top10['Genotype_ID'].values)
        ax.set_xlabel('Final Score', fontsize=12)
        ax.set_title('Top 10 Genotypes', fontsize=14)
        ax.invert_yaxis()
        
        # 3. Distribution of Scores
        ax = axes[2]
        ax.hist(ranking['Final_Score'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(ranking['Final_Score'].quantile(0.95), color='red', linestyle='--', 
                  linewidth=2, label='Elite threshold (95th percentile)')
        ax.set_xlabel('Final Score', fontsize=12)
        ax.set_ylabel('Number of Genotypes', fontsize=12)
        ax.set_title('Distribution of Genotype Scores', fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_path}genotype_ranking.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Ranking plot saved to {save_path}genotype_ranking.png")
    
    # ================================================================
    # 6. FULL PIPELINE
    # ================================================================
    def run_full_system(self, predictions_path="outputs/dl_predictions.csv"):
        """Run complete prediction & ranking system."""
        print("█" * 60)
        print("█  BREEDING PREDICTION & RANKING SYSTEM")
        print("█" * 60)
        
        # Load predictions
        print("\nLoading predictions...")
        predictions_df = pd.read_csv(predictions_path)
        print(f"  Predictions: {len(predictions_df)} records")
        print(f"  Genotypes: {predictions_df['Genotype_ID'].nunique()}")
        
        # If no Environment column, create one
        if 'Environment' not in predictions_df.columns:
            predictions_df['Environment'] = predictions_df['Location'] + '_' + predictions_df['Year'].astype(str)
        print(f"  Environments: {predictions_df['Environment'].nunique()}")
        
        # 1. Stability Analysis
        stability_df = self.compute_stability(predictions_df)
        
        # 2. Selection Index
        selection_df = self.compute_selection_index(
            predictions_df,
            weights={'DL_Prediction': 1.0}
        )
        
        # 3. Final Ranking
        ranking = self.rank_genotypes(
            predictions_df, stability_df,
            performance_weight=0.6, stability_weight=0.4
        )
        
        # 4. Recommendations
        recommendations = self.generate_recommendations(ranking)
        
        # 5. Visualization
        self.plot_ranking(ranking)
        
        # Save results
        os.makedirs("outputs", exist_ok=True)
        ranking.to_csv("outputs/final_genotype_ranking.csv")
        stability_df.to_csv("outputs/stability_scores.csv")
        
        print(f"\n✓ Results saved:")
        print(f"  - outputs/final_genotype_ranking.csv")
        print(f"  - outputs/stability_scores.csv")
        print(f"  - reports/figures/genotype_ranking.png")
        
        return ranking, stability_df, recommendations


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    system = BreedingPredictionSystem()
    ranking, stability, recommendations = system.run_full_system()