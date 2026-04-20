import pandas as pd
import numpy as np

class PredictiveElectoralMatrix:
    def __init__(self, constituency_name: str, custom_weights: dict = None):
        self.constituency = constituency_name
        self.factors = [
            "Incumbency_Effect",
            "Party_Strength",
            "Past_Work_OSINT",
            "Personal_Base",
            "Religious_Caste_Base",
            "Digital_Sentiment"
        ]
        
        # Default weights based on typical Indian election dynamics
        # (can be tuned from historical voting data per constituency)
        self.weights = custom_weights or {
            "Incumbency_Effect": 0.20,
            "Party_Strength": 0.15,
            "Past_Work_OSINT": 0.25,
            "Personal_Base": 0.15,
            "Religious_Caste_Base": 0.15,
            "Digital_Sentiment": 0.10
        }
        
        # Validate & normalize weights
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 0.01:
            self.weights = {k: v / total_w for k, v in self.weights.items()}
        
        self.candidates_data = []

    def add_candidate(self, name: str, scores_dict: dict, notes: str = ""):
        """
        Add a candidate with factor scores (0-10 scale).
        Scores derived from:
        - OSINT (sentiment analysis, news scraping)
        - Historical data (past margins, census, legislative records)
        - Qualitative mapping logic (example in usage)
        """
        # Ensure all factors present (default neutral if missing)
        for factor in self.factors:
            if factor not in scores_dict:
                scores_dict[factor] = 5.0
        
        entry = {"Candidate": name, **scores_dict, "Notes": notes}
        self.candidates_data.append(entry)

    def build_matrix(self) -> pd.DataFrame:
        """Builds the full weighted comparison matrix + PoW"""
        if not self.candidates_data:
            raise ValueError("No candidates added!")
        
        df = pd.DataFrame(self.candidates_data)
        
        # Weighted score calculation
        for factor in self.factors:
            df[f"{factor}_wt"] = df[factor] * self.weights[factor]
        
        df["Total_Weighted_Score"] = df[[f"{f}_wt" for f in self.factors]].sum(axis=1)
        
        # Head-to-head comparison ready (raw scores + weighted)
        # PoW Forecasting: Softmax normalization (probabilistic multi-candidate model)
        scores = df["Total_Weighted_Score"].values
        exp_scores = np.exp(scores - np.max(scores))  # Numerically stable
        df["PoW_Base (%)"] = (exp_scores / exp_scores.sum()) * 100
        
        # Additional logic: Simulate turnout/swing-voter variability
        # (Digital sentiment & real-time OSINT introduce volatility)
        np.random.seed(42)  # Reproducible
        swing = np.random.uniform(0.85, 1.15, len(df))  # ±15% swing factor
        df["Adjusted_PoW (%)"] = (df["PoW_Base (%)"] * swing)
        df["Adjusted_PoW (%)"] = df["Adjusted_PoW (%)"] / df["Adjusted_PoW (%)"].sum() * 100
        
        # Return clean matrix
        cols = ["Candidate"] + self.factors + ["Total_Weighted_Score", "PoW_Base (%)", "Adjusted_PoW (%)", "Notes"]
        return df[cols].round(2)

    def run_monte_carlo(self, df: pd.DataFrame, simulations: int = 1000) -> pd.DataFrame:
        """Monte Carlo simulation for robust PoW under uncertainty"""
        base_scores = df["Total_Weighted_Score"].values.copy()
        sim_results = []
        
        for _ in range(simulations):
            # Simulate real-world noise: sentiment shifts, turnout changes, OSINT updates
            noise = np.random.normal(0, 1.0, len(base_scores))
            noisy_scores = np.maximum(base_scores + noise, 0)
            exp_n = np.exp(noisy_scores - np.max(noisy_scores))
            pow_sim = (exp_n / exp_n.sum()) * 100
            sim_results.append(pow_sim)
        
        sim_df = pd.DataFrame(sim_results, columns=df["Candidate"])
        
        summary = pd.DataFrame({
            "Mean_PoW (%)": sim_df.mean().round(1),
            "Std_Dev": sim_df.std().round(1),
            "5th_Percentile": sim_df.quantile(0.05).round(1),
            "95th_Percentile": sim_df.quantile(0.95).round(1)
        })
        return summary

    def print_strategy_insights(self, df: pd.DataFrame):
        """Logic-based strategic recommendations"""
        print("\n=== STRATEGIC INSIGHTS (Actionable for Campaign) ===")
        leader = df.loc[df["Adjusted_PoW (%)"].idxmax(), "Candidate"]
        print(f"LEADING CANDIDATE: {leader} (PoW: {df['Adjusted_PoW (%)'].max():.1f}%)")
        
        # Example gap analysis
        incumbent_row = df[df["Candidate"].str.contains("Incumbent", na=False)]
        if not incumbent_row.empty:
            inc = incumbent_row.iloc[0]
            print("\nIncumbent Gaps:")
            for f in self.factors:
                if inc[f] < 6:
                    print(f"  • Weak on {f.replace('_', ' ')} → Prioritize targeted OSINT-driven outreach")
        
        print("\nGeneral Recommendations:")
        print("• Strong Digital Sentiment gaps → Launch viral social media + local influencer campaign")
        print("• Religious/Caste Base weakness → Consider alliance building or community events")
        print("• Anti-incumbency detected → Highlight verifiable Past Work (project photos, fund utilization reports)")


# ====================== FULLY FUNCTIONAL EXAMPLE USAGE ======================
if __name__ == "__main__":
    print("🚀 Predictive Electoral Analytics Matrix")
    print("Demo Constituency: Nawanshahr, Punjab (India)\n")
    
    analyzer = PredictiveElectoralMatrix("Nawanshahr, Punjab")
    
    # Add candidates using data from problem statement
    # (Scores mapped qualitatively → quantitatively with explicit logic)
    analyzer.add_candidate(
        name="Candidate A (Incumbent)",
        scores_dict={
            "Incumbency_Effect": 4.0,      # High but "Anticipated Anti-incumbency" penalty
            "Party_Strength": 9.0,         # Strong National Presence
            "Past_Work_OSINT": 7.0,        # Verified Dev. Projects
            "Personal_Base": 8.0,          # Traditional Loyalists
            "Religious_Caste_Base": 6.0,   # Split Support
            "Digital_Sentiment": 5.0       # Neutral/Negative
        },
        notes="Anticipated anti-incumbency"
    )
    
    analyzer.add_candidate(
        name="Candidate B (Challenger)",
        scores_dict={
            "Incumbency_Effect": 0.0,      # N/A (Challenger)
            "Party_Strength": 7.0,         # Regional Powerhouse
            "Past_Work_OSINT": 9.0,        # High Social Activism
            "Personal_Base": 9.0,          # Youth/Urban Appeal
            "Religious_Caste_Base": 8.0,   # Solidified Block
            "Digital_Sentiment": 9.0       # Highly Positive
        },
        notes="Strong digital & youth momentum"
    )
    
    analyzer.add_candidate(
        name="Candidate C (Independent)",
        scores_dict={
            "Incumbency_Effect": 0.0,      # N/A
            "Party_Strength": 3.0,         # Weak / Local Only
            "Past_Work_OSINT": 4.0,        # Limited Record
            "Personal_Base": 7.0,          # Hyper-Local Community
            "Religious_Caste_Base": 9.0,   # Minority Niche (strong in block)
            "Digital_Sentiment": 3.0       # Low Visibility
        },
        notes="Hyper-local community focus"
    )
    
    # Generate full matrix + PoW
    matrix = analyzer.build_matrix()
    print("1. MULTI-DIMENSIONAL COMPARISON MATRIX")
    print(matrix.to_string(index=False))
    
    # Monte Carlo probabilistic forecasting
    mc_summary = analyzer.run_monte_carlo(matrix, simulations=1000)
    print("\n2. PoW FORECAST (Monte Carlo - 1000 turnout/swing simulations)")
    print(mc_summary)
    
    # Strategic dashboard
    analyzer.print_strategy_insights(matrix)
    
    print("\n Model complete! Ready for real OSINT integration (API feeds for sentiment, census CSV, etc.).")
    print("Extend by: adding CSV input, ML model (scikit-learn on past elections), or live dashboard (Streamlit).")
