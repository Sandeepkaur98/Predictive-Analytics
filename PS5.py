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
        
        self.weights = custom_weights or {
            "Incumbency_Effect": 0.20,
            "Party_Strength": 0.15,
            "Past_Work_OSINT": 0.25,
            "Personal_Base": 0.15,
            "Religious_Caste_Base": 0.15,
            "Digital_Sentiment": 0.10
        }
        
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
        
        for factor in self.factors:
            df[f"{factor}_wt"] = df[factor] * self.weights[factor]
        
        df["Total_Weighted_Score"] = df[[f"{f}_wt" for f in self.factors]].sum(axis=1)
        
        scores = df["Total_Weighted_Score"].values
        exp_scores = np.exp(scores - np.max(scores))  
        df["PoW_Base (%)"] = (exp_scores / exp_scores.sum()) * 100
        
        np.random.seed(42)  
        swing = np.random.uniform(0.85, 1.15, len(df)) 
        df["Adjusted_PoW (%)"] = (df["PoW_Base (%)"] * swing)
        df["Adjusted_PoW (%)"] = df["Adjusted_PoW (%)"] / df["Adjusted_PoW (%)"].sum() * 100
        
        cols = ["Candidate"] + self.factors + ["Total_Weighted_Score", "PoW_Base (%)", "Adjusted_PoW (%)", "Notes"]
        return df[cols].round(2)

    def run_monte_carlo(self, df: pd.DataFrame, simulations: int = 1000) -> pd.DataFrame:
        """Monte Carlo simulation for robust PoW under uncertainty"""
        base_scores = df["Total_Weighted_Score"].values.copy()
        sim_results = []
        
        for _ in range(simulations):
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


if __name__ == "__main__":
    print(" Predictive Electoral Analytics Matrix")
    print("Demo Constituency: Nawanshahr, Punjab (India)\n")
    
    analyzer = PredictiveElectoralMatrix("Nawanshahr, Punjab")
    
    analyzer.add_candidate(
        name="Candidate A (Incumbent)",
        scores_dict={
            "Incumbency_Effect": 4.0,     
            "Party_Strength": 9.0,        
            "Past_Work_OSINT": 7.0,        
            "Personal_Base": 8.0,          
            "Religious_Caste_Base": 6.0, 
            "Digital_Sentiment": 5.0      
        },
        notes="Anticipated anti-incumbency"
    )
    
    analyzer.add_candidate(
        name="Candidate B (Challenger)",
        scores_dict={
            "Incumbency_Effect": 0.0,   
            "Party_Strength": 7.0,        
            "Past_Work_OSINT": 9.0,       
            "Personal_Base": 9.0,          
            "Religious_Caste_Base": 8.0,  
            "Digital_Sentiment": 9.0      
        },
        notes="Strong digital & youth momentum"
    )
    
    analyzer.add_candidate(
        name="Candidate C (Independent)",
        scores_dict={
            "Incumbency_Effect": 0.0,     
            "Party_Strength": 3.0,         
            "Past_Work_OSINT": 4.0,       
            "Personal_Base": 7.0,          
            "Religious_Caste_Base": 9.0,   
            "Digital_Sentiment": 3.0       
        },
        notes="Hyper-local community focus"
    )
    
    matrix = analyzer.build_matrix()
    print("1. MULTI-DIMENSIONAL COMPARISON MATRIX")
    print(matrix.to_string(index=False))
    
    mc_summary = analyzer.run_monte_carlo(matrix, simulations=1000)
    print("\n2. PoW FORECAST (Monte Carlo - 1000 turnout/swing simulations)")
    print(mc_summary)
        analyzer.print_strategy_insights(matrix)
    
    print("\n Model complete! Ready for real OSINT integration (API feeds for sentiment, census CSV, etc.).")
    print("Extend by: adding CSV input, ML model (scikit-learn on past elections), or live dashboard (Streamlit).")
