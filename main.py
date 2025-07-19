
import os
from src.feature_engineering import load_data, preprocess, extract_features
from src.model import score_wallets

def main():
    input_path = 'data/user-wallet-transactions.json'
    output_path = 'output/wallet_scores.csv'

    df_raw = load_data(input_path)
    df_processed = preprocess(df_raw)
    features_df = extract_features(df_processed)
    scored_df = score_wallets(features_df)

    os.makedirs('output', exist_ok=True)
    scored_df.to_csv(output_path, index=False)
    print(scored_df.head(10))
    print(f"Score range: {scored_df['score'].min()} – {scored_df['score'].max()}")
    print(f"✅ Saved wallet scores to {output_path}")

if __name__ == "__main__":
    main()
