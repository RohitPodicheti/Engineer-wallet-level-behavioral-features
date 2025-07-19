
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

def score_wallets(features_df, n_clusters=5):
    feature_cols = [col for col in features_df.columns if col not in ['userWallet']]

    features_df[feature_cols] = features_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df[feature_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_df['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_stats = features_df.groupby('cluster').agg({
        'total_volume_usd_deposit': 'mean',
        'total_volume_usd_redeemunderlying': 'mean',
        'repay_ratio': 'mean',
        'redeem_ratio': 'mean',
        'borrow_to_deposit_ratio': 'mean',
        'avg_volume_usd_per_txn': 'mean',
        'activity_intensity': 'mean',
    }).fillna(0)

    cluster_stats['raw_score'] = (
        np.log1p(cluster_stats['total_volume_usd_deposit']) * 100 +
        cluster_stats['repay_ratio'] * 300 -
        cluster_stats['redeem_ratio'] * 150 -
        cluster_stats['borrow_to_deposit_ratio'] * 100 +
        cluster_stats['avg_volume_usd_per_txn'] * 20 +
        cluster_stats['activity_intensity'] * 10
    )

    raw_scores = cluster_stats['raw_score']
    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if max_score != min_score:
        cluster_stats['score'] = ((raw_scores - min_score) / (max_score - min_score)) * 1000
    else:
        cluster_stats['score'] = 500

    score_map = cluster_stats['score'].to_dict()
    features_df['score'] = features_df['cluster'].map(score_map)

    return features_df[['userWallet', 'score']]
