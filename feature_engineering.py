
import pandas as pd
import numpy as np
import json

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    return df

def preprocess(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['amount'] = pd.to_numeric(df['actionData.amount'], errors='coerce')
    df['asset_price_usd'] = pd.to_numeric(df['actionData.assetPriceUSD'], errors='coerce')
    df['amount_usd'] = df['amount'] * df['asset_price_usd']
    return df

def extract_features(df):
    df['date'] = df['timestamp'].dt.date

    grouped = df.groupby(['userWallet', 'action']).agg(
        total_volume_usd=pd.NamedAgg(column="amount_usd", aggfunc="sum"),
        txn_count=pd.NamedAgg(column="amount_usd", aggfunc="count")
    ).reset_index()

    features = grouped.pivot(index="userWallet", columns="action", values=["total_volume_usd", "txn_count"]).fillna(0)
    features.columns = [f"{i}_{j}" for i, j in features.columns]
    features.reset_index(inplace=True)

    txn_span = df.groupby('userWallet')['timestamp'].agg(lambda x: (x.max() - x.min()).days + 1)
    active_days = df.groupby('userWallet')['date'].nunique()

    features = features.merge(txn_span.rename("txn_span_days"), on="userWallet", how="left")
    features = features.merge(active_days.rename("active_days"), on="userWallet", how="left")

    features['redeem_ratio'] = features.get('total_volume_usd_redeemunderlying', 0) / features.get('total_volume_usd_deposit', 1)
    features['repay_ratio'] = features.get('total_volume_usd_repay', 0) / features.get('total_volume_usd_borrow', 1)
    features['borrow_to_deposit_ratio'] = features.get('total_volume_usd_borrow', 0) / features.get('total_volume_usd_deposit', 1)
    features['avg_volume_usd_per_txn'] = features.filter(like='total_volume_usd').sum(axis=1) / features.filter(like='txn_count').sum(axis=1)
    features['activity_intensity'] = features.get('txn_count_borrow', 0) / features['active_days']

    features.replace([np.inf, -np.inf], 0, inplace=True)

    return features
