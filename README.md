# Engineer-wallet-level-behavioral-features
# 🏦 Aave Credit Scoring using DeFi Wallet Transactions

This project analyzes user transactions on Aave V2 and assigns a **credit score (0–1000)** to each DeFi wallet based on their financial behavior. It includes feature engineering, clustering, and scoring logic.

## 📁 Project Structure


---

## 🚀 How It Works

1. **Load Data**
   - Parses a JSON file containing Aave V2 user transactions.

2. **Preprocess & Feature Engineering**
   - Cleans timestamps and numerical fields.
   - Computes:
     - Total USD volume per action (deposit, borrow, repay, redeemUnderlying)
     - Transaction counts
     - Active days and span of activity
     - Repayment ratio, Redeem ratio

3. **Score Wallets**
   - Standardizes features
   - Applies **KMeans clustering** to group wallet behavior
   - Assigns a score to each cluster based on:
     - High deposit volume ✅
     - High repayment ratio ✅
     - Low redeem ratio and borrowing frequency ❌
   - Normalizes cluster scores to a 0–1000 scale.

---

## 📊 Sample Output

| userWallet | score |
|------------|--------|
| 0xabc123... | 894.3 |
| 0xdef456... | 472.1 |
| 0xghi789... | 211.5 |

---

## 🛠️ Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt


✅ Running the Pipeline
bash
Copy
Edit
python src/main.py
This will:

Read the input data

Generate features

Compute scores

Save the output in output/wallet_scores.csv

📬 Output
The final output file (wallet_scores.csv) contains:

userWallet: Wallet address

score: Credit score in the 0–1000 range



 Notes
This model is unsupervised and behavior-based.

Feature weights in scoring logic can be tuned based on business context.

Edge cases (no activity, extreme values) are handled with smoothing and clipping.

📄 License
This project is for educational purposes and is not financial advice.



