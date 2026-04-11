import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# load data
df = pd.read_csv("data/churn_data.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# helper
def save_show(title, filename):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

# --- Churn Count ---
plt.figure()
ax = sns.countplot(data=df, x="Churn", hue="Churn", legend=False)

total = len(df)
for p in ax.patches:
    pct = f'{100 * p.get_height()/total:.1f}%'
    ax.annotate(pct, (p.get_x() + p.get_width()/2, p.get_height()), ha='center')

save_show("Customer Churn Distribution", "eda_churn_count.png")

churn_pct = df["Churn"].value_counts(normalize=True) * 100
print(f"\nChurn: {churn_pct['Yes']:.1f}% | No Churn: {churn_pct['No']:.1f}%")

# --- Tenure ---
plt.figure()
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True)
save_show("Tenure Distribution", "eda_tenure.png")

avg_tenure = df.groupby("Churn")["tenure"].mean()
print(f"Avg tenure → Churn: {avg_tenure['Yes']:.1f}, No Churn: {avg_tenure['No']:.1f}")

# --- Monthly Charges ---
plt.figure()
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
save_show("Monthly Charges by Churn", "eda_monthly_charges.png")

avg_charge = df.groupby("Churn")["MonthlyCharges"].mean()
print(f"Avg charges → Churn: {avg_charge['Yes']:.2f}, No Churn: {avg_charge['No']:.2f}")

# --- Correlation ---
df["Churn_num"] = df["Churn"].map({"Yes": 1, "No": 0})
corr = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn_num"]].corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True)
save_show("Feature Correlation", "eda_correlation.png")

print(f"Tenure vs Churn: {corr['Churn_num']['tenure']:.2f}")
print(f"Charges vs Churn: {corr['Churn_num']['MonthlyCharges']:.2f}")

print("\nEDA done ✅")