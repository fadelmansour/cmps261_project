import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv("Sentiment_analysis_dataset.csv")

# Drop rows with missing Statement or Status
df = df.dropna(subset=['Statement', 'Status'])

# Remove entries where Statement is just whitespace
df = df[df['Statement'].str.strip().astype(bool)]

# Reset index after dropping rows
df.reset_index(drop=True, inplace=True)

# Split data
X = df['Statement']
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Convert text data into numerical form (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Apply SMOTE for balancing classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train_resampled, y_train_resampled)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Optional: Plot class distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='Status', order=df['Status'].value_counts().index)
plt.title('Class Distribution')
plt.xlabel('Count')
plt.ylabel('Mental Health Category')
plt.show()

# Sample Predictions
sample_statements = [
    "I feel so hopeless and empty inside.",
    "I'm doing okay today, just a bit tired.",
    "I constantly worry about everything, it's exhausting."
]

sample_tfidf = vectorizer.transform(sample_statements)
predictions = model.predict(sample_tfidf)

for statement, pred in zip(sample_statements, predictions):
    print(f"Statement: {statement}\nPredicted Mental Health Status: {pred}\n")

