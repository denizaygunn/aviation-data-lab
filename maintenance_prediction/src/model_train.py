import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load example sensor data (you can replace this with real data later)
df = pd.read_csv("../data/sensor_data.csv")

# Split features and target label
X = df.drop("failure", axis=1)
y = df["failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and performance report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))