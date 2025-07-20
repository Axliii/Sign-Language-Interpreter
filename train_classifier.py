import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import collections

# Load the feature vectors and labels
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Display class distribution
label_counts = collections.Counter(labels)
print("📊 Dataset class distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  Class {label}: {count} samples")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict on the test set and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"\n✅ Accuracy: {score * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_predict))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("\n💾 Model saved successfully to model.p")
