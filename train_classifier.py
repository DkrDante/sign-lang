
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load preprocessed dataset
with open('./dataset.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Ensure all feature vectors have the same length
max_length = max(len(sample) for sample in data_dict['data'])
data = [sample + [0] * (max_length - len(sample)) for sample in data_dict['data']]  # Padding

# Convert to NumPy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(data_dict['labels'])

# Encode labels as numbers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # Convert string labels (A-Z, gestures) into numbers

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42)
model.fit(x_train, y_train)

# Predict
y_predict = model.predict(x_test)

# Accuracy Score
score = accuracy_score(y_test, y_predict)
print(f'✅ {score * 100:.2f}% of samples classified correctly!')

# Save Model & Label Encoder
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)

print("🎯 Model and label encoder saved successfully!")

