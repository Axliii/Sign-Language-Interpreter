import pickle
from collections import Counter

# Load data
with open('data.pickle', 'rb') as f:
    d = pickle.load(f)

data = d['data']
labels = d['labels']

print(f"✅ Number of feature vectors: {len(data)}")
print(f"✅ Number of labels         : {len(labels)}")

# Label map: 0–25 = A–Z, 26–35 = digits 0–9
label_map = {i: chr(65 + i) if i <= 25 else str(i - 26) for i in range(36)}

# Count each label
label_counts = Counter(labels)

print("\n📊 Label distribution:")
for label, count in sorted(label_counts.items()):
    readable_label = label_map.get(label, f"? ({label})")
    print(f"  {readable_label:>2} (label {label:>2}): {count} samples")
