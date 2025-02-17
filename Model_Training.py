input_shape = (1848, 1)
model = build_model(input_shape)
model.summary()

features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=0.0, neginf=0.0)
features_matrix = features_matrix.reshape((-1, 1848, 1))

features_matrix = np.array(features_matrix, dtype=np.float32)
EEG_augmented_features, augmented_labels = DataAugmentation(features_matrix, mi_labels)

augmented_labels = np.where(augmented_labels == 1, 0, augmented_labels)  # Change mi_labels == 1 to 0
augmented_labels = np.where(augmented_labels == 2, 1, augmented_labels)  # Change mi_labels == 2 to 1

EEG_augmented_features = np.nan_to_num(EEG_augmented_features, nan=0.0, posinf=0.0, neginf=0.0)

scaler = MinMaxScaler()
n_samples, n_features = EEG_augmented_features.shape[0], np.prod(EEG_augmented_features.shape[1:])
features_flattened = EEG_augmented_features.reshape(n_samples, -1)
features_normalized = scaler.fit_transform(features_flattened)
features_normalized = features_normalized.reshape(EEG_augmented_features.shape)

left_mi_features = features_normalized[augmented_labels == 0]
right_mi_features = features_normalized[augmented_labels == 1]

# Combine the features back
features_matrix_variated = np.vstack((left_mi_features, right_mi_features))

# Update the labels to match the new features matrix
mi_labels = np.concatenate((np.zeros(left_mi_features.shape[0]),
                                      np.ones(right_mi_features.shape[0])))

# If needed, shuffle the dataset
indices = np.arange(features_matrix_variated.shape[0])
np.random.shuffle(indices)
features_normalized = features_matrix_variated[indices]
augmented_labels = mi_labels[indices]

encoder = OneHotEncoder(sparse_output=False, categories='auto')

augmented_labels_reshaped = np.asarray(augmented_labels).reshape(-1, 1)

mi_labels_categorical = encoder.fit_transform(augmented_labels_reshaped)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    features_normalized, mi_labels_categorical, test_size=0.5, random_state=42, stratify=mi_labels_categorical
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

input_shape = X_train.shape[1:]  # (33, 7, 4)
num_classes = 2

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis = 1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')