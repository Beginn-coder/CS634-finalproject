import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import brier_score_loss

# Load the sampled dataset
dataset_sampled = pd.read_csv('Lynardakis_Nicholas_sampled_dataset.csv')

# Store 'url' column separately if it exists
if 'url' in dataset_sampled.columns:
    url_data = dataset_sampled['url'].copy()  # Copy 'url' column for later use
    print("Vectorizing 'url' column...")
    vectorizer = CountVectorizer()
    X_urls_vectorized = vectorizer.fit_transform(url_data).toarray()
    dataset_sampled = dataset_sampled.drop(columns=['url'])
    X = np.hstack((dataset_sampled.iloc[:, :-1].values, X_urls_vectorized))
else:
    print("The 'url' column is not present in the dataset.")
    X = dataset_sampled.iloc[:, :-1].values  # Use only other features

y = dataset_sampled.iloc[:, -1].values

# Split Dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Label Encoding
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Function to calculate all metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    # Metrics using formulas
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)  # True Positive Rate (Recall)
    specificity = TN / (TN + FP)  # True Negative Rate (Specificity)
    precision = TP / (TP + FP)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
    error_rate = (FP + FN) / (TP + TN + FP + FN)
    bacc = (sensitivity + specificity) / 2
    fpr = FP / (FP + TN)  # False Positive Rate
    fnr = FN / (FN + TP)  # False Negative Rate
    brier_score = brier_score_loss(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    
    # Brier Skill Score (BSS)
    baseline_brier_score = brier_score_loss(y_true, np.ones_like(y_true) * y_true.mean())
    bss = 1 - (brier_score / baseline_brier_score) if baseline_brier_score != 0 else 0
    
    # Heidke Skill Score (HSS)
    hss_numerator = 2 * (TP * TN - FP * FN)
    hss_denominator = (TP + FP) * (FP + TN) + (TN + FN) * (FN + TP)
    hss = hss_numerator / hss_denominator if hss_denominator != 0 else 0  # Avoid division by zero
    
    # True Skill Statistics (TSS)
    tss = sensitivity + specificity - 1

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Accuracy': accuracy, 'Sensitivity (TPR)': sensitivity, 'Specificity (TNR)': specificity,
        'Precision': precision, 'F1 Score': f1, 'Error Rate': error_rate,
        'Balanced Accuracy (BACC)': bacc, 'FPR': fpr, 'FNR': fnr,
        'Brier Score (BS)': brier_score, 'Brier Skill Score (BSS)': bss, 'AUC': auc_score,
        'HSS': hss, 'TSS': tss
    }

# Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear', random_state=0, probability=True)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate SVM
svm_metrics = calculate_metrics(y_test, y_pred_svm)
print("SVM Metrics:")
for metric, value in svm_metrics.items():
    print(f"{metric}: {value}")

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate Random Forest
rf_metrics = calculate_metrics(y_test, y_pred_rf)
print("\nRandom Forest Metrics:")
for metric, value in rf_metrics.items():
    print(f"{metric}: {value}")

# K-Fold Cross-Validation for Random Forest
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

kf_metrics_list = []
for train_index, test_index in kf.split(X):
    X_train_kf, X_test_kf = X[train_index], X[test_index]
    y_train_kf, y_test_kf = y[train_index], y[test_index]

    # Feature Scaling
    X_train_kf = sc.fit_transform(X_train_kf)
    X_test_kf = sc.transform(X_test_kf)

    # Label Encoding
    y_train_kf = le.fit_transform(y_train_kf)
    y_test_kf = le.transform(y_test_kf)

    rf_classifier.fit(X_train_kf, y_train_kf)
    y_pred_kf = rf_classifier.predict(X_test_kf)

    metrics_kf = calculate_metrics(y_test_kf, y_pred_kf)
    kf_metrics_list.append(metrics_kf)

# Calculate average metrics across all folds
avg_metrics = {key: np.mean([metrics[key] for metrics in kf_metrics_list]) for key in kf_metrics_list[0]}
print("\nAverage Metrics over 10-Fold Cross-Validation:")
for metric, value in avg_metrics.items():
    print(f"{metric}: {value}")

# LSTM Model with try-except and debugging prints
try:
    max_features = 12000  # Adjust as needed
    max_length = 100  # Adjust based on URL length or padding requirements

    if 'url_data' in locals():  # Check if 'url_data' was stored
        print("Vectorizing 'url' data for LSTM...")
        X_vectorized = vectorizer.fit_transform(url_data).toarray()

        print("Splitting data into train and test sets for LSTM...")
        X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
            X_vectorized, y, test_size=0.25, random_state=0
        )

        # Ensure y_train_lstm and y_test_lstm are numeric
        le_lstm = LabelEncoder()
        y_train_lstm = le_lstm.fit_transform(y_train_lstm)
        y_test_lstm = le_lstm.transform(y_test_lstm)

        print("Padding sequences for LSTM...")
        X_train_lstm = pad_sequences(X_train_lstm, maxlen=max_length)
        X_test_lstm = pad_sequences(X_test_lstm, maxlen=max_length)

        # Convert to float32
        X_train_lstm = X_train_lstm.astype(np.float32)
        X_test_lstm = X_test_lstm.astype(np.float32)

        print("Building LSTM model...")
        lstm_model = Sequential()
        lstm_model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_length))
        lstm_model.add(SpatialDropout1D(0.2))
        lstm_model.add(LSTM(100))
        lstm_model.add(Dense(1, activation='sigmoid'))

        lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("Training LSTM model...")
        lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=64, verbose=2)

        print("Evaluating LSTM model...")
        y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype("int32")
        lstm_metrics = calculate_metrics(y_test_lstm, y_pred_lstm.flatten())
        print("\nLSTM Metrics:")
        for metric, value in lstm_metrics.items():
            print(f"{metric}: {value}")

    else:
        print("LSTM processing skipped as 'url' data is not available.")

except Exception as e:
    print("An error occurred during the LSTM block execution:", e)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    df_cm = pd.DataFrame(cm, columns=np.unique(y_test), index=np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(6, 3))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 10})
    plt.title(title)
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(confusion_matrix(y_test, y_pred_svm), "SVM Confusion Matrix Heat Map")
plot_confusion_matrix(confusion_matrix(y_test, y_pred_rf), "Random Forest Confusion Matrix Heat Map")
plot_confusion_matrix(confusion_matrix(y_test, y_pred_rf), "LSTM Confusion Matrix Heat Map")
