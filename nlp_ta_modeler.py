# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

from sklearn.model_selection import cross_val_score


# Model Evaluation with CV
def evaluate_model_cv(X, y, model, cv_folds=5, key=None):
    print("Evaluating model with cv.")
    cv_scores = cross_val_score(model, X, y, cv=cv_folds)
    mean_cv_accuracy = cv_scores.mean()
    print(f"Mean CV Accuracy for {key}: {mean_cv_accuracy * 100:.2f}%")

# Split data into train and test data
def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Training model
def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

# Model Evaluation with test set
def evaluate_model(X_test, y_test, model, key=None):
    print("Evaluating model.")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy for {key}: {accuracy * 100:.2f}%")
    return y_pred, accuracy

# Calculate evaluation metrics
def compute_metrics(y_true, y_pred, model_key):
    cls_report=classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"""
        For {model_key}
        Classification report  \n {cls_report} 
        Confusion Matrix \n {cm}
        """)

# Plot confusion matrix
def plot_confusionmatrix(y_true, y_pred, model_key):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix by row 
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Use ConfusionMatrixDisplay to plot the normalized confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=np.unique(y_true))
    
    disp.plot(include_values=True, cmap='viridis', ax=ax, xticks_rotation='horizontal',
              values_format=".2%")  
    
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f"Confusion Matrix for {model_key} (Percentages)")
    plt.show()

