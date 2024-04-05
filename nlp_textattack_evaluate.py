
# Import libraries

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb

from nlp_ta_handler import load_pkl_file, extract_features
from nlp_ta_modeler import split_train_test, train_model, evaluate_model_cv, evaluate_model, compute_metrics, \
    plot_confusionmatrix




# Load EDA augmented dataset
augmented_data_eda= load_pkl_file('./data/augmented_data_eda.pkl')
augmented_label_eda = load_pkl_file('./data/augmented_label_eda.pkl')

# Load CKL augmented dataset
augmented_data_ckl= load_pkl_file('./data/augmented_data_ckl.pkl')
augmented_label_ckl= load_pkl_file('./data/augmented_label_ckl.pkl')

# Load CA augmented dataset
augmented_data_ca= load_pkl_file('./data/augmented_data_ca.pkl')
augmented_label_ca= load_pkl_file('./data/augmented_label_ca.pkl')

# Feature Extraction for EDA
X_train_counts_eda = extract_features(augmented_data_eda)
print(X_train_counts_eda.shape)

# Feature Extraction for CKL
X_train_counts_ckl = extract_features(augmented_data_ckl)
print(X_train_counts_ckl.shape)

# Feature Extraction for CA
X_train_counts_ca = extract_features(augmented_data_ca)
print(X_train_counts_ca.shape)

# Split EDA augmented dataset
X_train_eda, X_test_eda, y_train_eda, y_test_eda = split_train_test(X_train_counts_eda, augmented_label_eda)

# Split CKL augmented dataset
X_train_ckl, X_test_ckl, y_train_ckl, y_test_ckl = split_train_test(X_train_counts_ckl, augmented_label_ckl)

# Split CA augmented dataset
X_train_ca, X_test_ca, y_train_ca, y_test_ca = split_train_test(X_train_counts_ca, augmented_label_ca)


# Define Ridge classifier
ridclf = RidgeClassifier(alpha=1.0, solver="sparse_cg")

# Define MNK classifier
mnbclf = MultinomialNB(alpha=1)

# Define RF classifier
rfclf = RandomForestClassifier(n_estimators=100)  

# Define XGBoost classifier
xgbclf = xgb.XGBClassifier(n_estimators=100)


# Train model for EDA
model_eda_ridclf= train_model(X_train_eda, y_train_eda, ridclf)
model_eda_mnbclf= train_model(X_train_eda, y_train_eda, mnbclf)
model_eda_rfclf= train_model(X_train_eda, y_train_eda, rfclf)
model_eda_xgbclf= train_model(X_train_eda, y_train_eda, xgbclf)


# Model Evaluation with cv for EDA
evaluate_model_cv(X_train_counts_eda, augmented_label_eda, ridclf, key="EDA_RID")
evaluate_model_cv(X_train_counts_eda, augmented_label_eda, mnbclf, key="EDA_MNB")
evaluate_model_cv(X_train_counts_eda, augmented_label_eda, rfclf, key="EDA_RF")
evaluate_model_cv(X_train_counts_eda, augmented_label_eda, xgbclf, key="EDA_XGB")


# Model Evaluation with test data for EDA
y_pred_eda_rid, _ = evaluate_model(X_test_eda, y_test_eda, model_eda_ridclf, key="EDA_RID")
y_pred_eda_mnb, _ = evaluate_model(X_test_eda, y_test_eda, model_eda_mnbclf, key="EDA_MNB")
y_pred_eda_rf, _ = evaluate_model(X_test_eda, y_test_eda, model_eda_rfclf, key="EDA_RF")
y_pred_eda_xgb, _ = evaluate_model(X_test_eda, y_test_eda, model_eda_xgbclf, key="EDA_XGB")


compute_metrics(y_test_eda, y_pred_eda_rid, model_key="EDA_RID")
plot_confusionmatrix(y_test_eda, y_pred_eda_rid, model_key="EDA_RID")

compute_metrics(y_test_eda, y_pred_eda_mnb, model_key="EDA_MNB")
plot_confusionmatrix(y_test_eda, y_pred_eda_mnb, model_key="EDA_MNB")

compute_metrics(y_test_eda, y_pred_eda_rf, model_key="EDA_RF")
plot_confusionmatrix(y_test_eda, y_pred_eda_rf, model_key="EDA_RF")

compute_metrics(y_test_eda, y_pred_eda_xgb, model_key="EDA_XGB")
plot_confusionmatrix(y_test_eda, y_pred_eda_xgb, model_key="EDA_XGB")


# Train model for CKL
model_ckl_ridclf= train_model(X_train_ckl, y_train_ckl, ridclf)
model_ckl_mnbclf= train_model(X_train_ckl, y_train_ckl, mnbclf)
model_ckl_rfclf= train_model(X_train_ckl, y_train_ckl, rfclf)
model_ckl_xgbclf= train_model(X_train_ckl, y_train_ckl, xgbclf)


# Model Evaluation with cv for CKL
evaluate_model_cv(X_train_counts_ckl, augmented_label_ckl, ridclf, key="CKL_RID")
evaluate_model_cv(X_train_counts_ckl, augmented_label_ckl, mnbclf, key="CKL_MNB")
evaluate_model_cv(X_train_counts_ckl, augmented_label_ckl, rfclf, key="CKL_RF")
evaluate_model_cv(X_train_counts_ckl, augmented_label_ckl, xgbclf, key="CKL_XGB")

# Model Evaluation with test data for CKL
y_pred_ckl_rid, _ = evaluate_model(X_test_ckl, y_test_ckl, model_ckl_ridclf, key="CKL_RID")
y_pred_ckl_mnb, _ = evaluate_model(X_test_ckl, y_test_ckl, model_ckl_mnbclf, key="CKL_MNB")
y_pred_ckl_rf, _ = evaluate_model(X_test_ckl, y_test_ckl, model_ckl_rfclf, key="CKL_RF")
y_pred_ckl_xgb, _ = evaluate_model(X_test_ckl, y_test_ckl, model_ckl_xgbclf, key="CKL_XGB")


compute_metrics(y_test_ckl, y_pred_ckl_rid, model_key="CKL_RID")
plot_confusionmatrix(y_test_ckl, y_pred_ckl_rid, model_key="CKL_RID")

compute_metrics(y_test_ckl, y_pred_ckl_mnb, model_key="CKL_MNB")
plot_confusionmatrix(y_test_ckl, y_pred_ckl_mnb, model_key="CKL_MNB")

compute_metrics(y_test_ckl, y_pred_ckl_rf, model_key="CKL_RF")
plot_confusionmatrix(y_test_ckl, y_pred_ckl_rf, model_key="CKL_RF")

compute_metrics(y_test_ckl, y_pred_ckl_xgb, model_key="CKL_XGB")
plot_confusionmatrix(y_test_ckl, y_pred_ckl_xgb, model_key="CKL_XGB")


# Train model for CA
model_ca_ridclf= train_model(X_train_ca, y_train_ca, ridclf)
model_ca_mnbclf= train_model(X_train_ca, y_train_ca, mnbclf)
model_ca_rfclf= train_model(X_train_ca, y_train_ca, rfclf)
model_ca_xgbclf= train_model(X_train_ca, y_train_ca, xgbclf)


# Model Evaluation with cv for CA
evaluate_model_cv(X_train_counts_ca, augmented_label_ca, ridclf, key="CA_RID")
evaluate_model_cv(X_train_counts_ca, augmented_label_ca, mnbclf, key="CA_MNB")
evaluate_model_cv(X_train_counts_ca, augmented_label_ca, rfclf, key="CA_RF")
evaluate_model_cv(X_train_counts_ca, augmented_label_ca, xgbclf, key="CA_XGB")


# Model Evaluation with test data for CA
y_pred_ca_rid, _ = evaluate_model(X_test_ca, y_test_ca, model_ca_ridclf, key="CA_RID")
y_pred_ca_mnb, _ = evaluate_model(X_test_ca, y_test_ca, model_ca_mnbclf, key="CA_MNB")
y_pred_ca_rf, _ = evaluate_model(X_test_ca, y_test_ca, model_ca_rfclf, key="CA_RF")
y_pred_ca_xgb, _ = evaluate_model(X_test_ca, y_test_ca, model_ca_xgbclf, key="CA_XGB")


compute_metrics(y_test_ca, y_pred_ca_rid, model_key="CA_RID")
plot_confusionmatrix(y_test_ca, y_pred_ca_rid, model_key="CA_RID")

compute_metrics(y_test_ca, y_pred_ca_mnb, model_key="CA_MNB")
plot_confusionmatrix(y_test_ca, y_pred_ca_mnb, model_key="CA_MNB")

compute_metrics(y_test_ca, y_pred_ca_rf, model_key="CA_RF")
plot_confusionmatrix(y_test_ca, y_pred_ca_rf, model_key="CA_RF")

compute_metrics(y_test_ca, y_pred_ca_xgb, model_key="CA_XGB")
plot_confusionmatrix(y_test_ca, y_pred_ca_xgb, model_key="CA_XGB")









