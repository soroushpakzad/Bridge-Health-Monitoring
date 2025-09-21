## Hi there 👋
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from scipy.stats import genpareto
from sklearn.svm import OneClassSVM
!pip install keras-tuner
import kerastuner as kt  # make sure to install keras-tuner (pip install keras-tuner)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# STEP 1: Load and Split the Dataset
# =============================================================================
# Load the full training data (Normal Condition)
X_train_full = pd.read_excel(io="Freq_Z24N.xlsx", sheet_name="Train Set", header=0, engine="openpyxl").values

# Split the training set into training (80%) and validation (20%) sets
X_train, X_val = train_test_split(X_train_full, test_size=0.2, random_state=42)

# Load the test set, which remains unseen. It contains two parts:
#   - first part: Inspection - Normal Condition (samples 1 to 864)
#   - second part: Inspection - Damaged Condition (samples 865 to end)
X_test_full = pd.read_excel(io="Freq_Z24N.xlsx", sheet_name="Test Set", header=0, engine="openpyxl").values
X_test_normal = X_test_full[:864]    # Inspection - Normal Condition
X_test_damaged = X_test_full[864:]     # Inspection - Damaged Condition

# Standardize features using training data statistics (from X_train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_normal_scaled = scaler.transform(X_test_normal)
X_test_damaged_scaled = scaler.transform(X_test_damaged)

input_dim = X_train_scaled.shape[1]

# =============================================================================
# STEP 2: Define a Model Builder for Hyperparameter Tuning (VAE)
# =============================================================================
def build_vae(hp):
    # Choose hyperparameters
    latent_dim = hp.Choice('latent_dim', values=[8, 16, 32])
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    l2_reg = hp.Float('l2_reg', min_value=1e-4, max_value=1e-2, sampling='LOG')
    beta = hp.Float('beta', min_value=0.05, max_value=0.5, step=0.05)

    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    # "z_mean" will be used later as the latent representation.
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

    # Decoder
    decoder_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x_dec = layers.Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(decoder_inputs)
    x_dec = layers.BatchNormalization()(x_dec)
    x_dec = layers.Dropout(dropout_rate)(x_dec)
    x_dec = layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x_dec)
    x_dec = layers.BatchNormalization()(x_dec)
    x_dec = layers.Dropout(dropout_rate)(x_dec)
    x_dec = layers.Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x_dec)
    x_dec = layers.BatchNormalization()(x_dec)
    x_dec = layers.Dropout(dropout_rate)(x_dec)
    decoder_outputs = layers.Dense(input_dim, activation='linear')(x_dec)

    # Build decoder model (used for prediction later)
    decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

    # Custom loss layer
    class VAELossLayer(layers.Layer):
        def __init__(self, beta=beta, **kwargs):
            super().__init__(**kwargs)
            self.beta = beta

        def call(self, inputs):
            x, x_decoded, z_mean, z_log_var = inputs
            reconstruction_loss = tf.reduce_mean(tf.square(x - x_decoded), axis=1) * input_dim
            kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            total_loss = tf.reduce_mean(reconstruction_loss + self.beta * kl_loss)
            self.add_loss(total_loss)
            return x_decoded

    vae_outputs = decoder(z)
    vae_loss_output = VAELossLayer(name='vae_loss')([encoder_inputs, vae_outputs, z_mean, z_log_var])
    vae = Model(encoder_inputs, vae_loss_output, name="vae")
    vae.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=hp.Float('learning_rate', 1e-4, 1e-3, sampling='LOG')
    ))
    return vae

# =============================================================================
# STEP 3: Hyperparameter Tuning Using Keras Tuner (VAE)
# =============================================================================
tuner = kt.RandomSearch(
    build_vae,
    objective='val_loss',
    max_trials=10,  # Adjust the number of trials as needed
    executions_per_trial=1,
    directory='vae_tuner_dir',
    project_name='anomaly_detection_tuning'
)

# Use the 20% validation split (from the training set) for tuning
tuner.search(X_train_scaled, X_train_scaled,
             epochs=100,
             batch_size=64,
             validation_data=(X_val_scaled, X_val_scaled),
             callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.001)],
             verbose=1)

# Retrieve the best model and hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
print(f"  latent_dim: {best_hps.get('latent_dim')}")
print(f"  dropout_rate: {best_hps.get('dropout_rate')}")
print(f"  l2_reg: {best_hps.get('l2_reg')}")
print(f"  beta: {best_hps.get('beta')}")
print(f"  learning_rate: {best_hps.get('learning_rate')}")

# Build the VAE model with best hyperparameters and train it
best_model = build_vae(best_hps)
early_stop = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True, min_delta=0.001)
history = best_model.fit(X_train_scaled, X_train_scaled,
                         epochs=500,
                         batch_size=64,
                         validation_data=(X_val_scaled, X_val_scaled),
                         callbacks=[early_stop],
                         verbose=1)

# =============================================================================
# STEP 4: Compute VAE Reconstruction Errors (for reference)
# =============================================================================
train_recon = best_model.predict(X_train_scaled)
train_errors = np.mean(np.square(X_train_scaled - train_recon), axis=1)

val_recon = best_model.predict(X_val_scaled)
val_errors = np.mean(np.square(X_val_scaled - val_recon), axis=1)

test_normal_recon = best_model.predict(X_test_normal_scaled)
test_normal_errors = np.mean(np.square(X_test_normal_scaled - test_normal_recon), axis=1)

test_damaged_recon = best_model.predict(X_test_damaged_scaled)
test_damaged_errors = np.mean(np.square(X_test_damaged_scaled - test_damaged_recon), axis=1)

# =============================================================================
# STEP 4.5: Extract Latent Representations Using the Trained VAE Encoder
# =============================================================================
encoder = Model(inputs=best_model.input, outputs=best_model.get_layer("z_mean").output)

latent_train = encoder.predict(X_train_scaled)
latent_val = encoder.predict(X_val_scaled)
latent_test_normal = encoder.predict(X_test_normal_scaled)
latent_test_damaged = encoder.predict(X_test_damaged_scaled)

# =============================================================================
# STEP 5: One-Class SVM Model on the Latent Representations
# =============================================================================
param_grid = {
    'nu': [0.01, 0.05, 0.1],
    'gamma': ['scale', 0.01, 0.001]
}

best_ocsvm = None
best_outlier_fraction = np.inf
best_params = {}

for nu in param_grid['nu']:
    for gamma in param_grid['gamma']:
        ocsvm = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
        ocsvm.fit(latent_train)
        preds = ocsvm.predict(latent_train)  # +1 for inliers, -1 for outliers
        outlier_fraction = np.mean(preds == -1)
        if outlier_fraction < best_outlier_fraction:
            best_outlier_fraction = outlier_fraction
            best_params = {'nu': nu, 'gamma': gamma}
            best_ocsvm = ocsvm

print("Best One-Class SVM parameters found:")
print(best_params)
print(f"Fraction of outliers on training data: {best_outlier_fraction:.4f}")

ocsvm_model = OneClassSVM(nu=best_params['nu'], kernel='rbf', gamma=best_params['gamma'])
ocsvm_model.fit(latent_train)

# =============================================================================
# STEP 6: Compute Anomaly Scores Using the One-Class SVM
# =============================================================================
train_scores = ocsvm_model.decision_function(latent_train)
val_scores = ocsvm_model.decision_function(latent_val)
test_normal_scores = ocsvm_model.decision_function(latent_test_normal)
test_damaged_scores = ocsvm_model.decision_function(latent_test_damaged)

# =============================================================================
# STEP 6.5: Compute Composite Anomaly Scores by Fusing VAE Error and OCSVM Score
# =============================================================================
# Normalize VAE reconstruction errors using training data stats
min_train_vae = np.min(train_errors)
max_train_vae = np.max(train_errors)
norm_train_vae = (train_errors - min_train_vae) / (max_train_vae - min_train_vae)
norm_val_vae = (val_errors - min_train_vae) / (max_train_vae - min_train_vae)
norm_test_normal_vae = (test_normal_errors - min_train_vae) / (max_train_vae - min_train_vae)
norm_test_damaged_vae = (test_damaged_errors - min_train_vae) / (max_train_vae - min_train_vae)

# Convert OCSVM scores to anomaly measures (higher anomaly = more positive)
train_ocsvm_anomaly = -train_scores
val_ocsvm_anomaly = -val_scores
test_normal_ocsvm_anomaly = -test_normal_scores
test_damaged_ocsvm_anomaly = -test_damaged_scores

min_train_ocsvm = np.min(train_ocsvm_anomaly)
max_train_ocsvm = np.max(train_ocsvm_anomaly)
norm_train_ocsvm = (train_ocsvm_anomaly - min_train_ocsvm) / (max_train_ocsvm - min_train_ocsvm)
norm_val_ocsvm = (val_ocsvm_anomaly - min_train_ocsvm) / (max_train_ocsvm - min_train_ocsvm)
norm_test_normal_ocsvm = (test_normal_ocsvm_anomaly - min_train_ocsvm) / (max_train_ocsvm - min_train_ocsvm)
norm_test_damaged_ocsvm = (test_damaged_ocsvm_anomaly - min_train_ocsvm) / (max_train_ocsvm - min_train_ocsvm)

# =============================================================================
# STEP 7: Jointly Tune the Fusion Weight and EVT Tail Probability
# =============================================================================
# First compute composite scores for the training set (using a provisional fusion weight)
# that will be used to fit the EVT model.
# (We use the same fusion formula here as later.)
composite_train = norm_train_vae * 0.5 + norm_train_ocsvm * 0.5

# Define candidate fusion weights and candidate tail probabilities.
alpha_candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
tail_prob_candidates = [0.90, 0.95, 0.99]

best_total_error = np.inf
best_alpha = None
best_tail_prob = None

# For each candidate, compute the composite scores on the test sets and then the EVT threshold.
# Evaluate by counting false positives (on Inspection-Normal) and false negatives (on Inspection-Damaged).
for candidate_alpha in alpha_candidates:
    # Compute composite scores for test sets using the candidate fusion weight.
    comp_test_normal_candidate = candidate_alpha * norm_test_normal_vae + (1 - candidate_alpha) * norm_test_normal_ocsvm
    comp_test_damaged_candidate = candidate_alpha * norm_test_damaged_vae + (1 - candidate_alpha) * norm_test_damaged_ocsvm
    # We use the training composite scores (with provisional fusion weight) to compute u and fit EVT.
    u = np.percentile(composite_train, 90)
    excesses = composite_train[composite_train > u] - u
    params = genpareto.fit(excesses)
    for candidate_tail in tail_prob_candidates:
         # EVT threshold using the candidate tail probability.
         threshold_candidate = u + genpareto.ppf(candidate_tail, *params)
         # Compute misclassification errors on test sets:
         # False positives: Inspection-Normal samples with composite score >= threshold.
         # False negatives: Inspection-Damaged samples with composite score < threshold.
         false_positives = np.sum(comp_test_normal_candidate >= threshold_candidate)
         false_negatives = np.sum(comp_test_damaged_candidate < threshold_candidate)
         total_error = false_positives + false_negatives
         if total_error < best_total_error:
              best_total_error = total_error
              best_alpha = candidate_alpha
              best_tail_prob = candidate_tail

print("Best joint tuning parameters found:")
print(f"  Fusion weight (alpha): {best_alpha}")
print(f"  EVT Tail Probability: {best_tail_prob}")
print(f"  Total misclassification error (FP + FN): {best_total_error}")

# Now use the best parameters to compute final composite scores.
composite_train = best_alpha * norm_train_vae + (1 - best_alpha) * norm_train_ocsvm
composite_val   = best_alpha * norm_val_vae   + (1 - best_alpha) * norm_val_ocsvm
composite_test_normal = best_alpha * norm_test_normal_vae + (1 - best_alpha) * norm_test_normal_ocsvm
composite_test_damaged = best_alpha * norm_test_damaged_vae + (1 - best_alpha) * norm_test_damaged_ocsvm

# Compute EVT threshold on training composite scores using best_tail_prob.
u = np.percentile(composite_train, 90)
excesses = composite_train[composite_train > u] - u
params = genpareto.fit(excesses)
threshold_evt = u + genpareto.ppf(best_tail_prob, *params)
print(f"Final EVT-calibrated composite threshold: {threshold_evt:.4f}")

# =============================================================================
# STEP 8: Combine Composite Anomaly Scores into a Single DataFrame
# =============================================================================
df_train = pd.DataFrame({
    "Composite_Anomaly_Score": composite_train,
    "Dataset": "Training - Normal"
})
df_val = pd.DataFrame({
    "Composite_Anomaly_Score": composite_val,
    "Dataset": "Validation - Normal"
})
df_test_normal = pd.DataFrame({
    "Composite_Anomaly_Score": composite_test_normal,
    "Dataset": "Inspection - Normal"
})
df_test_damaged = pd.DataFrame({
    "Composite_Anomaly_Score": composite_test_damaged,
    "Dataset": "Inspection - Damaged"
})
all_scores_df = pd.concat([df_train, df_val, df_test_normal, df_test_damaged], ignore_index=True)

summary_df = pd.DataFrame({
    "Mean_Train_Composite_Score": [np.mean(composite_train)],
    "Std_Train_Composite_Score": [np.std(composite_train)],
    "Mean_Val_Composite_Score": [np.mean(composite_val)],
    "Std_Val_Composite_Score": [np.std(composite_val)],
    "Mean_Test_Normal_Composite_Score": [np.mean(composite_test_normal)],
    "Std_Test_Normal_Composite_Score": [np.std(composite_test_normal)],
    "Mean_Test_Damaged_Composite_Score": [np.mean(composite_test_damaged)],
    "Std_Test_Damaged_Composite_Score": [np.std(composite_test_damaged)]
})

# =============================================================================
# STEP 9: Save Results to an Excel File with Two Sheets
# =============================================================================
output_excel_file = "AnomalyDetection_Tuned_Results.xlsx"
with pd.ExcelWriter(output_excel_file) as writer:
    all_scores_df.to_excel(writer, sheet_name="Composite_Anomaly_Scores", index=False)
    summary_df.to_excel(writer, sheet_name="Composite_Score_Summary", index=False)

print(f"Results saved to '{output_excel_file}'.")

# =============================================================================
# STEP 10: Visualization (Optional)
# =============================================================================
plt.figure(figsize=(12, 6))
plt.scatter(range(len(composite_train)), composite_train, s=3, label="Training - Normal")
plt.scatter(len(composite_train) + np.arange(len(composite_val)), composite_val,
            s=3, c='green', label="Validation - Normal")
plt.scatter(len(composite_train) + len(composite_val) + np.arange(len(composite_test_normal)),
            composite_test_normal, s=3, c='orange', label="Inspection - Normal")
plt.scatter(len(composite_train) + len(composite_val) + len(composite_test_normal) + np.arange(len(composite_test_damaged)),
            composite_test_damaged, s=3, c='red', label="Inspection - Damaged")
# Plot the EVT-calibrated threshold.
plt.axhline(threshold_evt, c='black', linestyle='--', label="EVT Threshold")
plt.ylabel("Composite Anomaly Score")
plt.xlabel("Sample Index")
plt.title("Anomaly Detection Performance (Fusion: VAE + One-Class SVM)")
plt.legend()
plt.show()
<!--
**soroushpakzad/soroushpakzad** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
