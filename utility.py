import torch
import numpy as np
from sklearn.metrics import r2_score

def print_environment():
    torch.set_float32_matmul_precision('high') # “highest” (default, float32), “high” (TensorFloat32), or “medium” (bfloat16)
    print('torch.get_float32_matmul_precision(): {}'.format(torch.get_float32_matmul_precision()))
    print('torch.backends.cuda.matmul.allow_tf32: {}'.format(torch.backends.cuda.matmul.allow_tf32))
    print('Torch version: {}'.format(torch.__version__))
    print('CUDA available: {}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print('torch.cuda.device_count(): {}'.format(torch.cuda.device_count()))
        print('torch.cuda.current_device(): {}'.format(torch.cuda.current_device()))
        print('torch.cuda.device(0): {}'.format(torch.cuda.device(0)))
        print('torch.cuda.get_device_name(0): {}'.format(torch.cuda.get_device_name(0)))

# -----------------------------
# Individual Metric Functions
# -----------------------------

def mean_bias_error(df, prediction_col='Predictions', label_col='Labels'):
    """
    Compute the Mean Bias Error (MBE), which is the average of (prediction - label).
    Positive MBE indicates systematic overprediction; negative indicates underprediction.
    """
    error = df[prediction_col] - df[label_col]
    return error.mean()

def median_absolute_error(df, prediction_col='Predictions', label_col='Labels'):
    """
    Compute the Median Absolute Error (MedAE) of predictions.
    """
    abs_error = (df[prediction_col] - df[label_col]).abs()
    return abs_error.median()

def r2_metric(df, prediction_col='Predictions', label_col='Labels'):
    """
    Compute the R² Score, which measures how well the predictions explain the variance in the data.
    """
    return r2_score(df[label_col], df[prediction_col])

def hit_rate_within_tolerance(df, prediction_col='Predictions', label_col='Labels', tolerance=0.10):
    """
    Compute the percentage of predictions within a certain relative tolerance (default ±10%) of the true values.
    """
    relative_error = ((df[prediction_col] - df[label_col]).abs() / df[label_col])
    return (relative_error <= tolerance).mean() * 100  # percentage

def quantile_overprediction_error(df, prediction_col='Predictions', label_col='Labels', quantile=0.95):
    """
    Compute the specified quantile (default 95th percentile) of the overprediction error.
    Only considers cases where prediction > label.
    """
    over_errors = (df[prediction_col] - df[label_col]).clip(lower=0)
    if (df[prediction_col] > df[label_col]).sum() == 0:
        return float('nan')
    return over_errors[over_errors > 0].quantile(quantile)

def mse_underpredictions(df, prediction_col='Predictions', label_col='Labels'):
    """
    Compute the Mean Squared Error (MSE) for underpredictions only (when prediction < label).
    Returns NaN if there are no underpredictions.
    """
    mask = df[prediction_col] < df[label_col]
    if mask.sum() == 0:
        return float('nan')
    under_sq_errors = (df[label_col] - df[prediction_col])[mask].pow(2)
    return under_sq_errors.mean()

def normalized_overprediction_cost(df, prediction_col='Predictions', label_col='Labels', beta=5.0):
    """
    Compute a normalized cost metric for overpredictions.
    The total cost (beta * overprediction error) is normalized by the sum of true labels.
    """
    error = df[prediction_col] - df[label_col]
    over_cost = beta * error.clip(lower=0)
    return over_cost.sum() / df[label_col].sum()

def mse_overpredictions_all_rows(df, prediction_col='Predictions', label_col='Labels'):
    """
    Compute the Mean Squared Error (MSE) for overpredictions, averaging over all rows.
    Overpredictions are defined as (prediction - label) if positive, otherwise 0.
    """
    squared_over_errors = (df[prediction_col] - df[label_col]).clip(lower=0).pow(2)
    return squared_over_errors.mean()

def mse_overpredictions_positive_only(df, prediction_col='Predictions', label_col='Labels'):
    """
    Compute the Mean Squared Error (MSE) for overpredictions only on rows where prediction > label.
    Returns NaN if there are no overpredictions.
    """
    mask = df[prediction_col] > df[label_col]
    if mask.sum() == 0:
        return float('nan')
    squared_over_errors = (df[prediction_col] - df[label_col])[mask].pow(2)
    return squared_over_errors.mean()

def smart_provision_metric(df, prediction_col='Predictions', label_col='Labels', alpha=0.5, beta=5.0, epsilon_beta=0.1):
    """
    Compute the Smart Provision loss as an evaluation metric.
    
    For each row:
      - If (prediction - label) < -epsilon_beta, a mild linear penalty is applied: alpha * |error|.
      - If (prediction - label) > 0, a strong penalty is applied: beta * error.
      - Otherwise, no penalty is applied.
    
    The final metric is the mean loss over all rows.
    """
    error = df[prediction_col] - df[label_col]
    under_loss = np.where(error < -epsilon_beta, alpha * np.abs(error), 0)
    over_loss = np.where(error > 0, beta * error, 0)
    total_loss = under_loss + over_loss
    return total_loss.mean()

def utilization_rate(df, prediction_col='Predictions', label_col='Labels'):
    """
    Compute the Utilization Rate as the ratio of the average prediction to the average label,
    expressed as a percentage.
    """
    return (df[prediction_col].mean() / df[label_col].mean()) * 100

def overprediction_percentage(df, prediction_col='Predictions', label_col='Labels'):
    """
    Compute the percentage of predictions that exceed the true labels.
    """
    overprediction_count = np.sum(df[prediction_col] > df[label_col])
    total_predictions = len(df)
    return (overprediction_count / total_predictions) * 100

def area_of_violation(df, pred='Predictions', label='Labels', dt=1.0):
    # dt = time‑step length in seconds
    return ((df[pred] - df[label]).clip(lower=0) * dt).sum()

def cvf(df, pred='Predictions', label='Labels'):
    return ( (df[pred] > df[label]).mean() * 100 )

def burst_aware_aov(df, pred='Predictions', label='Labels', p=2, dt=1.0):
    """
    p : exponent >1  (2 = quadratic, 3 = cubic …)
    dt: sample period in seconds
    """
    err = (df[pred] - df[label]).clip(lower=0)
    return (err.pow(p) * dt).sum()

def burst_severity_index(df, pred, label, p=2, dt=1.0):
    aov_p = burst_aware_aov(df, pred, label, p, dt)
    duration = len(df) * dt
    # convert back to Mbps by taking the p‑root of the mean
    return (aov_p / duration) ** (1/p)

# -----------------------------
# Wrapper Function to Evaluate All Metrics
# -----------------------------

def evaluate_model_metrics(df, prediction_col='Predictions', label_col='Labels'):
    """
    Evaluate and print multiple metrics for the model's predictions.
    """
    metrics = {}
    
    # Basic error metrics
    metrics['Mean Bias Error (MBE)'] = mean_bias_error(df, prediction_col, label_col)
    metrics['Median Absolute Error (MedAE)'] = median_absolute_error(df, prediction_col, label_col)
    metrics['R² Score'] = r2_metric(df, prediction_col, label_col)
    metrics['Hit Rate within 10% Tolerance (%)'] = hit_rate_within_tolerance(df, prediction_col, label_col, tolerance=0.10)
    
    # Overprediction-specific metrics
    metrics['95th Percentile Overprediction Error'] = quantile_overprediction_error(df, prediction_col, label_col, quantile=0.95)
    metrics['MSE Overpredictions (all rows)'] = mse_overpredictions_all_rows(df, prediction_col, label_col)
    metrics['MSE Overpredictions (positive only)'] = mse_overpredictions_positive_only(df, prediction_col, label_col)
    
    # Underprediction metric
    metrics['MSE of Underpredictions'] = mse_underpredictions(df, prediction_col, label_col)
    
    # Cost-based metric
    metrics['Normalized Overprediction Cost'] = normalized_overprediction_cost(df, prediction_col, label_col, beta=5.0)
    
    # Custom smart provision loss metric
    metrics['Smart Provision Evaluation Metric (mean loss)'] = smart_provision_metric(df, prediction_col, label_col, alpha=0.5, beta=5.0, epsilon_beta=0.1)

    metrics['Utilization Rate (%)'] = utilization_rate(df, prediction_col, label_col)
    metrics['Percentage of Overprediction (%)'] = overprediction_percentage(df, prediction_col, label_col)

    metrics['Area of Violation (Mbit·s)']     = area_of_violation(df, prediction_col, label_col, 1.0)
    metrics['CVF - Capacity Violation %']     = cvf(df, prediction_col, label_col)

    metrics['Burst-Aware Area of Violation (Mbit^2·s) - Quadratic']     = burst_aware_aov(df, prediction_col, label_col, p=2, dt=1.0)
    metrics['SPIKE2 score - 	(Normalized) Mbps - Quadratic']     = burst_severity_index(df, prediction_col, label_col, p=2, dt=1.0)

    metrics['Burst-Aware Area of Violation (Mbit^3·s) - Cubic']     = burst_aware_aov(df, prediction_col, label_col, p=3, dt=1.0)
    metrics['SPIKE3 score - 	(Normalized) Mbps - Cubic']     = burst_severity_index(df, prediction_col, label_col, p=3, dt=1.0)

    util_min = 73.8196
    util_max = 81.2383
    vov_min = 106.2393
    vov_max = 150.3802

    util_rate = utilization_rate(df, prediction_col, label_col)
    VoV = burst_aware_aov(df, prediction_col, label_col, p=2, dt=1.0)
    print(f'Utilization Rate: {util_rate}, VoV: {VoV}')

    u_norm = (util_rate - util_min ) / (util_max - util_min)
    inv_a_norm = (1 - (VoV - vov_min) / (vov_max - vov_min))
    print(f'u_norm: {u_norm}, inv_a_norm: {inv_a_norm}')
    efficiency_index = 0.5 * u_norm + 0.5 * inv_a_norm
    metrics['Efficiency Index'] = efficiency_index
    print(f'Efficiency Index: {efficiency_index}')

    # Print out all metrics
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics