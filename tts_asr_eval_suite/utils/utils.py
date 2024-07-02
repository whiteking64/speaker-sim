import numpy as np


def bootstrap_ci_df(df, column_name, metric_func=np.mean, B=5000, alpha=0.05, precision=2):
    """
    Compute the bootstrap confidence interval for a given metric of a DataFrame column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The column name for which the metric is calculated.
    metric_func (function): The function to compute the metric (default: np.mean).
    B (int): The number of bootstrap samples (default: 5000).
    alpha (float): The significance level for the confidence interval (default: 0.05).

    Returns:
    tuple: The metric of the data and the (lower, upper) bounds of the confidence interval.
    """
    data = df[column_name].values  # Extract the column as a numpy array
    bootstrap_samples = np.random.choice(data, (B, len(data)), replace=True)
    bootstrap_stats = np.array([metric_func(sample) for sample in bootstrap_samples])
    ci_lower = np.percentile(bootstrap_stats, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    mean = metric_func(data)
    ci_upper = ci_upper
    ci_lower = ci_lower

    margin = (ci_upper - ci_lower) / 2 if not np.isnan(ci_lower) else float("nan")
    formatted_result = f"{mean:.{precision}f}% ± {margin:.{precision}f}%"

    return formatted_result
