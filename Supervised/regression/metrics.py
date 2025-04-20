def evaluate_model(y_true, y_pred):
    """
    Calcule les métriques : MSE, RMSE, MAE, MAPE, R²
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Les longueurs de y_true et y_pred ne correspondent pas")

    errors = [yt - yp for yt, yp in zip(y_true, y_pred)]
    abs_errors = [abs(e) for e in errors]
    squared_errors = [e**2 for e in errors]

    mse = sum(squared_errors) / len(squared_errors)
    rmse = mse ** 0.5
    mae = sum(abs_errors) / len(abs_errors)
    mape = sum(abs_errors[i] / y_true[i] if y_true[i] != 0 else 0 for i in range(len(y_true))) * 100 / len(y_true)

    y_mean = sum(y_true) / len(y_true)
    ss_total = sum((yt - y_mean) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R²": r2
    }
