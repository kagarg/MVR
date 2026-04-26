from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import bartlett, levene, pearsonr, ttest_ind
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss


PLOTS_DIR = Path("plots")
REPORT_PATH = Path("report.txt")


@dataclass
class TrendModel:
    name: str
    params: np.ndarray
    pvalues: np.ndarray
    aic: float
    bic: float
    rss: float
    predict: Callable[[np.ndarray], np.ndarray]
    formula: str


def load_v3_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, sep=";", quotechar='"')
    if "V3" not in df.columns:
        raise ValueError("В CSV не найден столбец 'V3'.")

    v3 = pd.to_numeric(df["V3"], errors="coerce")
    if v3.isna().any():
        v3 = v3.interpolate().bfill().ffill()

    dt_index = pd.date_range("2012-01-01 00:00:00", periods=len(v3), freq="h")
    return pd.Series(v3.values, index=dt_index, name="V3")


def foster_stuart_stats(series: pd.Series) -> tuple[float, float, float]:
    x = series.values
    if len(x) < 3:
        return np.nan, np.nan, np.nan

    max_so_far = x[0]
    min_so_far = x[0]
    s_vals = []
    d_vals = []

    for xi in x[1:]:
        u = 1 if xi > max_so_far else 0
        l = 1 if xi < min_so_far else 0
        s_vals.append(u - l)
        d_vals.append(u + l)
        max_so_far = max(max_so_far, xi)
        min_so_far = min(min_so_far, xi)

    s_stat = float(np.sum(s_vals))
    d_stat = float(np.sum(d_vals))
    z_approx = s_stat / np.sqrt(d_stat + 1e-9)
    return s_stat, d_stat, z_approx


def stationarity_tests(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    values = clean.values

    runs_z, runs_p = runstest_1samp(values, cutoff="median")

    half = len(values) // 2
    first_half = values[:half]
    second_half = values[half:]

    t_stat, t_p = ttest_ind(first_half, second_half, equal_var=False)
    lev_stat, lev_p = levene(first_half, second_half)

    chunks = [chunk.astype(float) for chunk in np.array_split(values, 4) if len(chunk) > 1]
    if len(chunks) >= 2:
        bart_stat, bart_p = bartlett(*chunks)
    else:
        bart_stat, bart_p = np.nan, np.nan

    fs_s, fs_d, fs_z = foster_stuart_stats(clean)

    try:
        adf_p = adfuller(values, autolag="AIC")[1]
    except Exception:
        adf_p = np.nan

    try:
        # У KPSS p-value табличный и может быть усечен на границах [0.01, 0.10].
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_p = kpss(values, regression="c", nlags="auto")[1]
    except Exception:
        kpss_p = np.nan

    return {
        "runs_z": float(runs_z),
        "runs_p": float(runs_p),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_p),
        "bartlett_stat": float(bart_stat),
        "bartlett_p": float(bart_p),
        "foster_stuart_s": float(fs_s),
        "foster_stuart_d": float(fs_d),
        "foster_stuart_z_approx": float(fs_z),
        "adf_p": float(adf_p),
        "kpss_p": float(kpss_p),
    }


def save_series_plot(series: pd.Series, ma_window: int, title: str, path: Path) -> None:
    # Явный сдвиг на половину периода: MA(t) = avg(x[t-window+1 : t]) со сдвигом -window/2.
    ma = series.rolling(window=ma_window, min_periods=max(2, ma_window // 2)).mean().shift(-(ma_window // 2))

    plt.figure(figsize=(14, 5))
    plt.plot(series.index, series.values, label="Ряд", linewidth=1)
    plt.plot(ma.index, ma.values, label=f"Скользящее среднее (окно={ma_window}, сдвиг={ma_window // 2})", linewidth=2)
    plt.title(title)
    plt.xlabel("Дата")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_acf_pacf_plots(series: pd.Series, title_prefix: str, acf_path: Path, pacf_path: Path) -> None:
    n = len(series)
    max_lag = min(60, max(10, n // 5))

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(series.dropna(), lags=max_lag, ax=ax)
    ax.set_title(f"АКФ: {title_prefix}")
    fig.tight_layout()
    fig.savefig(acf_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_pacf(series.dropna(), lags=max_lag, ax=ax, method="ywm")
    ax.set_title(f"ЧАКФ: {title_prefix}")
    fig.tight_layout()
    fig.savefig(pacf_path, dpi=150)
    plt.close(fig)


def infer_model_type(series: pd.Series, window: int = 24 * 7) -> tuple[str, float, float]:
    rolling_mean = series.rolling(window=window, min_periods=max(10, window // 3)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(10, window // 3)).std()
    joined = pd.concat([rolling_mean, rolling_std], axis=1).dropna()
    if len(joined) < 10:
        return "additive", 0.0, np.nan

    corr, corr_pvalue = pearsonr(joined.iloc[:, 0], joined.iloc[:, 1])
    model = "multiplicative" if corr > 0.5 and (series > 0).all() else "additive"
    return model, float(corr), float(corr_pvalue)


def fit_trend_models(t: np.ndarray, y: np.ndarray) -> list[TrendModel]:
    candidates: list[TrendModel] = []

    def make_metrics(name: str, params: np.ndarray, pvalues: np.ndarray, y_pred: np.ndarray, formula: str, pred_fn):
        resid = y - y_pred
        rss = float(np.sum(resid ** 2))
        n = len(y)
        k = len(params)
        rss_safe = max(rss, 1e-12)
        aic = float(n * np.log(rss_safe / n) + 2 * k)
        bic = float(n * np.log(rss_safe / n) + k * np.log(n))
        candidates.append(
            TrendModel(
                name=name,
                params=params,
                pvalues=pvalues,
                aic=aic,
                bic=bic,
                rss=rss,
                predict=pred_fn,
                formula=formula,
            )
        )

    X_lin = sm.add_constant(t)
    lin_fit = sm.OLS(y, X_lin).fit()
    lin_params = lin_fit.params
    make_metrics(
        "linear",
        lin_params,
        lin_fit.pvalues,
        lin_fit.predict(X_lin),
        f"y = {lin_params[0]:.6f} + {lin_params[1]:.6f}*t",
        lambda tt: lin_params[0] + lin_params[1] * tt,
    )

    X_poly = np.column_stack([np.ones_like(t), t, t ** 2])
    poly_fit = sm.OLS(y, X_poly).fit()
    poly_params = poly_fit.params
    make_metrics(
        "polynomial_2",
        poly_params,
        poly_fit.pvalues,
        poly_fit.predict(X_poly),
        f"y = {poly_params[0]:.6f} + {poly_params[1]:.6f}*t + {poly_params[2]:.12f}*t^2",
        lambda tt: poly_params[0] + poly_params[1] * tt + poly_params[2] * tt ** 2,
    )

    t_pos = t.copy()
    X_log = sm.add_constant(np.log(t_pos))
    log_fit = sm.OLS(y, X_log).fit()
    log_params = log_fit.params
    make_metrics(
        "logarithmic",
        log_params,
        log_fit.pvalues,
        log_fit.predict(X_log),
        f"y = {log_params[0]:.6f} + {log_params[1]:.6f}*ln(t)",
        lambda tt: log_params[0] + log_params[1] * np.log(tt),
    )

    if np.all(y > 0):
        X_pow = sm.add_constant(np.log(t_pos))
        pow_fit = sm.OLS(np.log(y), X_pow).fit()
        c0, c1 = pow_fit.params
        make_metrics(
            "power",
            pow_fit.params,
            pow_fit.pvalues,
            np.exp(c0 + c1 * np.log(t_pos)),
            f"y = exp({c0:.6f}) * t^{c1:.6f}",
            lambda tt: np.exp(c0 + c1 * np.log(tt)),
        )

        X_exp = sm.add_constant(t)
        exp_fit = sm.OLS(np.log(y), X_exp).fit()
        e0, e1 = exp_fit.params
        make_metrics(
            "exponential",
            exp_fit.params,
            exp_fit.pvalues,
            np.exp(e0 + e1 * t),
            f"y = exp({e0:.6f} + {e1:.6f}*t)",
            lambda tt: np.exp(e0 + e1 * tt),
        )

    return sorted(candidates, key=lambda m: m.aic)


def fit_sinusoid(seasonal_component: pd.Series, period: int = 24):
    y = seasonal_component.values
    x = np.arange(1, len(y) + 1, dtype=float)

    def sinus(tt, a, phi, c):
        return a * np.sin(2 * np.pi * tt / period + phi) + c

    p0 = [np.std(y), 0.0, np.mean(y)]
    with warnings.catch_warnings():
        # Если параметры синуса плохо идентифицируются, переходим на fallback выше по коду.
        warnings.simplefilter("error", OptimizeWarning)
        params, _ = curve_fit(sinus, x, y, p0=p0, maxfev=10000)
    return params, sinus


def quality_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nonzero = np.where(y_true != 0, y_true, np.nan)
    mape = float(np.nanmean(np.abs(err / nonzero)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def format_p_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value == 0.0:
        return "< 1.00e-300"
    return f"{value:.2e}"


def build_monthly_profile(
    series: pd.Series,
    model_type: str,
) -> pd.Series:
    if model_type == "additive":
        monthly_profile = series.groupby(series.index.month).mean()
        monthly_profile = monthly_profile - monthly_profile.mean()
    else:
        monthly_profile = series.groupby(series.index.month).mean()
        profile_mean = monthly_profile.mean()
        if pd.notna(profile_mean) and abs(profile_mean) > 1e-9:
            monthly_profile = monthly_profile / profile_mean
        else:
            monthly_profile = pd.Series(1.0, index=monthly_profile.index)

    return monthly_profile.reindex(range(1, 13)).interpolate(limit_direction="both").bfill().ffill()


def run_full_analysis(series_hourly: pd.Series) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    freq_map = {
        "hours": (series_hourly, 24),
        "days": (series_hourly.resample("D").mean(), 7),
        "weeks": (series_hourly.resample("W-MON").mean(), 4),
        "months": (series_hourly.resample("MS").mean(), 12),
    }

    report_lines: list[str] = []
    report_lines.append("Первичный анализ временного ряда V3")
    report_lines.append("=" * 60)

    for freq_name, (s, ma_window) in freq_map.items():
        report_lines.append(f"\n[{freq_name.upper()}]")
        report_lines.append(f"Длина ряда: {len(s)}")
        report_lines.append(f"Проверка стационарности выполнена для частоты {freq_name}.")

        save_series_plot(
            s,
            ma_window=ma_window,
            title=f"V3 ({freq_name}) + скользящее среднее",
            path=PLOTS_DIR / f"series_ma_{freq_name}.png",
        )
        save_acf_pacf_plots(
            s,
            title_prefix=f"V3 ({freq_name})",
            acf_path=PLOTS_DIR / f"acf_{freq_name}.png",
            pacf_path=PLOTS_DIR / f"pacf_{freq_name}.png",
        )

        tests = stationarity_tests(s)
        for k, v in tests.items():
            report_lines.append(f"{k}: {v:.6f}")

        nonrandom_flags = []
        if tests["runs_p"] < 0.05:
            nonrandom_flags.append("критерий серий отвергает случайность")
        if tests["bartlett_p"] < 0.05 or tests["levene_p"] < 0.05:
            nonrandom_flags.append("дисперсия непостоянна")
        if tests["t_p"] < 0.05:
            nonrandom_flags.append("средние неодинаковы на подвыборках")
        if tests["adf_p"] > 0.05 or tests["kpss_p"] < 0.05:
            nonrandom_flags.append("есть признаки нестационарности")

        if nonrandom_flags:
            report_lines.append("Вывод: обнаружены неслучайные компоненты: " + "; ".join(nonrandom_flags) + ".")
        else:
            report_lines.append("Вывод: явных неслучайных компонент по выбранным тестам не обнаружено.")

    report_lines.append("\n" + "=" * 60)
    report_lines.append("Декомпозиция и прогноз почасового ряда с суточной и месячной сезонностью")

    model_type, amp_corr, amp_corr_pvalue = infer_model_type(series_hourly)
    report_lines.append(
        "Выбран тип модели: "
        f"{model_type} (corr(rolling_mean, rolling_std)={amp_corr:.4f}, p-value={format_p_value(amp_corr_pvalue)})"
    )
    if np.isnan(amp_corr_pvalue):
        report_lines.append("Вывод по типу модели: недостаточно данных для оценки статистической значимости корреляции.")
    elif amp_corr_pvalue < 0.05:
        report_lines.append("Вывод по типу модели: корреляция статистически значима на уровне 0.05.")
    else:
        report_lines.append("Вывод по типу модели: статистически значимая связь не подтверждена на уровне 0.05.")

    control_start = series_hourly.index.max().to_period("M").to_timestamp()
    train = series_hourly[series_hourly.index < control_start]
    control = series_hourly[series_hourly.index >= control_start]

    report_lines.append(f"Обучающий отрезок: {train.index.min()} .. {train.index.max()} ({len(train)} точек)")
    report_lines.append(f"Контрольный отрезок: {control.index.min()} .. {control.index.max()} ({len(control)} точек)")

    decomposition = seasonal_decompose(train, model=model_type, period=24, extrapolate_trend="freq")
    seasonal_comp = decomposition.seasonal

    if model_type == "additive":
        day_deseasonalized = train - seasonal_comp
    else:
        day_deseasonalized = train / seasonal_comp.replace(0, np.nan)
        day_deseasonalized = day_deseasonalized.interpolate().bfill().ffill()

    monthly_profile = build_monthly_profile(day_deseasonalized, model_type=model_type)
    monthly_profile_str = ", ".join(f"{month}:{value:.4f}" for month, value in monthly_profile.items())
    report_lines.append(f"Месячная сезонность (профиль по месяцам): {monthly_profile_str}")

    monthly_train_component = train.index.month.map(monthly_profile).values
    if model_type == "additive":
        deseasonalized = day_deseasonalized - monthly_train_component
    else:
        safe_monthly = np.where(np.abs(monthly_train_component) < 1e-9, np.nan, monthly_train_component)
        deseasonalized = day_deseasonalized / safe_monthly
        deseasonalized = deseasonalized.replace([np.inf, -np.inf], np.nan).interpolate().bfill().ffill()

    t_train = np.arange(1, len(deseasonalized) + 1, dtype=float)
    trend_models = fit_trend_models(t_train, deseasonalized.values)
    best_model = trend_models[0]
    trend_fitted_train = best_model.predict(t_train)

    report_lines.append("\nКандидаты тренда (по возрастанию AIC):")
    for tm in trend_models:
        pvals_str = ", ".join(format_p_value(float(p)) for p in tm.pvalues)
        report_lines.append(
            f"- {tm.name}: AIC={tm.aic:.3f}, BIC={tm.bic:.3f}, RSS={tm.rss:.3f}, p-values=[{pvals_str}], {tm.formula}"
        )

    report_lines.append(f"Выбран тренд: {best_model.name}")
    best_model_pvals = ", ".join(format_p_value(float(p)) for p in best_model.pvalues)
    best_model_max_p = float(np.max(best_model.pvalues))
    significance_conclusion = (
        "все коэффициенты статистически значимы на уровне 0.05"
        if best_model_max_p < 0.05
        else "не все коэффициенты статистически значимы на уровне 0.05"
    )
    report_lines.append(f"p-values выбранного тренда: [{best_model_pvals}]")
    report_lines.append(f"Вывод по значимости коэффициентов: {significance_conclusion}.")

    sinus_ok = True
    try:
        sinus_params, sinus_fn = fit_sinusoid(seasonal_comp, period=24)
        report_lines.append(
            "Синусоидальная аппроксимация сезонности: "
            f"S(t) = {sinus_params[0]:.6f}*sin(2*pi*t/24 + {sinus_params[1]:.6f}) + {sinus_params[2]:.6f}"
        )
    except Exception:
        sinus_ok = False
        seasonal_profile = seasonal_comp.groupby(seasonal_comp.index.hour).mean()
        report_lines.append("Синусоидальная аппроксимация не сошлась; используется средний почасовой профиль сезонности.")

    n_train = len(train)
    n_control = len(control)
    t_future = np.arange(n_train + 1, n_train + n_control + 1, dtype=float)
    trend_forecast = best_model.predict(t_future)

    if sinus_ok:
        seasonal_forecast = sinus_fn(t_future, *sinus_params)
    else:
        seasonal_forecast = control.index.hour.map(seasonal_profile).values

    monthly_forecast = control.index.month.map(monthly_profile).values

    if model_type == "additive":
        y_pred = trend_forecast + seasonal_forecast + monthly_forecast
        final_formula = "Y(t) = Trend(t) + S_day(t) + S_month(t) + e(t)"
    else:
        y_pred = trend_forecast * seasonal_forecast * monthly_forecast
        final_formula = "Y(t) = Trend(t) * S_day(t) * S_month(t) * e(t)"

    metrics = quality_metrics(control.values, y_pred)

    report_lines.append("\nИтоговая модель декомпозиции:")
    report_lines.append(final_formula)
    report_lines.append(f"Trend(t): {best_model.formula}")
    report_lines.append(f"p-values коэффициентов Trend(t): [{best_model_pvals}]")
    report_lines.append("S_day(t): суточная сезонность, оцененная по часовому профилю/синусоиде с period=24.")
    report_lines.append("S_month(t): месячная сезонность, оцененная как профиль по месяцам календарного года.")
    report_lines.append("Trend(t) оценивался по ряду после исключения и суточной, и месячной сезонности.")

    report_lines.append("\nТочность прогноза на контрольном отрезке (последний месяц):")
    for k, v in metrics.items():
        report_lines.append(f"{k}: {v:.6f}")

    plt.figure(figsize=(14, 5))
    plt.plot(control.index, control.values, label="Control (fact)", linewidth=1.5)
    plt.plot(control.index, y_pred, label="Forecast", linewidth=1.5)
    plt.title("Прогноз на контрольный месяц")
    plt.xlabel("Дата")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "forecast_control_month.png", dpi=150)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(train.index, deseasonalized.values, label="Ряд без суточной и месячной сезонности", linewidth=1)
    plt.plot(train.index, trend_fitted_train, label=f"Тренд: {best_model.name}", linewidth=2)
    plt.title("Подгонка тренда на обучающем отрезке")
    plt.xlabel("Дата")
    plt.ylabel("kWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "trend_fit_train.png", dpi=150)
    plt.close()

    report_lines.append("\nОбщие выводы:")
    report_lines.append("1) Проверка стационарности выполнена отдельно для 4 частот дискретизации: часы, дни, недели и месяцы.")
    report_lines.append("2) Для почасового ряда построена декомпозиционная модель с суточной и месячной сезонностью и выбранным трендом.")
    report_lines.append("3) Прогноз на декабрь 2014 сформирован как комбинация тренда, суточной и месячной сезонности; ошибка оценена по MAE/RMSE/MAPE.")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    series = load_v3_series(Path("data.csv"))
    run_full_analysis(series)
    print("Готово. Сформированы:")
    print(f"- {REPORT_PATH}")
    print(f"- {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
