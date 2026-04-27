import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class AnalysisSettings:
    frequency: float = 1000.0
    start_test: float = 99.0
    search_window: float = 5.0
    jump: float = 18.0
    compression_number: int = 12
    compression_duration: float = 13.0
    regression_window: float = 5.0
    regression_increment: float = 0.5
    regression_delay_from_compression_start: float = 1.0
    regression_stop_before_end: float = 1.0
    excluded_regressions: set[int] | None = None
    sd_window: float = 0.3
    extension: float = 6.0


def parse_excluded_regressions(raw: str) -> set[int]:
    cleaned = [x.strip() for x in raw.split(",") if x.strip()]
    return {int(x) for x in cleaned}


def analyze_vop(data: np.ndarray, settings: AnalysisSettings):
    if settings.excluded_regressions is None:
        settings.excluded_regressions = {4, 5, 6, 7, 8, 9, 10, 11, 12}

    number_of_regressions = int(
        (
            settings.compression_duration
            - settings.regression_delay_from_compression_start
            - settings.regression_stop_before_end
            - settings.regression_window
        )
        / settings.regression_increment
    )

    start_test_raw = int(settings.start_test * settings.frequency)
    search_window_raw = int(settings.search_window * settings.frequency)
    jump_raw = int(settings.jump * settings.frequency)
    sd_window_raw = int(settings.sd_window * settings.frequency)

    time = np.arange(len(data)) / settings.frequency

    half_window = sd_window_raw // 2
    rolling_mean = np.array(
        [
            np.mean(data[i - half_window : i + half_window])
            if i - half_window >= 0 and i + half_window < len(data)
            else np.nan
            for i in range(len(data))
        ]
    )

    dt = 1 / settings.frequency
    rolling_mean_derivative = np.gradient(rolling_mean, dt)

    peak_values: list[float] = []
    inflation_start_times: list[float] = []

    start = start_test_raw
    for _ in range(settings.compression_number):
        if start + search_window_raw >= len(rolling_mean_derivative):
            break
        end = start + search_window_raw
        window = rolling_mean_derivative[start:end]

        if np.all(np.isnan(window)):
            start += jump_raw
            continue

        local_max_idx = np.nanargmax(window)
        global_idx = start + local_max_idx

        peak_values.append(rolling_mean_derivative[global_idx])
        inflation_start_times.append(global_idx / settings.frequency)

        start = global_idx + jump_raw

    fig, ax = plt.subplots(figsize=(14, 8))

    (line1,) = ax.plot(time, data, linestyle="-", label="Signal")
    (line2,) = ax.plot(time, rolling_mean, color="black", linewidth=2, label="Rolling Mean")
    (line4,) = ax.plot(
        time,
        rolling_mean_derivative / 10 - 0.2,
        color="red",
        linewidth=1,
        label="Slope",
    )

    peak_values_scaled = np.array(peak_values) / 10 - 0.2
    (line5,) = ax.plot(
        inflation_start_times,
        peak_values_scaled,
        "o",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2,
        label="Inflation Start",
    )

    all_slopes = []
    all_regressions = []

    for compression_idx in range(settings.compression_number):
        compression_number = compression_idx + 1
        if compression_number in settings.excluded_regressions:
            continue
        if compression_idx >= len(inflation_start_times):
            continue

        regression_start = inflation_start_times[compression_idx]

        compression_slopes = []
        compression_regressions = []

        for i in range(number_of_regressions):
            start_time = (
                regression_start
                + settings.regression_delay_from_compression_start
                + i * settings.regression_increment
            )
            end_time = start_time + settings.regression_window

            start_raw = int(start_time * settings.frequency)
            end_raw = int(end_time * settings.frequency)

            x_reg = time[start_raw:end_raw]
            y_reg = data[start_raw:end_raw]

            if len(x_reg) < 2:
                continue

            slope, intercept = np.polyfit(x_reg, y_reg, 1)

            compression_slopes.append(slope)
            compression_regressions.append(
                {
                    "regression_number": i + 1,
                    "start_time": start_time,
                    "end_time": end_time,
                    "slope": slope,
                    "intercept": intercept,
                }
            )

            x_reg_extended = np.linspace(
                start_time - settings.extension,
                end_time + settings.extension,
                1000,
            )
            y_reg_extended = slope * x_reg_extended + intercept

            ax.plot(x_reg_extended, y_reg_extended, linewidth=2)

            x_label = start_time
            y_label = slope * x_label + intercept

            ax.text(
                x_label - 0.2,
                y_label,
                f"{i + 1}",
                fontsize=10,
                ha="right",
                va="center",
            )

            y_start = slope * start_time + intercept
            y_end = slope * end_time + intercept

            ax.plot(
                start_time,
                y_start,
                "o",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=2,
            )

            ax.plot(
                end_time,
                y_end,
                "o",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=2,
            )

        all_slopes.append(compression_slopes)
        all_regressions.append(compression_regressions)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage / Derived Signals")
    ax.set_title("VOP Signal Analysis")
    ax.legend(handles=[line1, line2, line4, line5])
    ax.grid(False)

    rows = []
    for compression_idx, compression_regs in enumerate(all_regressions, start=1):
        for reg in compression_regs:
            rows.append(
                {
                    "compression": compression_idx,
                    "regression_number": reg["regression_number"],
                    "start_time_s": round(reg["start_time"], 3),
                    "end_time_s": round(reg["end_time"], 3),
                    "slope": reg["slope"],
                }
            )

    slopes_df = pd.DataFrame(rows)
    return fig, slopes_df, inflation_start_times, peak_values


st.set_page_config(page_title="VOP Signal Analysis", layout="wide")
st.title("VOP Signal Analysis Web App")

st.markdown(
    "Upload a plain-text signal file (one numeric sample per line) and run the same analysis workflow as the original script."
)

uploaded_file = st.file_uploader("Upload VOP text file", type=["txt", "csv", "dat"])

with st.sidebar:
    st.header("Settings")
    frequency = st.number_input("Sampling frequency (Hz)", min_value=1.0, value=1000.0)

    st.subheader("Peak detection")
    start_test = st.number_input("StartTest (s)", value=99.0)
    search_window = st.number_input("SearchWindow (s)", value=5.0)
    jump = st.number_input("Jump (s)", value=18.0)
    compression_number = st.number_input("CompressionNumber", min_value=1, value=12, step=1)

    st.subheader("Regression")
    compression_duration = st.number_input("CompressionDuration (s)", value=13.0)
    regression_window = st.number_input("RegressionWindow (s)", value=5.0)
    regression_increment = st.number_input("RegressionIncrement (s)", value=0.5)
    regression_delay = st.number_input("Regression_Delay_from_CompressionStart (s)", value=1.0)
    regression_stop = st.number_input("Regression_Stop_before_End (s)", value=1.0)
    excluded_regressions_raw = st.text_input(
        "ExcludedRegressions (comma separated)",
        value="4,5,6,7,8,9,10,11,12",
    )
    sd_window = st.number_input("SDwindow (s)", value=0.3)
    extension = st.number_input("Regression extension (s)", value=6.0)

if uploaded_file is None:
    st.info("Please upload a data file to start analysis.")
else:
    try:
        data = np.loadtxt(io.BytesIO(uploaded_file.getvalue()))
        settings = AnalysisSettings(
            frequency=frequency,
            start_test=start_test,
            search_window=search_window,
            jump=jump,
            compression_number=int(compression_number),
            compression_duration=compression_duration,
            regression_window=regression_window,
            regression_increment=regression_increment,
            regression_delay_from_compression_start=regression_delay,
            regression_stop_before_end=regression_stop,
            excluded_regressions=parse_excluded_regressions(excluded_regressions_raw),
            sd_window=sd_window,
            extension=extension,
        )

        fig, slopes_df, inflation_start_times, peak_values = analyze_vop(data, settings)
        st.pyplot(fig)

        st.subheader("Detected inflation starts")
        st.write(pd.DataFrame({"inflation_start_s": inflation_start_times, "peak_value": peak_values}))

        st.subheader("Regression slopes")
        if slopes_df.empty:
            st.warning("No regression windows were generated with the current settings.")
        else:
            st.dataframe(slopes_df, use_container_width=True)
            st.download_button(
                "Download slopes CSV",
                data=slopes_df.to_csv(index=False),
                file_name="vop_regression_slopes.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Could not process file: {exc}")
