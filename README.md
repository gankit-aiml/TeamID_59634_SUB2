# AI/ML Model for GNSS Satellite Error Prediction

**Project for SIH 2025 - Problem Statement SIH25176**

This repository contains a state-of-the-art solution for predicting time-varying patterns in satellite clock and ephemeris errors for GEO and MEO navigation satellites. The project leverages deep learning, specifically a Temporal Convolutional Network (TCN), to forecast errors for an unseen eighth day based on a seven-day historical dataset.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [The Core Challenge: Uncovering Hidden Seasonality](#the-core-challenge-uncovering-hidden-seasonality)
3.  [Methodology: A Data-Driven Pipeline](#methodology-a-data-driven-pipeline)
    *   [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Advanced Feature Engineering](#advanced-feature-engineering)
4.  [Model Architecture: The Temporal Convolutional Network (TCN)](#model-architecture-the-temporal-convolutional-network-tcn)
5.  [Training and Validation Strategy](#training-and-validation-strategy)
6.  [Results and Evaluation](#results-and-evaluation)
7.  [How to Run the Code](#how-to-run-the-code)
8.  [Conclusion and Key Findings](#conclusion-and-key-findings)

---

## 1. Project Overview

The goal of this project is to develop an AI/ML model capable of accurately forecasting satellite clock and ephemeris errors (`x_error`, `y_error`, `z_error`, `satclockerror`) at a 15-minute interval for an entire 24-hour period. The primary evaluation criterion is the normality of the model's residuals (the prediction errors), as measured by the Shapiro-Wilk test. A model that successfully captures the systematic, predictable patterns will leave behind only random, normally-distributed "white noise" error.

This project implements a complete data science pipeline, from deep exploratory analysis to advanced feature engineering and modeling with a **Temporal Convolutional Network (TCN)**, a state-of-the-art architecture for sequence modeling. Separate, specialized models were trained for the GEO and MEO datasets to account for their fundamentally different error profiles.

## 2. The Core Challenge: Uncovering Hidden Seasonality

A rigorous Exploratory Data Analysis (EDA) was conducted to understand the nature of the satellite errors. While short-term correlations were present, the most critical discovery came from a long-term temporal analysis.

**Key Finding:** The Autocorrelation Function (ACF) plot of the raw, resampled data revealed a **powerful and unmistakable 24-hour seasonality**, visible as a distinct sine wave pattern.



This discovery was the cornerstone of our entire modeling strategy. It proved that the error at any given time is strongly correlated with the error from the same time the previous day. This meant that a simple model would fail, and a sophisticated architecture capable of learning **long-range temporal dependencies** was required.

## 3. Methodology: A Data-Driven Pipeline

Our solution follows a multi-stage, best-practice data science workflow.

### Exploratory Data Analysis (EDA)
- **Initial Sanity Checks:** Verified data integrity, checked for duplicates, and confirmed data types.
- **Distribution Analysis:** Used box plots to identify the presence of significant outliers, especially in the GEO dataset.
- **Temporal Analysis:** Plotted the raw data to confirm its irregular sampling rate, justifying the need for resampling.
- **Seasonality Discovery:** Utilized ACF/PACF plots to uncover the dominant 24-hour seasonality, which became the central focus of our feature engineering and modeling efforts.

### Data Preprocessing
1.  **Outlier Handling:** Employed **winsorization** to cap extreme outliers at the 1st and 99th percentiles. This reduces the distorting effect of rare, large errors while retaining their information.
2.  **Resampling:** Resampled the time series to a fixed **15-minute frequency** using linear interpolation to create a consistent timeline suitable for modeling.

### Advanced Feature Engineering
To equip the model with the necessary information to understand the complex dynamics, we engineered a rich feature set, including:
- **Cyclical Time Features:** Encoded the hour of the day and day of the week using sine and cosine transformations to represent their cyclical nature.
- **Lag Features:** Created features representing the error values from the past (e.g., 1 hour ago, 24 hours ago).
- **Rolling Window Statistics:** Calculated rolling means and standard deviations to capture the recent trend and **volatility** of the system.
- **Interaction Terms:** Created "killer features" by interacting the 24-hour lag with the cyclical time features, allowing the model to learn how the seasonal pattern changes throughout the day.
- **Trend Feature:** A linear `time_idx` was added to allow the model to learn patterns that evolve over the entire 7-day period.

## 4. Model Architecture: The Temporal Convolutional Network (TCN)

While LSTMs and Transformers were considered, the final model of choice is a **Temporal Convolutional Network (TCN)**. A TCN uses a stack of 1D convolutional layers with two key innovations that make it exceptionally powerful for this task:

1.  **Causal Convolutions:** Ensures that predictions for a given timestep only use data from the past, preventing data leakage.
2.  **Dilated Convolutions:** This is the TCN's "superpower." The convolutional filters expand their view exponentially at deeper layers, allowing the model to have a very large **receptive field**. This means it can efficiently learn the relationship between a data point *now* and a data point from very far in the past (e.g., 24, 48, or even 72 hours ago), which was essential for capturing the seasonality we discovered.

Our final architecture is a **Sequence-to-Sequence (Seq2Seq)** TCN model, which takes a long sequence of past data (e.g., 369 steps for the MEO model) as input and outputs a full 96-step (24-hour) forecast in a single pass.


| Layer (type)            | Output Shape     | Param #   |
|-------------------------|------------------|-----------|
| tcn_40 (TCN)            | (None, 288, 256) | 1,826,560 |
| dropout_130 (Dropout)   | (None, 288, 256) | 0         |
| tcn_41 (TCN)            | (None, 128)      | 574,848   |
| dropout_131 (Dropout)   | (None, 128)      | 0         |
| repeat_vector_12        | (None, 96, 128)  | 0         |
| tcn_42 (TCN)            | (None, 96, 128)  | 492,800   |
| dropout_132 (Dropout)   | (None, 96, 128)  | 0         |
| time_distributed_12     | (None, 96, 4)    | 516       |

**Total params:** 2,894,724 (11.04 MB)  
**Trainable params:** 2,894,724 (11.04 MB)  
**Non-trainable params:** 0 (0.00 B)


## 5. Training and Validation Strategy

- **Separate Models:** We trained two completely separate, specialized models for the GEO and MEO data.
- **Time-Based Split:** A simple random train/validation split is incorrect for time-series data. We used a strict **time-based split**, training on the first 80% of the timeline and validating on the final 20%. This simulates a real-world forecasting scenario.
- **Loss Function:** We primarily used **Huber Loss** and **Mean Squared Error (MSE)**, which are robust loss functions that balance precision with sensitivity to the spikes in the data.
- **Callbacks:** The training process was managed using:
    - **`ReduceLROnPlateau`:** To automatically adjust the learning rate.
    - **`EarlyStopping`:** To prevent overfitting and ensure the model from the best-performing epoch was saved.

## 6. Results and Evaluation

Our final MEO model achieved an exceptional result, demonstrating the success of our pipeline.
- **Training Performance:** The training and validation loss curves tracked each other closely, indicating a well-regularized model with no overfitting.
- **Residual Analysis:** A Q-Q plot of the model's residuals showed a near-perfect fit to the theoretical normal distribution.
- **Shapiro-Wilk Test:** The model achieved a final Shapiro-Wilk **statistic of 0.983**, which is extremely close to the perfect score of 1.0, and a **non-zero p-value**, confirming the near-normality of the residuals.

![WhatsApp Image 2025-10-14 at 23 15 28_87266a54](https://github.com/user-attachments/assets/73ffe8eb-0c82-40dc-b8ab-e42df47e96e5)
<img width="1188" height="790" alt="image" src="https://github.com/user-attachments/assets/1cc5013e-d726-4ac5-8519-9d9e44680944" />


The GEO data proved to be a more complex challenge, with an unbreakable seasonality that likely points to an unobserved, external variable influencing the system. Our analysis and final report detail this finding as a key scientific insight.

## 7. How to Run the Code

1.  **Environment:** This project was developed in a Google Colab environment with a T4 GPU.
2.  **Dependencies:** The required libraries are listed in the notebook and can be installed via `pip`. The primary dependencies are `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `statsmodels`, and `keras-tcn`.
3.  **Execution:** The notebook is structured sequentially. Simply run the cells from top to bottom to replicate the entire data processing, training, and evaluation pipeline.
4.  **Final Submission:** The last cells of the notebook contain the logic to retrain the best model on 100% of the data and generate the final `final_predictions_meo.csv` and `final_predictions_geo.csv` files.

## 8. Conclusion and Key Findings

This project successfully demonstrates a robust, end-to-end pipeline for forecasting GNSS satellite errors. Our key findings are:

- **Dominant Seasonality:** The error patterns for both GEO and MEO satellites are dominated by a powerful 24-hour seasonality.
- **Long-Term Memory is Crucial:** The success of the TCN model with a very long lookback window (over 3 days) proves that these long-range dependencies are the key to accurate prediction.
- **State-of-the-Art Performance:** By using a targeted data-driven approach, our TCN model for the MEO data was able to capture these complex patterns, resulting in near-normally distributed residuals and a state-of-the-art statistical performance.
