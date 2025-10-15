# TCN-ISRO: Temporal Convolutional Network for MEO Satellite Error Prediction

## 🚀 Project Overview

This Jupyter Notebook (`TCN_ISRO.ipynb`) implements a **Temporal Convolutional Network (TCN)**, a powerful type of neural network for sequence modeling, to analyze and potentially predict positioning and clock errors for **Medium Earth Orbit (MEO)** satellites using data provided for an ISRO (Indian Space Research Organisation) challenge.

The primary goal is to model the time-series behavior of errors in the X, Y, and Z coordinates, as well as the satellite clock error, to improve the accuracy of navigation and positioning services.

---

## ✨ Key Features

* **Time Series Data Handling:** Robust data loading, concatenation, and time-index sorting using `pandas`.
* **Data Cleaning:** Identification and removal of duplicate timestamp entries.
* **Data Exploration (EDA):** Initial statistical and visual (boxplot) analysis of the error metrics.
* **TCN Implementation:** Utilizes the `keras-tcn` library to build a Temporal Convolutional Network model architecture.
* **Advanced Preprocessing:** Uses `RobustScaler` to normalize the data, which is suitable for data that may contain outliers, as suggested by the exploratory data analysis.
* **Deep Learning Tools:** Imports core modules from `tensorflow.keras`, including various layers (Dense, LayerNormalization, MultiHeadAttention, Dropout, etc.), suggesting a complex, potentially hybrid model architecture is being developed (e.g., TCN with Transformer components).

---

## 💻 Prerequisites and Installation

To run this notebook, you need Python and the following packages. The notebook was developed in a Kaggle environment using Python 3.

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone <repository-link>
    cd TCN-ISRO
    ```

2.  **Install Required Libraries:**
    The notebook explicitly installs the `keras-tcn` library, which relies on `tensorflow` and `keras` (version 3.5.0+).

    ```bash
    !pip install keras-tcn
    # Other standard libraries (pandas, numpy, matplotlib, seaborn, scikit-learn) are typically included in environments like Kaggle/Colab.
    ```

---

## 📂 Data

The notebook uses data from a Kaggle dataset designed for an ISRO challenge on error prediction.

* **Source:** `dogedev2006/sih-isro-error`
* **Files Used:**
    * `/kaggle/input/sih-isro-error/SIH_Data_PS-08/DATA_MEO_Train.csv`
    * `/kaggle/input/sih-isro-error/SIH_Data_PS-08/DATA_MEO_Train2.csv`
* **Features (Time Series Columns):**
    * `x_error (m)`
    * `y_error (m)`
    * `z_error (m)`
    * `satclockerror (m)`

---

## 📝 Notebook Structure

The notebook is logically organized into the following high-level steps:

1.  **Setup and Installation:** Imports necessary tools, logs into KaggleHub, and installs `keras-tcn`.
2.  **Library Imports:** Loads all required scientific and deep learning libraries, including `pandas`, `sklearn` preprocessors, `tensorflow.keras`, and `tcn`.
3.  **`#EXPLORATORY DATA ANALYSIS`:**
    * Loads and combines `DATA_MEO_Train.csv` and `DATA_MEO_Train2.csv`.
    * Handles data types (converts `utc_time` to datetime).
    * Performs data checks (`.info()`, `.describe()`).
    * Removes duplicates (145 duplicates identified and dropped).
    * Visualizes data distribution with a boxplot.
4.  **Data Preprocessing (Subsequent cells):** (Anticipated steps based on imports) Scaling the data using `RobustScaler` and creating sequence data (lookback windows) for the time series model.
5.  **TCN Model Definition:** (Anticipated step) Building the TCN-based deep learning model, likely using the imported `TCN` and other Keras layers.
6.  **Model Training and Evaluation:** (Anticipated step) Compiling, fitting, and evaluating the TCN model on the prepared time series data.

---

## 🏃 Usage

To reproduce the analysis and model training:

1.  Ensure you are running the notebook in an environment (like Kaggle or Google Colab) that can access the specified Kaggle data sources.
2.  Execute all cells sequentially.
3.  The EDA section provides insights into the error distribution and scale, which informs the choice of `RobustScaler`.
4.  The final model output (which is not included in the provided snapshot) will show the performance metrics for the time series prediction.

---

## 🤝 Contribution

* **Author:** *[Insert Your Name/Kaggle Username Here]*
* **License:** *[Insert License Type Here, e.g., MIT, Apache 2.0]*
