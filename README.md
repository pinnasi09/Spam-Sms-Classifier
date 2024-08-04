# Spam-Sms-Classifier
# Spam SMS Classifier

This project is a machine learning model that classifies SMS messages as either spam or ham (not spam). The classifier is built using Python and various machine learning libraries, and it includes a graphical user interface (GUI) for ease of use.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Spam SMS Classifier is designed to help users automatically filter out spam messages from their SMS inbox. This project includes data preprocessing, feature extraction, model training, evaluation, and a GUI for classifying new SMS messages.

## Features

- Preprocesses raw SMS data
- Extracts relevant features using techniques like TF-IDF
- Trains a machine learning model to classify messages
- Evaluates the model's performance on a test set
- Provides a GUI for classifying SMS messages

## Installation

To install and run this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/spam-sms-classifier.git
    cd spam-sms-classifier
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the spam SMS classifier, follow these steps:

1. Ensure you have the SMS dataset in the proper format (CSV file with `message` and `label` columns).
2. Run the main script to preprocess the data, train the model, evaluate the model, and launch the GUI:
    ```bash
    python main.py
    ```

### Example Dataset Format

The dataset should be a CSV file with two columns:
- `message`: The content of the SMS message
- `label`: The classification label (`spam` or `ham`)

### GUI Usage

1. Enter an SMS message into the text area.
2. Click the "Classify" button to get the prediction result displayed below the button.

## Data

The dataset used for this project should contain SMS messages labeled as spam or ham. A typical dataset is formatted as a CSV file with two columns:
- `message`: The content of the SMS message
- `label`: The classification label (`spam` or `ham`)

## Model

The classifier is built using the following machine learning libraries:
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Tkinter

The model training involves the following steps:
1. Preprocessing the text data (e.g., tokenization, removing stop words)
2. Extracting features using techniques like TF-IDF
3. Training a machine learning model (e.g., Multinomial Naive Bayes)

## Results

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics provide insights into how well the classifier distinguishes between spam and ham messages.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature-or-bugfix-name
    ```
3. Commit your changes:
    ```bash
    git commit -m "Description of your changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-or-bugfix-name
    ```
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## MY 
i have developed this project in order to classify the spam sms . 

