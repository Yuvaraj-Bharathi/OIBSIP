# OIBSIP - Data Analytics Level 1 Projects

This repository contains the Level 1 Data Analytics tasks completed for the OASIS INFOBYTE SIP internship.

## Tasks Included

- `yuvarajbharathi_Task1.py`
- `yuvarajbharathi_Task2.py`
- `yuvarajbharathi_Task3.py`
- `yuvarajbharathi_Task4.py`

## Project Objectives

### Task 1: EDA on Retail Sales Data

Perform exploratory data analysis on retail sales data to understand sales patterns, customer demographics, product performance, and business insights.

### Task 2: Customer Segmentation Analysis

Use RFM features and K-Means clustering to segment customers into meaningful groups for targeted marketing.

### Task 3: Cleaning Data

Clean a messy dataset by handling null values, duplicate rows, inconsistent formats, data type issues, and outliers.

### Task 4: Sentiment Analysis

Build sentiment classification models using TF-IDF, Naive Bayes, and Logistic Regression, then evaluate model performance.

## Setup

```powershell
pip install -r requirements.txt
```

## How to Run

Replace the dataset path in each command with your downloaded CSV file.

```powershell
python yuvarajbharathi_Task1.py --csv "data/retail_sales.csv"
python yuvarajbharathi_Task2.py --csv "data/online_retail.csv"
python yuvarajbharathi_Task3.py --csv "data/messy_dataset.csv"
python yuvarajbharathi_Task4.py --csv "data/sentiment.csv" --text-column "review" --label-column "sentiment"
python yuvarajbharathi_Task4.py --csv "data/train.ft.txt"
```

All charts and output files are saved in an `outputs` folder.

## Expected Outputs

- Dataset inspection summaries
- Descriptive statistics
- Charts and visualizations
- Cleaned CSV file for Task 3
- Customer segment CSV file for Task 2
- Model evaluation metrics for Task 4
- Confusion matrices and error analysis

## Task 4 Dataset Note

If you use Kaggle's "Amazon Reviews for Sentiment Analysis" dataset, it downloads as:

- `train.ft.txt`
- `test.ft.txt`

Place `train.ft.txt` inside the `data` folder and run:

```powershell
python yuvarajbharathi_Task4.py --csv "data/train.ft.txt"
```

The script automatically converts `__label__1` to `negative` and `__label__2` to `positive`.

## Submission Checklist

- GitHub repository name should be `OIBSIP`.
- Include this `README.md` file in the repository.
- Upload all source code files.
- Upload screenshots/charts or generated outputs if required by the evaluator.
- Add project demonstration links if you create videos or live demos.
- Final file naming follows the required format: `YourName_TaskNumber`.

Final file names:

- `yuvarajbharathi_Task1.py`
- `yuvarajbharathi_Task2.py`
- `yuvarajbharathi_Task3.py`
- `yuvarajbharathi_Task4.py`

## Notes for Jupyter Notebook Submission

These scripts are designed to be easy to convert into Jupyter notebooks. Copy each logical section into separate notebook cells and add markdown observations after every chart, because the task list specifically asks for explanation and observations.
