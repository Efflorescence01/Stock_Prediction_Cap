# Stock Analysis & Prediction Capstone 

## Table of Contents 
- Dateset
- Project Overview
- Project Flowchart
- Project Organization

### Dataset 
There is a new dataset called Cleaned_Dataset containing a CSV with the GOOG data frame. We will be using this CSV file to create our machine-learning models.
The GOOG data frame contains the columns:
- open
- high
- low
- close
- adjusted close
- volume
- Tomorrow_Adj_close
- Target

The datasets were obtained on Kaggle: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks. This dataset is updated regularly by the main contributor of that page. As such the current dataset that I have on hand will be behind compared to the one on the website. As a note, the dataset was retrieved on March.3,2024. 
The dataset contains 3 CSV files: 
- sp500_index.csv
    - `Date`
    - `S&P500`
- sp500_companies.csv
    - `Exchange`
    - `Symbol`
    - `Shortname`
    - `Longname`
    - `Sector`
    - `Industry`
    - `Currentprice`
    - `Marketcap`
    - `Ebitda`
    - `Revenuegrowth`
    - `City`
    - `State`
    - `Country`
    - `Fulltimeemployees`
    - `Longbusinesssummary`
    - `Weight`
- sp500_stocks.csv
    - `Date`
    - `Symbol`
    - `Adj Close`
    - `Close`
    - `High`
    - `Low`
    - `Open`
    - `Volume`
### Project Overview 
  This is a data analysis & machine learning project for stock trading and investing. It is focused on the S&P 500 index and the companies within that index during the period from 2010 to the present year (2024). This project will utilize Juypterlab/Notebook to perform data cleaning, data analysis, and modeling. The language of choice will be Python. We will also be using data science-related libraries such as matplotlib, pandas, NumPy, and Scikitlearn. As a stock trading enthusiast, I started this project to improve my skills and knowledge in data analysis regarding stock data. Previously, I only worked with simple statistical terms such as averages, and maximum and minimum values. As such, I wanted a more granular perspective of stock data to improve my analysis of my trades and thus increase my earnings and win rate.Individuals/businesses engaged in investing or trading activities may obtain some insight from this project. 
### Walkthrough Demo

### Project Flowchart 
    1. Open up the Stock_API Notebook
        a. Follow the markdown cells to set up your API keys and to understand the documentation for the API.
        b. Run the cells until you are able to save the file

    2. Open up the Stock Modeling Notebook
        a. Import the saved CSV file
        b. Preprocess the data frame and add the features that you desire.
        c. Run the models with your data frame and evaluate the results.
### Project Organization 
    
