# Stock Analysis & Prediction Capstone
![alt text](https://miro.medium.com/v2/resize:fit:1252/format:webp/0*SaNg8uUaKCMQSS5g.jpg)

## Table of Contents
- [Latest Updates](#Latest-Updates)
- [Project Overview](#Project-Overview)
- [Installation](#Installing-&-Running-Project)
- [Dateset](#Dataset)
- [Project Flowchart](#Project-Flowchart)
- [Project Organization](#Project-Organization)

### Latest Updates
As this is still my first data science project, I will continue to update it for further improvements. Some things that I will be updating are the code(More dynamic, better readability) and  the machine-learning models(Trying new models or improving on the current ones).

### Project Overview 
  This is a data analysis & machine learning project for stock trading and investing. It is focused on the S&P 500 index and the companies within that index during the period from 2010 to the present year (2024). This project will utilize Juypterlab/Notebook to perform data cleaning, data analysis, and modeling. The language of choice will be Python. We will also be using data science-related libraries such as matplotlib, pandas, NumPy, and Scikitlearn. As a stock trading enthusiast, I started this project to improve my skills and knowledge in data analysis regarding stock data. Previously, I only worked with simple statistical terms such as averages, and maximum and minimum values. As such, I wanted a more granular perspective of stock data to improve my analysis of my trades and thus increase my earnings and win rate.Individuals/businesses engaged in investing or trading activities may obtain some insight from this project.

### Installing & Running Project
1. We will be using Anaconda to help run our notebooks. Download Anaconda [Here](https://www.anaconda.com/download/success).
2. Next, in Anaconda Prompt, create an environment by typing 'conda create -n (Name of Environment)'.
3. Activate the environment by typing 'conda activate (Name of Environment)'.
4. Finally, we can install all the necessary packages to run the notebooks. We will be installing 'pandas', 'numpy', 'matplotlib', 'scikit-learn','Tensorflow'.
   We will either use the 'pip' command or the 'conda install' command to install these libraries. [Installation](https://www.geeksforgeeks.org/python-add-packages-to-anaconda-environment/) Use the link for more information on how to install packages/libraries to your Anaconda Environment.
5. Now, we can download the notebooks in the repository and run them on Jupyterlab/Jupyternotebook.



### Dataset
The datasets were obtained on Kaggle: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks. This dataset is updated regularly by the main contributor of that page. As such the current dataset that I have on hand will be behind compared to the one on the website. As a note, the dataset was retrieved on March.3,2024. 
The dataset contains 4 CSV files. The first three CSV files were retrieved from Kaggle and the GOOG CSV file was retrieved using an API from [Alpha Vantage](https://www.alphavantage.co/documentation/).

- GOOG.csv
    - `open`
    - `high`
    - `low`
    - `close`
    - `adjusted close`
    - `volume`
    - `Tomorrow_Adj_close`
    - `Target`

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

### Libraries Used
- [Numpy](https://numpy.org/doc/stable/index.html)
- [Pandas](https://pandas.pydata.org/docs/)
- [Tensorflow/Keras](https://www.tensorflow.org/guide/keras)
- [Matplotlib](https://matplotlib.org/stable/)
- [Scikit Learn](https://scikit-learn.org/stable/user_guide.html)
- 

### Project Flowchart 
    1. Open up the Stock_API Notebook
        a. Follow the markdown cells to set up your API keys and to understand the documentation for the API.
        b. Run the cells until you are able to save the file

    2. Open up the Stock Modeling Notebook
        a. Import the saved CSV file
        b. Preprocess the data frame and add the features that you desire.
        c. Run the models with your data frame and evaluate the results.
### Project Organization 
    - Cleaned_Dataset file contains the cleaned data frames
    - Presentations file contains all the Sprint Presentations
    - Stock Data API & Cleaning file contains the API and Stock data cleaning notebooks
    - Stock Modeling file contains the Jupyter files for running the machine learning models
    - README.md is a overview of the entire repository
    - app.py is the python file that creates a Streamlit app showcasing stock information for a specified stock
    - gitignore file tracks all the files that do not need to be updated or tracked
