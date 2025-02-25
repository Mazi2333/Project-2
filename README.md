# Project-2 Disaster Response
# Disaster Response Pipeline Project

## Project Overview
This project is part of the Udacity Data Scientist Nanodegree Program. The goal is to build a Natural Language Processing (NLP) model to categorize messages from real-life disaster events. The project includes an ETL pipeline for data processing, a machine learning pipeline for message classification, and a web application for real-time message categorization.

## File Structure
- **app**
  - `run.py` - Flask file to run the web application.
  - **templates**
    - `master.html` - Main page of the web application.
    - `go.html` - Classification result page of the web application.
- **data**
  - `disaster_messages.csv` - Dataset containing disaster-related messages.
  - `disaster_categories.csv` - Dataset containing categories for the messages.
  - `process_data.py` - ETL pipeline script to clean and store data in a SQLite database.
- **models**
  - `train_classifier.py` - Machine learning pipeline script to train and save the classifier.
- `README.md` - Project documentation.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/disaster-response-pipeline.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Instructions
1. **Run the ETL pipeline** to clean and store data:
    ```bash
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```
2. **Run the ML pipeline** to train and save the classifier:
    ```bash
    python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```
3. **Run the web app**:
    ```bash
    python app/run.py
    ```
4. Open the web app in your browser:
    ```
    http://localhost:3001
    ```

## Project Components
1. **ETL Pipeline**: Extracts data from CSV files, cleans the data, and stores it in a SQLite database.
2. **ML Pipeline**: Loads data from the SQLite database, trains a classifier, and saves the model as a pickle file.
3. **Web Application**: A Flask web app that takes user input messages and returns classification results.

## Libraries Used
- Python
- NumPy
- Pandas
- Scikit-Learn
- NLTK
- SQLAlchemy
- Flask
- Plotly

## Acknowledgements
This project was completed as part of the Udacity Data Scientist Nanodegree Program in collaboration with Figure Eight.
