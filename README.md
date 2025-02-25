# Project-3 Recommender IBM
# IBM Watson Studio Recommender System Project

## Project Overview
This project is part of the Udacity Data Scientist Nanodegree Program. The goal is to build a recommendation system using data from the IBM Watson Studio platform. The project involves exploratory data analysis, rank-based recommendations, user-user collaborative filtering, content-based recommendations, and matrix factorization.

## File Structure
- **data**
  - `user-item-interactions.csv` - Dataset containing user interactions with articles.
  - `articles_community.csv` - Dataset containing article information.
- **notebooks**
  - `Recommendations_with_IBM.ipynb` - Jupyter notebook containing the project code and analysis.
- `README.md` - Project documentation.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ibm-recommender-system.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Instructions
1. **Run the Jupyter notebook** to execute the project code and analysis:
    ```bash
    jupyter notebook notebooks/Recommendations_with_IBM.ipynb
    ```

## Project Components
1. **Exploratory Data Analysis (EDA)**: Explore the dataset to understand its structure and key characteristics.
2. **Rank-Based Recommendations**: Recommend the most popular articles based on the number of interactions.
3. **User-User Collaborative Filtering**: Find users with similar interaction patterns and recommend articles that similar users have interacted with.
4. **Content-Based Recommendations**: Use Natural Language Processing (NLP) to recommend articles based on content similarity.
5. **Matrix Factorization**: Use Singular Value Decomposition (SVD) to predict user-article interactions and make recommendations.

## Libraries Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- NLTK
- Matplotlib

## Acknowledgements
This project was completed as part of the Udacity Data Scientist Nanodegree Program in collaboration with IBM Watson Studio.
