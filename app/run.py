import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from collections import Counter
from wordcloud import WordCloud
import base64
import io
import nltk
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    punctuation = set(string.punctuation)
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip(
    ) for tok in tokens if tok.lower() not in stop_words and tok not in punctuation]
    return clean_tokens


# Load data
engine = create_engine('sqlite:///../data/Response.db')
df = pd.read_sql_table('disaster_response', engine)

# Load model
model = joblib.load("../models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    # Extract data for visualizations

    # 1. Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # 2. Count of Different Categories
    category_counts = df.drop(columns=['id', 'message', 'original', 'genre']).sum(
    ).sort_values(ascending=False)
    category_names = list(category_counts.index)

    # 3. Top 10 Most Frequent Disaster Categories
    top_categories = category_counts.head(10)
    top_category_names = list(top_categories.index)

    # 4. Most Used Words in Messages
    all_messages = ' '.join(df['message'])
    tokens = tokenize(all_messages)
    word_counts = Counter(tokens)
    top_words = dict(word_counts.most_common(20))  # Top 20 words
    word_names = list(top_words.keys())
    word_counts_values = list(top_words.values())

    # Generate word cloud
    stop_words = set(stopwords.words("english"))
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          stopwords=stop_words).generate_from_frequencies(top_words)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    wordcloud_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    # Create visualizations
    graphs = [
        # Graph 1: Distribution of Message Genres
        {
            'data': [Bar(x=genre_names, y=genre_counts, marker=dict(color='rgb(55, 83, 109)'))],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        # Graph 2: Count of Different Categories
        {
            'data': [Bar(x=category_names, y=category_counts, marker=dict(color='rgb(26, 118, 255)'))],
            'layout': {
                'title': 'Count of Messages by Category',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        },
        # Graph 3: Top 10 Most Frequent Disaster Categories
        {
            'data': [Bar(x=top_category_names, y=top_categories, marker=dict(color='rgb(50, 171, 96)'))],
            'layout': {
                'title': 'Top 10 Most Frequent Disaster Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        },
        # Graph 4: Most Used Words in Messages
        {
            'data': [Bar(x=word_names, y=word_counts_values, marker=dict(color='rgb(222, 45, 38, 0.8)'))],
            'layout': {
                'title': 'Top 20 Most Used Words in Messages',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Word"}
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs and word cloud
    return render_template('master.html', ids=ids, graphJSON=graphJSON, wordcloud_base64=wordcloud_base64)


@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html page
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
