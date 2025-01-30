import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import multiprocessing
from joblib import load
from testing import preprocess_text, split_complex_sentence

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")


# Load the vectorizer and model
tfidf_vectorizer_AE = load("models/tfidf_vectorizer_AE.pkl")
SVM_opinion_category_classifier = load("models/SVM_opinion_category_classifier.pkl")


#  Aspect and Sentiment analysis for a random text
def extract_aspects(sentence):
    """Extract aspects from a single sentence."""
    important = nlp(sentence)
    target = ''
    for token in important:
        if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
            target = token.text
        elif token.pos_ == 'NOUN':
            target = token.text
    return target if target else "restaurant"  # Default if no aspect found

def analyze_sentiment(sentence):
    """Analyze sentiment of a single sentence."""
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(sentence)
    if sentiment_scores['compound'] >= 0.05:
        return "positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

def absa_random(text):
    """Perform aspect-based sentiment analysis with multiprocessing."""
    
    # Breaking complex sentence into simple sentences
    simplified_sentences = split_complex_sentence(preprocess_text(text, 1))
    
    # Multiprocessing for Aspect Extraction
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        aspects = pool.map(extract_aspects, simplified_sentences)
    
    # Vectorize the extracted aspects using the pre-fitted TF-IDF Vectorizer
    aspects_tfidf = tfidf_vectorizer_AE.transform(aspects)

    # Predict the Opinion Category for each aspect
    predicted_aspects = SVM_opinion_category_classifier.predict(aspects_tfidf)

    # Multiprocessing for Sentiment Analysis
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        predicted_sentiments = pool.map(analyze_sentiment, simplified_sentences)
    
    # Return the results in a formatted structure
    result = "\n".join([f"{aspect},{category},{sentiment}"
                        for aspect, category, sentiment in zip(aspects, predicted_aspects, predicted_sentiments)])

    return result




if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python main.py <csv_filename>")
        sys.exit(1)

    data = sys.argv[1]
    resultcsv = absa_random(data)
    print(resultcsv) # Print the result CSV to the console