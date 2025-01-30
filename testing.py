# Importing other files
import sys
import pandas as pd
import spacy
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load
from multiprocessing import Pool
import multiprocessing
import numpy as np
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Load the vectorizer and model
tfidf_vectorizer_AE = load("models/tfidf_vectorizer_AE.pkl")
SVM_opinion_category_classifier = load("models/SVM_opinion_category_classifier.pkl")
tfidf_vectorizer_SA = load("models/tfidf_vectorizer_SA.pkl")
svm_sentiment_classifier = load("models/svm_sentiment_classifier.pkl")



## Translate text into English
def translate(text):
	translator = GoogleTranslator(source='auto', target='en')
	return translator.translate(text)

# text1 = "बढ़िया खाना था"
# translate(text1)


## Handling Emojies

# Define emojis (expanded with your emojis)
positive_emojis = { "😄", "😀", "😁", "😆", "😂", "🤣", "😊", "🙂", "😎", "😉", "😍", "🥰", "😘", "😋", "😛", "😝", "😜", "🤪", "🤩", "🥳", "❤️", "👍", "👌", "🤟", "🍔", "🍕", "🍣", "🍰", "🍹", "🍷", "🍺", "🍦", "🍯", "🥞", "🍟", "🍩", "🥼", "🎉", "🎊", "🥳", "🍚", "🍘", "🍥", "🥠", "🥮", "🍢", "🍡", "🍧", "🍨", "🍦", "🥧", "🧁", "🍰", "🎂", "🍮", "🍭", "🍬", "🍫", "🍿", "🍩", "🍪", "🌰", "🥜", "🍯", "🍻", "🥂", "🍷", "🍾", "😇", "🤒", "👏"}
negative_emojis = { "😔", "😞", "😢", "😭", "😤", "😠", "😡", "🤬", "😩", "😫", "🥺", "😖", "😣", "😠", "😤", "😷", "🤒", "🤕", "😐", "😶", "😒", "😏", "🙁", "🥶", "😨", "😱", "😰", "😳", "🥵", "😳", "😵", "🤯", "🤐", "🤮", "🥴", "🤢", "👎", "😈", "👿", "💔", "😩", "😔", "😞", "😢", "😭" }

# Add space between word and emoji
def handle_emoji(sentence):
    # Regex pattern for positive and negative emojis
    emoji_pattern = re.compile(r'|'.join(map(re.escape, positive_emojis | negative_emojis)))
    
    # Replace emojis with respective labels
    def replace_emoji(match):
        emoji_char = match.group(0)
        if emoji_char in positive_emojis:
            return "good"
        elif emoji_char in negative_emojis:
            return "bad"
        return emoji_char  # This case should never occur as we handle all emojis

    # Replace emojis in sentence
    sentence = emoji_pattern.sub(replace_emoji, sentence)
    
    return sentence

## Data PreProcessing

# Data Cleaning
def clean_text(text):
    if text is not None:
        text = text.lower() # Lowercase the text
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)   # Remove special characters and punctuation
    return text

# Text preprocessing functions
def preprocess_text(text, state=0):
    if state == 1:
        text = translate(text)
    if text:
        text = handle_emoji(text)
        text = clean_text(text)
        doc = nlp(text)
        text = " ".join([token.lemma_ for token in doc if (not token.is_stop or token.text.lower()  =="not") and len(token.text) > 2])
        return text if len(text.split()) > 1 else ""
    return ""
    
    
### Function to tokenize text into sentences
def split_complex_sentence(texts):
    if isinstance(texts, str):  # If a single string is passed, convert it to a list
        texts = [texts]

    simpler_sentences = []
    for doc in nlp.pipe(texts, batch_size=100):  # Process texts in batches
        temp_sentence = []
        for token in doc:
            if token.dep_ == "cc":  # Coordinating conjunction (splitting criterion)
                if temp_sentence:
                    simpler_sentences.append(" ".join(temp_sentence))
                temp_sentence = []
            else:
                temp_sentence.append(token.text)
        if temp_sentence:
            simpler_sentences.append(" ".join(temp_sentence))
    return simpler_sentences


def parallel_apply(df, func):
    """Apply a function in parallel using multiprocessing."""
    num_chunks = multiprocessing.cpu_count()
    df_split = np.array_split(df, num_chunks)
    
    with multiprocessing.Pool(processes=num_chunks) as pool:
        df = pd.concat(pool.map(func, df_split))  # Apply function in parallel
    return df



# Ensure preprocess_text works on a dataframe partition
def preprocess_dataframe(df):
	df['Cleaned_Review'] = df['Review'].apply(preprocess_text)  # Apply text processing
	return df

# Aspect extraction function
def extract_aspect(sentence):
	# Use Spacy to process the sentence
	doc = nlp(sentence)
	
	target = ''
	temp_target = ''

	# Loop through tokens and extract the subject (noun)
	for token in doc:
		if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
			target = token.text
		if target == '' and token.pos_ == 'NOUN':
			temp_target = token.text

	if target == '': 
		target = temp_target

	return target
      
# Function to apply 'extract_aspect' row by row
def apply_extract_aspect(df):
	df['Opinion_Target'] = df['Review'].apply(extract_aspect)
	return df
      

def main(filename):

	input_df = pd.read_csv(filename)

	input_df['idx'] = range(10, (len(input_df) + 1)*10, 10)
	input_df.columns = ['Review', 'idx']

	### Converting every Review text into simple sentences and forming new dataframe as discussed earlier
	input_df['Simplified_Review'] = input_df['Review'].apply(lambda x: split_complex_sentence(x))
	input_df = input_df.explode('Simplified_Review')

	# Drop the original 'Review' column and rename the simplified column as 'Review'
	input_df = input_df.drop(columns=['Review']).rename(columns={'Simplified_Review': 'Review'})

	# Apply preprocessing in parallel
	input_df = parallel_apply(input_df, preprocess_dataframe)


	### Deleting rows where cleaned_review column is empty
	input_df = input_df[input_df['Cleaned_Review'] != ""]
	# input_df

	### Drop the custom index and reset to the default integer-based index
	input_df.reset_index(drop=True, inplace=True)

	## Rearranging the column order
	# Create a list of column names in the desired order
	desired_order = ['idx', 'Review', 'Cleaned_Review']

	# Rearrange the columns
	input_df = input_df[desired_order]


	# Apply aspect extraction in parallel
	input_df = parallel_apply(input_df, apply_extract_aspect)


	## Predict the Opinion_Category 

	# Vectorize the Opinion_Target using the  tfidf_vectorizer
	sample_tfidf = tfidf_vectorizer_AE.transform(input_df['Opinion_Target'])

	# Predicting the Opinion_Category 
	predicted_aspect_category = SVM_opinion_category_classifier.predict(sample_tfidf)

	# adding column Opinion_Category
	input_df['Opinion_Category'] = predicted_aspect_category

	# unique values of Opinion_Category 
	values = input_df['Opinion_Category'].value_counts()
	#print(values)

	### Handle missing values of Opinion_Target Column 
	# Opinion_Target = "" and Opinion_Category = "FOOD#QUALITY" then food of before # is Opinion_Target
	# Assumption just  to handle missing value
	def missing_value_handler(df):
		return df['Opinion_Category'].split('#')[0].lower() if pd.isna(df['Opinion_Target']) else df['Opinion_Target']

	input_df['Opinion_Target'] = input_df.apply(missing_value_handler, axis = 1)


	# Predict the Opinion_Polarity
	# Vectorize the sample sentence using the same tfidf_vectorizer
	# Note:- this function is defined in sentiment_analysis.ipynb notebook
	sample_tfidf = tfidf_vectorizer_SA.transform(input_df['Cleaned_Review'])

	# Predict the sentiment for the sample sentence
	Predicted_Opinion_Polarity = svm_sentiment_classifier.predict(sample_tfidf)

	# adding new column Opinion_Polarity in dataset
	input_df['Opinion_Polarity'] = Predicted_Opinion_Polarity

	return input_df

# **********************Charts ******************************

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_bar_graph(grouped, column_name, title, xlabel, ylabel, file_name, color='skyblue'):
    ax = grouped[column_name].plot(kind='bar', color=color, edgecolor='gray')
    # Add data labels
    for i, value in enumerate(grouped[column_name]):
        ax.text(i, value + 2, f'{value:.2f}%', ha='center', fontsize=10, fontweight='bold', color='black')

    # Set labels, title, and other customizations
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.grid(linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_stacked_bar_graph(grouped, columns, title, xlabel, ylabel, file_name):
    ax = grouped[columns].plot(kind='bar', stacked=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_name, format='png', dpi=300)
    plt.close()

def plot_doughnut_chart(grouped, file_name):
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightcyan']
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        grouped['Count'],
        labels=None,
        autopct='',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4)
    )
    legend_labels = [f'{category}: {percentage:.1f}%' for category, percentage in zip(grouped['Opinion_Category'], grouped['Percentage of Total Reviews'])]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Opinion_Category")
    plt.axis('equal')
    plt.title('Aspect Category Distribution (Doughnut Chart)')
    plt.tight_layout()
    plt.savefig(file_name, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_countplot(df, file_name):
    sns.set_palette("Set2")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x="Opinion_Polarity", data=df, order=df["Opinion_Polarity"].value_counts().index)
    plt.title("Sentiment Polarity Distribution for Reviews")
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Count")
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', fontweight='bold')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(file_name, format='png', dpi=300)
    plt.close()

def graphs(df):
    # Graph 1: Percentage of Positive Sentiment
    grouped = df.groupby(['Opinion_Category', 'Opinion_Polarity']).size().unstack(fill_value=0)
    grouped['Percentage of Positive Opinion_Polarity'] = (grouped['positive'] / (grouped['positive'] + grouped['negative'])) * 100
    plot_bar_graph(grouped, 'Percentage of Positive Opinion_Polarity', 'Percentage of Positive Sentiment based on Aspect Category',
                   'Opinion_Category', 'Percentage of Positive Sentiments (for Reviews)', 'static/output_graph_images/sample_plot1.png')

    # Graph 2: Percentage of Negative Sentiment
    grouped['Percentage of Negative Opinion_Polarity'] = (grouped['negative'] / (grouped['positive'] + grouped['negative'])) * 100
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=grouped.index, y=grouped['Percentage of Negative Opinion_Polarity'], hue=grouped.index, palette="Blues_d", legend=False)
    plt.title("Percentage of Negative Sentiment based on Aspect Category")
    plt.xlabel("Opinion_Category")
    plt.ylabel("Percentage of Negative Sentiments (for Reviews)")
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, fontweight='bold')
    ax.yaxis.grid(linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/output_graph_images/sample_plot2.png', format='png', dpi=300)
    plt.close()

    # Graph 3: Positive and Negative Sentiment Side by Side
    plot_stacked_bar_graph(grouped, ['Percentage of Positive Opinion_Polarity', 'Percentage of Negative Opinion_Polarity'],
                           'Percentage of Positive and Negative Sentiment based on Aspect Category',
                           'Aspect Category', 'Percentage', 'static/output_graph_images/sample_plot3.png')

    # Graph 4: Aspect Category Distribution (Doughnut Chart)
    grouped = df['Opinion_Category'].value_counts().reset_index()
    grouped.columns = ['Opinion_Category', 'Count']
    grouped['Percentage of Total Reviews'] = (grouped['Count'] / grouped['Count'].sum()) * 100
    plot_doughnut_chart(grouped, 'static/output_graph_images/sample_plot4.png')

    # Graph 5: Sentiment Polarity Distribution
    plot_countplot(df, 'static/output_graph_images/sample_plot5.png')


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":

	if len(sys.argv) != 2:
		print("Usage: python main.py <csv_filename>")
		sys.exit(1)

	data = sys.argv[1]
	
	if data.lower().endswith('.csv'):
		print(f'Uploaded File: {data}')
		
		input_df = main(data)
		graphs(input_df)
	
		
		
		
