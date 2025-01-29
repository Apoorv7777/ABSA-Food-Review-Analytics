# Importing other files
import sys
import pandas as pd
import spacy
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from deep_translator import GoogleTranslator
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle

# Setting NLTK resources
# Define the NLTK data path
NLTK_DATA_PATH = os.path.expanduser("~/nltk_data")  # Change this path if needed

# Ensure the NLTK data directory exists
os.makedirs(NLTK_DATA_PATH, exist_ok=True)

# Function to check if a resource exists and download if missing
def download_nltk_resource(resource, subfolder="corpora"):
    resource_path = os.path.join(NLTK_DATA_PATH, subfolder, resource)
    if not os.path.exists(resource_path):
        print(f"Downloading {resource}...")
        nltk.download(resource, download_dir=NLTK_DATA_PATH)
    # else:
    #     print(f"{resource} already exists, skipping download.")

# Check and download required NLTK resources
download_nltk_resource("stopwords.zip")
download_nltk_resource("punkt.zip", "tokenizers")
download_nltk_resource("vader_lexicon.zip", "sentiment")

# Set NLTK's data path
nltk.data.path.append(NLTK_DATA_PATH)

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



# Load the vectorizer and model
with open("models/tfidf_vectorizer_AE.pkl", "rb") as f:
    tfidf_vectorizer_AE = pickle.load(f)

with open("models/SVM_opinion_category_classifier.pkl", "rb") as f:
    SVM_opinion_category_classifier = pickle.load(f)

with open("models/tfidf_vectorizer_SA.pkl", "rb") as f:
    tfidf_vectorizer_SA = pickle.load(f)

with open("models/svm_sentiment_classifier.pkl", "rb") as f:
    svm_sentiment_classifier = pickle.load(f)


## Translate text into English
def translate(text):
	translator = GoogleTranslator(source='auto', target='en')
	return translator.translate(text)

# text1 = "बढ़िया खाना था"
# translate(text1)


## Handling Emojies

positive_emojis = """😄😀😁😆😂🤣😊🙂😎😉😍🥰😘😋😛😝😜🤪🤩🥳🥰❤️👍👌🤟🍔🍕🍣🍰🍹🍷🍺
	            🍦🍯🥞🍟🍩🥼🎉🎊🥳🍚🍘🍥🥠🥮🍢🍡🍧🍨🍦🥧🧁🍰🎂🍮🍭🍬🍫🍿🍩🍪🌰🥜
	            🍯🍻🥂🍷🍾😀😃😄😆😅🥲\😊😇🙂😀😃😄😁😅😆😂🙂😊🤒👏"""

    
negative_emojis = """😔😞😢😭😤😠😡🤬😩😫🥺😖😣😠😤😷🤒🤕😐😶😒😏🙁🥶😨😱😰😳🥵😳😵
	             🤯🤐🤮🥴🤢👎😈👿💔😩😔😞😢😭"""

# add space between word and emoji
def handle_emoji_helper(text):
	i = 0
	lst = list(text)
	for word in text:
		if word != " " and (word in positive_emojis or word in negative_emojis):
			lst.insert(i," ")
		i+=1
	return "".join(lst)

def handle_emoji(sentence):
	sentence = handle_emoji_helper(sentence)
	words = sentence.split()
	converted_sentence = []
	for word in words:
		if word in positive_emojis:
			converted_sentence.append("good") # Don't change ( one word sentences will be removed in
		elif word in negative_emojis:                                     #      preprocessing )
			converted_sentence.append("bad")
		else:
			converted_sentence.append(word)
	return " ".join(converted_sentence)

# # Example usage:
# input_sentence = "😀"
# input_sentence = "बढ़िया खाना था😀"
# converted = handle_emoji(input_sentence)
# print(converted)


## Data PreProcessing

# Data Cleaning
def clean_text(text):
	if text is not None:
		# Lowercase the text
		text = text.lower()
		# Remove special characters and punctuation
		text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
	return text

# Text preprocessing functions
def preprocess_text(text, state = 0):
	text = handle_emoji(text)
	if state == 1:
		text = translate(text)
	if text is not None:
		if state == 0:
			text = clean_text(text)
			text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if (word.lower() not in stop_words or word.lower() == 'not') and len(word)>2])
			text = text if len(text.split()) > 1 else ""    
		
	return text
    
    
### Function to tokenize text into sentences
def split_complex_sentence(text):
	doc = nlp(text) # Parse the text
	simpler_sentences = []
	temp_sentence = []
    
	# Iterate through the parsed tokens
	for token in doc:
		# Check for coordinating conjunctions (e.g., "and," "but") to split the sentence
		if token.dep_ == "cc":
			if temp_sentence:
				simpler_sentences.append(" ".join(temp_sentence))
			temp_sentence = []
		else:
			temp_sentence.append(token.text)

	if temp_sentence:
		simpler_sentences.append(" ".join(temp_sentence))

	sentences = []
	for sentence in simpler_sentences:
		sentences.append(sent_tokenize(sentence))

	flattened_sentences = [sentence for sublist in sentences for sentence in sublist]
	return flattened_sentences



def main(filename):

	input_df = pd.read_csv(filename)

	input_df['idx'] = range(10, (len(input_df) + 1)*10, 10)
	input_df.columns = ['Review', 'idx']

	### Converting every Review text into simple sentences and forming new dataframe as discussed earlier
	new_df_list = []  # Create a list to store DataFrames

	for i, (idx, text) in enumerate(zip(input_df['idx'], input_df['Review'])):
		sentences = split_complex_sentence(text)  # split_complex_sentence defined in the start
		sentences_df = pd.DataFrame({'Review': sentences, 'idx': range(idx, idx + len(sentences))})
		new_df_list.append(sentences_df)

	new_df = pd.concat(new_df_list, ignore_index=True)  # Concatenate all DataFrames at once



	### Replacing old dataset by new dataset
	input_df = new_df.copy()
	# input_df.head() # print dataset head


	### Preprocessing Review to create new column cleaned_review
	from tqdm import tqdm

	tqdm.pandas()

	# df_temp['y'] = df_temp.ratings.progress_map(label)
	input_df['Cleaned_Review'] = input_df['Review'].progress_map(preprocess_text)
	#input_df.head()


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


	# Opinion Target (Aspect) Extraction
	# An empty list for obtaining the extracted aspects from sentences.  
	aspects =[]

	sentences = input_df['Review']
	sentences = sentences.astype(str)
	# Performing Aspect Extraction 
	for sen in sentences: 
		important = nlp(sen)  # Fix the variable name 'sen'
		descriptive_item = '' 
		target = '' 
		temp_target = ''
		for token in important: 
			if token.dep_ == 'nsubj' and token.pos_ == 'NOUN': 
				target = token.text   
			if target =='' and token.pos_ == 'NOUN': 
				temp_target = token.text 

		if target == '': target = temp_target
		aspects.append(target)

	# creating a dataset having sinlge column = Opinion_Target
	new_column_df = pd.DataFrame({'Opinion_Target' : aspects})

	# Concatenate new column ( Opinion_Target ) into input_df dataset horizontally (along columns)
	input_df = pd.concat([input_df, new_column_df], axis=1)
	# input_df.head()


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

def graphs(df):

	# Graph : 1
	# Group the DataFrame by 'Opinion_Category' and 'Opinion_Polarity' and count occurrences
	grouped = df.groupby(['Opinion_Category', 'Opinion_Polarity']).size().unstack(fill_value=0)

	# Calculate the percentage of positive Opinion_Polarity for each Opinion_Category
	grouped['Percentage of Positive Opinion_Polarity'] = (grouped['positive'] / (grouped['positive'] + grouped['negative'])) * 100

	# Set a custom color palette
	sns.set_palette("pastel")

	# Create a bar plot with a nicer color palette
	ax = grouped['Percentage of Positive Opinion_Polarity'].plot(kind='bar', color='skyblue', edgecolor='gray')

	# Add data labels on top of each bar
	for i, percentage in enumerate(grouped['Percentage of Positive Opinion_Polarity']):
		ax.text(i, percentage + 2, f'{percentage:.2f}%', ha='center', fontsize=10, fontweight='bold', color='black')

	# Set the labels and title
	ax.set_xlabel('Opinion_Category')
	ax.set_ylabel('Percentage of Positive Sentiments (for Reviews)')
	plt.title('Percentage of Positive Sentiment based on Aspect Category')

	# Customize the grid lines
	ax.yaxis.grid(linestyle='--', alpha=0.6)

	# Rotate the x-axis labels for better readability
	plt.xticks(rotation=45)

	# Increase the figure size for better aesthetics
	plt.gcf().set_size_inches(10, 6)

	# Save the plot as an image with higher resolution and adjust bbox_inches
	plt.savefig('static/output_graph_images/sample_plot1.png', format='png', dpi=300, bbox_inches='tight')

	# Show the plot
	#plt.tight_layout()
	#plt.show()




	# Graph : 2
	# Group the DataFrame by 'Opinion_Category' and 'Opinion_Polarity' and count occurrences
	grouped = df.groupby(['Opinion_Category', 'Opinion_Polarity']).size().unstack(fill_value=0)

	# Calculate the percentage of negative Opinion_Polarity for each Opinion_Category
	grouped['Percentage of Negative Opinion_Polarity'] = (grouped['negative'] / (grouped['positive'] + grouped['negative'])) * 100

	# Set a custom color palette
	sns.set_palette("pastel")

	# Create a bar plot with improved styling
	plt.figure(figsize=(10, 6))  # Adjust the figure size
	ax = sns.barplot(x=grouped.index, y=grouped['Percentage of Negative Opinion_Polarity'],hue=grouped.index, palette="Blues_d",legend=False)

	# Customize the plot's appearance
	plt.title("Percentage of Negative Sentiment based on Aspect Category")
	plt.xlabel("Opinion_Category")
	plt.ylabel("Percentage of Negative Sentiments (for Reviews)")

	# Rotate the x-axis labels for better readability
	plt.xticks(rotation=45)

	# Add data labels on top of each bar
	for p in ax.patches:
		ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
				ha='center', va='center', fontsize=10, fontweight='bold')

	# Add a background grid for better visual separation
	ax.yaxis.grid(linestyle='--', alpha=0.6)

	# Increase the spacing between the plot and the x-axis label
	plt.tight_layout()

	# Save the plot as an image with higher resolution
	plt.savefig('static/output_graph_images/sample_plot2.png', format='png', dpi=300)

	# Show the plot
	#plt.show()



	# Graph : 3
	# Group the DataFrame by 'Opinion_Category' and 'Opinion_Polarity' and count occurrences
	grouped = input_df.groupby(['Opinion_Category', 'Opinion_Polarity']).size().unstack(fill_value=0)

	# Calculate the percentage of positive and negative Opinion_Polarity for each Opinion_Category
	grouped['Percentage of Positive Opinion_Polarity'] = (grouped['positive'] / (grouped['positive'] + grouped['negative'])) * 100
	grouped['Percentage of Negative Opinion_Polarity'] = (grouped['negative'] / (grouped['positive'] + grouped['negative'])) * 100

	# Plot the percentages of positive and negative Opinion_Polarity for each Opinion_Category
	ax = grouped[['Percentage of Positive Opinion_Polarity', 'Percentage of Negative Opinion_Polarity']].plot(kind='bar', color=['skyblue', 'salmon'])
	ax.set_xlabel('Aspect Category')
	ax.set_ylabel('Percentage')
	plt.title('Percentage of Positive and Negative Sentiment based on Aspect Category')
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.savefig('static/output_graph_images/sample_plot3.png',format='png', dpi=300)
	#plt.show()
	

	# Graph : 4
	# Group the DataFrame by 'Opinion_Category' and count occurrences
	grouped = input_df['Opinion_Category'].value_counts().reset_index()
	grouped.columns = ['Opinion_Category', 'Count']

	# Calculate the percentage of total reviews for each category
	grouped['Percentage of Total Reviews'] = (grouped['Count'] / grouped['Count'].sum()) * 100

	# Set colors for the doughnut chart
	colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightcyan']

	# Create a doughnut chart for Opinion_Category Distribution
	fig, ax = plt.subplots()

	# Plot the pie chart (outer circle)
	wedges, texts, autotexts = ax.pie(
	    grouped['Count'],
	    labels=None,  # No labels on slices
	    autopct='',
	    startangle=140,
	    colors=colors,
	    wedgeprops=dict(width=0.4)  # Set the width to create a doughnut chart
	)

	# Add a legend with color-coded labels and percentages in a table
	legend_labels = [f'{category}: {percentage:.1f}%' for category, percentage in zip(grouped['Opinion_Category'], grouped['Percentage of Total Reviews'])]
	ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5), title="Opinion_Category")

	plt.axis('equal')
	plt.title('Aspect Category Distribution (Doughnut Chart)')

	plt.tight_layout()

	# Save the plot as an image with higher resolution and adjust bbox_inches
	plt.savefig('static/output_graph_images/sample_plot4.png', format='png', dpi=300, bbox_inches='tight')

	# Show the plot
	#plt.show()

	# Graph : 5
	# Set a custom color palette for the plot
	sns.set_palette("Set2")

	# Create a countplot using Seaborn
	plt.figure(figsize=(8, 6))  # Adjust the figure size
	sns.countplot(x="Opinion_Polarity", data=df, order=df["Opinion_Polarity"].value_counts().index)

	# Customize the plot's appearance
	plt.title("Sentiment Polarity Distribution for reviews")
	plt.xlabel("Sentiment Polarity")
	plt.ylabel("Count")

	# Add data labels on top of each bar
	ax = plt.gca()
	for p in ax.patches:
		ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
		        ha='center', va='center', fontsize=12, color='black', fontweight='bold')

	# Rotate x-axis labels for better readability
	plt.xticks(rotation=0)

	# Add a background grid for better visual separation
	plt.grid(axis='y', linestyle='--', alpha=0.7)

	# Increase the spacing between the plot and the x-axis label
	plt.tight_layout()

	# Save the plot as an image with higher resolution
	plt.savefig('static/output_graph_images/sample_plot5.png', format='png', dpi=300)

	# Show the plot
	#plt.show()



### -----------------------xxxxxxxxxxxxxxxxxxxxxxxxxxxxX-----------------------------

# Aspect and Sentiment analysis for a random text
def absa_random(text):
	### Text
	#text = "The pizza is delicious but service is terrible. The atmosphere is also not good.😀"
	# text = "😀"
	# text = "बढ़िया खाना था😀"  # Hindi Language


	### Breaking complex sentence into simple sentences
	simplified_sentences = split_complex_sentence(preprocess_text(text,1))
	#print(simplified_sentences)

	### Opinion Target(Aspect) Extraction
	# An empty list for obtaining the extracted aspects from sentences.  
	ext_aspects = [] 
	aspects =[]

	# Performing Aspect Extraction 
	for sen in simplified_sentences: 
		important = nlp(sen)  # Fix the variable name 'sen'
		descriptive_item = '' 
		target = '' 
		for token in important: 
			if token.dep_ == 'nsubj' and token.pos_ == 'NOUN': 
				target = token.text
			elif token.pos_ == 'NOUN': 
				target = token.text
		if target == "": target = "restaurant"
		aspects.append(target)

	######print("\nASPECTS : ",aspects)


	### Predict the Opinion_Category

	# Vectorize the sample sentence using the same tfidf_vectorizer
	sample_tfidf = tfidf_vectorizer_AE.transform(aspects)

	# Predict the sentiment for the sample sentence
	predicted_aspect = SVM_opinion_category_classifier.predict(sample_tfidf)

	#########print(f"Predicted Aspect Category: {predicted_aspect}")


	### Predict the Opinion_Polarity

	# Create a SentimentIntensityAnalyzer object
	sid = SentimentIntensityAnalyzer()

	# Preprocess the sample sentences
	preprocessed_sample_sentences = [preprocess_text(sentence,1) for sentence in simplified_sentences]

	predicted_sentiments = []
	# Analyze sentiment
	for sentence in preprocessed_sample_sentences:
		sentiment_scores = sid.polarity_scores(sentence)
		# Determine the sentiment label
		if sentiment_scores['compound'] >= 0.05:
			sentiment_label = "positive"
		# else : sentiment_label = 'negative'
		elif sentiment_scores['compound'] <= -0.05:
			sentiment_label = "negative"
		else:
			sentiment_label = "neutral"

		predicted_sentiments.append(sentiment_label)
	#     print(f"Text: {sentence}")
	#     print(f"Compound Score: {sentiment_scores['compound']}")
	#######print(f"Predicted Sentiment: {predicted_sentiments}")


	'''## Resultant Table: 
	result_df = pd.DataFrame({"Aspect_Term" : aspects,
		                  "Aspect_Category" : predicted_aspect,
		                  "Sentiment" : predicted_sentiments})
	#######print("\n"result_df)
	return result_df'''
	
	res = zip(aspects,predicted_aspect,predicted_sentiments)
	rows=[]
	for i,j,k in res:
		rows.append(",".join([i,j,k]))
	sentence_info = "\n".join(rows)
	
	return sentence_info


if __name__ == "__main__":

	if len(sys.argv) != 2:
		print("Usage: python main.py <csv_filename>")
		sys.exit(1)

	data = sys.argv[1]
	
	if data.lower().endswith('.csv'):
		print(f'Uploaded File: {data}')
		
		input_df = main(data)
		graphs(input_df)
	else:
		# Handle text data
		result_csv = absa_random(data)
		print(result_csv) # Print the result CSV to the console
		
		
		
		
