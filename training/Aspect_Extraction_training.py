
# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
df = pd.read_csv("datasets/training_dataset/Preprocessed_Restuarant_Dataset.csv")


# Split the data into training and testing sets
X1 = df['Opinion_Target']
y_category = df['Opinion_Category']

X1_train, X1_test, y_category_train, y_category_test = train_test_split(X1, y_category, test_size=0.2, random_state=42)


# Define a list of vectorization techniques and their names
vectorization_techniques1 = [
    ("TF-IDF", TfidfVectorizer(max_features=5000, sublinear_tf=True, stop_words='english')),
    ("Count Vectorization (BoW)", CountVectorizer(max_features=5000, stop_words='english'))
]


# Define a list of classifiers
classifiers1 = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=10, criterion="entropy"),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
    "Logistic Regression": LogisticRegression(random_state=0),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=0),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=0),
    "Bagging": BaggingClassifier(n_estimators=50, random_state=0),
    "Extra Trees": ExtraTreesClassifier(n_estimators=50, criterion="entropy", random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
}



 # Comparing different classifers with different vectorization techniques to predict Opinion_Category   
def compare_classifiers():
	results = []
	# Iterate through vectorization techniques and classifiers
	for technique_name, vectorizer in vectorization_techniques1:
		X1_train_vectorized = vectorizer.fit_transform(X1_train)
		X1_test_vectorized = vectorizer.transform(X1_test)

		for classifier_name, classifier in classifiers1.items():
			classifier.fit(X1_train_vectorized, y_category_train)
			y_category_pred = classifier.predict(X1_test_vectorized)
			category_accuracy = accuracy_score(y_category_test, y_category_pred)

			result = {
			"Technique": technique_name,
			"Classifier": classifier_name,
			"Accuracy": category_accuracy
		    }

			results.append(result)

	# Create a DataFrame from the results
	results_df = pd.DataFrame(results)

	# Pivot the DataFrame to form the result table
	pivot_table = pd.pivot_table(results_df, values='Accuracy',
			         index='Classifier', columns='Technique')

	# Fill NaN values with a placeholder (e.g., "N/A")
	pivot_table = pivot_table.fillna("N/A")

	return pivot_table


# Training  SVM classifier with TF-IDF vectorization

# # To ignore any warning while runtime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Text vectorization using TF-IDF
tfidf_vectorizer_AE = TfidfVectorizer(max_features=5000, sublinear_tf=True, stop_words='english')
X1_train_tfidf = tfidf_vectorizer_AE.fit_transform(X1_train)
X1_test_tfidf = tfidf_vectorizer_AE.transform(X1_test)

# Training  SVM classifier
SVM_opinion_category_classifier = SVC()
SVM_opinion_category_classifier.fit(X1_train_tfidf, y_category_train)


# Save the trained TF-IDF vectorizer
with open("pickle_files/tfidf_vectorizer_AE.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer_AE, f)

# Save the trained SVM model
with open("pickle_files/SVM_opinion_category_classifier.pkl", "wb") as f:
    pickle.dump(SVM_opinion_category_classifier, f)

#main

# Check if this script is being run as the main program
if __name__ == "__main__":
	
	#compare differ classifiers to predict Opinion Category
	pivot_table = compare_classifiers()
	
	# Display the result table
	print("Results Table:\n")
	print(pivot_table)

	print("\nUsing SVM classifier with TF-IDF vectorization")
   	
	# Predicting Opinion Category
	y_category_pred = SVM_opinion_category_classifier.predict(X1_test_tfidf)


	# Calculate accuracy and print classification report
	category_accuracy = accuracy_score(y_category_test, y_category_pred)
	category_report = classification_report(y_category_test, y_category_pred)

	print(f"\nOpinion Category Accuracy : {category_accuracy}")
	print("\nOpinion Category Classification Report:")
	print(category_report)




