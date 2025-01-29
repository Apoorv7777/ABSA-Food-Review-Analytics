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

#Load the dataset
df = pd.read_csv("../datasets/training_dataset/Preprocessed_Restuarant_Dataset.csv")


# Split the data into training and testing sets
X = df['Token_Text']
y_polarity = df['Opinion_Polarity']

X_train, X_test, y_polarity_train, y_polarity_test = train_test_split(X, y_polarity, test_size=0.2, random_state=42) 

# Define a list of text vectorization techniques
vectorization_techniques = [
    ("TF-IDF", TfidfVectorizer(max_features=5000, sublinear_tf=True, stop_words='english')),
    ("Count Vectorization (BoW)", CountVectorizer(max_features=5000, stop_words='english'))
]

# Define a list of classifiers
classifiers = {
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

# Comparing different classifers with different vectorization techniques
# to predict Opinion_Polarity

def compare_classifiers():
    
    results = []
    # Iterate through vectorization techniques and classifiers
    for technique_name, vectorizer in vectorization_techniques:
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        for classifier_name, classifier in classifiers.items():
            classifier.fit(X_train_vectorized, y_polarity_train)
            y_polarity_pred = classifier.predict(X_test_vectorized)
            polarity_accuracy = accuracy_score(y_polarity_test, y_polarity_pred)
            if classifier_name =="Support Vector Machine":
                vectorc = classifier

            result = {
                "Technique": technique_name,
                "Classifier": classifier_name,
                "Accuracy": polarity_accuracy
            }

            results.append(result)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Pivot the DataFrame to form the table
    pivot_table = pd.pivot_table(results_df, values='Accuracy', 
                                index='Classifier', columns='Technique')

    # Fill NaN values with a placeholder (e.g., "N/A")
    pivot_table = pivot_table.fillna("N/A")

    return pivot_table
    


# Text vectorization using TF-IDF
tfidf_vectorizer_SA = TfidfVectorizer(max_features=5000, sublinear_tf=True, stop_words='english')
X_train_tfidf = tfidf_vectorizer_SA.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer_SA.transform(X_test)


# Train a SVM classifier 
svm_sentiment_classifier = SVC()
svm_sentiment_classifier.fit(X_train_tfidf, y_polarity_train)


# Save the trained TF-IDF vectorizer
with open("../models/tfidf_vectorizer_SA.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer_SA, f)

# Save the trained SVM model
with open("../models/svm_sentiment_classifier.pkl", "wb") as f:
    pickle.dump(svm_sentiment_classifier, f)


#main

if __name__ == "__main__":

	# compare different classifiers to predict Opinion Polarity
    pivot_table = compare_classifiers()
    
    print("Results Table:\n")  # Display the result table
    print(pivot_table)

    print("\nUsing SVM classifier with TF-IDF vectorization")
	
	# Predictions with the best classifier (Support Vector Classifier)
    y_polarity_pred = svm_sentiment_classifier.predict(X_test_tfidf)
	
	
	# Calculate accuracy and print classification report
    polarity_accuracy = accuracy_score(y_polarity_test, y_polarity_pred)
    polarity_report = classification_report(y_polarity_test, y_polarity_pred)

    print(f"\nOpinion Polarity Accuracy : {polarity_accuracy}")
    print("\nOpinion Polarity Classification Report:")
    print(polarity_report)

    



