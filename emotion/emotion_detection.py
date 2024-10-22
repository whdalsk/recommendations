# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from imblearn.over_sampling import RandomOverSampler


# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Read Dataset
file_path = 'data/val.txt'
val_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])
file_path = 'data/test.txt'
test_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])
file_path = 'data/train.txt'
train_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])

train_df.info()

test_df.info()

val_df.info()

from collections import Counter

# Count occurrences
val_counts = Counter(val_df['Emotion'])
train_counts = Counter(train_df['Emotion'])
test_counts = Counter(test_df['Emotion'])

percentage_appearance_val = {k: v / len(val_df) * 100 for k, v in val_counts.items()}
percentage_appearance_train = {k: v / len(train_df) * 100 for k, v in train_counts.items()}
percentage_appearance_test = {k: v / len(test_df) * 100 for k, v in test_counts.items()}

print(percentage_appearance_val)
print(percentage_appearance_train)
print(percentage_appearance_test)

# Filter out 'surprise' and 'love' emotions using query()
val_df = val_df.query('Emotion != "surprise" and Emotion != "love"')
test_df = test_df.query('Emotion != "surprise" and Emotion != "love"')
train_df = train_df.query('Emotion != "surprise" and Emotion != "love"')

emotion_counts = train_df['Emotion'].value_counts()

print("Remaining Emotion Counts in train_df:")
print(emotion_counts)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text) # Remove non-alphabet characters
    text = text.lower() # Convert to lowercase
    text = text.split() # Split into words
    text = [word for word in text if word not in stop_words] # Remove stopwords
    return ' '.join(text)

train_df['Text'] = train_df['Text'].apply(preprocess_text)
val_df['Text'] = val_df['Text'].apply(preprocess_text)
test_df['Text'] = test_df['Text'].apply(preprocess_text)

vectorizer = CountVectorizer(max_features=3000)
svm = SVC(kernel="linear", C=0.5, random_state=42)
logistic = LogisticRegression(random_state=42, max_iter=1000)

svm_pipeline = Pipeline([("vectorizer", vectorizer), ("svm", svm)])
logistic_pipeline = Pipeline([("vectorizer", vectorizer), ("logistic", logistic)])

voting_classifier = VotingClassifier(
    estimators=[
        ("svm", svm_pipeline),
        ("logistic", logistic_pipeline)
    ],
    voting='hard'
)

voting_classifier.fit(train_df['Text'], train_df['Emotion'])

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Train Accuracy: {train_acc:.2f}")
    print(f"Validation Accuracy: {val_acc:.2f}")
    print(f"Test Accuracy: {test_acc:.2f}")
    
    return train_pred, val_pred, test_pred

# Evaluate the VotingClassifier
train_pred, val_pred, test_pred = evaluate_model(
    voting_classifier,
    train_df['Text'], train_df['Emotion'],
    val_df['Text'], val_df['Emotion'],
    test_df['Text'], test_df['Emotion']
)

train_true=train_df['Emotion']
val_true=val_df['Text']
test_true=test_df['Text']
labels=    voting_classifier.classes_

subset_test_df = test_df.sample(n=1000, random_state=42)  # Sample a subset for faster visualization

# Predictions for subset
subset_test_preds = voting_classifier.predict(subset_test_df['Text'])

# Plot confusion matrix for subset
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.heatmap(confusion_matrix(subset_test_df['Emotion'], subset_test_preds),
#             annot=True, fmt='d', xticklabels=labels, yticklabels=labels, ax=ax)
# ax.set_title('Subset Test Confusion Matrix')
# ax.set_xlabel('Predicted')
# ax.set_ylabel('True')
# plt.show()

test_acc = accuracy_score(test_df['Emotion'], test_pred)

print('The Model  has an accuracy of {:.2f}%'.format(test_acc * 100))

class_report = classification_report(test_df['Emotion'], test_pred, target_names=labels)

print(class_report)

# texts = [
#     "I'm feeling happy and excited today",
#     "I really don't even know why this is done for me!",
#     "I feel overwhelmed with sorrow",
#     "I was completely taken aback when everyone shouted 'Happy Birthday!'",
# ]

# for custom_text in texts:
#     processed_text = preprocess_text(custom_text)
#     predicted_emotion = voting_classifier.predict([processed_text])
#     print(f"Text: {custom_text}")
#     print(f"Predicted Emotion: {predicted_emotion[0]}")

# Translation
import googletrans

translator = googletrans.Translator()

input_text = ""
result = translator.translate(input_text, dest='en')
text = result.text

processed_text = preprocess_text(text)
predicted_emotion = voting_classifier.predict([processed_text])
print(f"Text: {text}")
print(f"Predicted Emotion: {predicted_emotion[0]}")