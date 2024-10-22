from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import googletrans
from symspellpy_ko import KoSymSpell, Verbosity
from flask import Flask, request, jsonify
import json
import requests
from kss import split_sentences
from pykospacing import Spacing
import random

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Read Dataset
file_path = './static/data/val.txt'
val_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])
file_path = './static/data/test.txt'
test_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])
file_path = './static/data/train.txt'
train_df = pd.read_csv(file_path, sep=';', header=None, names=['Text', 'Emotion'])

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
print(train_df['Emotion'].values)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text) # Remove non-alphabet characters
    text = re.sub(r'\s{2,}', ' ', text) # 2칸 이상 공백 1칸으로 변경
    text = text.lower() # Convert to lowercase
    text = text.split() # Split into words
    text = [word for word in text if word not in stop_words] # Remove stopwords
    return ' '.join(text)

def correct_spacing(text):
    spacing = Spacing()
    corrected_text = spacing(text)
    return corrected_text

train_df['Text'] = train_df['Text'].apply(preprocess_text)
val_df['Text'] = val_df['Text'].apply(preprocess_text)
test_df['Text'] = test_df['Text'].apply(preprocess_text)

from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()
vectorizer = CountVectorizer(max_features=3000)
svm = SVC(kernel="linear", C=0.5, random_state=42)
logistic = LogisticRegression(random_state=42, max_iter=1000)

# svm_pipeline = Pipeline([("vectorizer", vectorizer), ("svm", svm)])
# 텍스트 데이터를 TF-IDF로 벡터화하고 naive_bayes 모델 학습
naive_bayes_pipeline = Pipeline([("vectorizer", vectorizer), ("naive_bayes", naive_bayes)])
logistic_pipeline = Pipeline([("vectorizer", vectorizer), ("logistic", logistic)])

voting_classifier = VotingClassifier(
    estimators=[
        ("naive_bayes", naive_bayes_pipeline),
        ("logistic", logistic_pipeline)
    ],
    voting='soft'
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
labels=voting_classifier.classes_

subset_test_df = test_df.sample(n=1000, random_state=42)  # Sample a subset for faster visualization

# Predictions for subset
subset_test_preds = voting_classifier.predict(subset_test_df['Text'])

test_acc = accuracy_score(test_df['Emotion'], test_pred)

print('The Model  has an accuracy of {:.2f}%'.format(test_acc * 100))

class_report = classification_report(test_df['Emotion'], test_pred, target_names=labels)

print(class_report)

# translator = googletrans.Translator()

# Kakao Translation
def kor2eng(query):
    url = "https://translate.kakao.com/translator/translate.json"

    headers = {
        "Referer": "https://translate.kakao.com/",
        "User-Agent": "Mozilla/5.0"
    }

    data = {
        "queryLanguage": "kr",
        "resultLanguage": "en",
        "q": query
    }

    resp = requests.post(url, headers=headers, data=data)
    data = resp.json()
    output = data['result']['output'][0][0]
    return output

def page1(request):
    return render(request, 'emotion/page1.html')

def page2(request):
    return render(request, 'emotion/page2.html')

def page3(request):
    return render(request, 'emotion/page3.html')

# Input
# def input(request):
#     return render(request, 'emotion/input.html')

#텍스트처리
app = Flask(__name__)

@app.route('/result.do', methods=['POST'])
def result(request):
    if request.method == 'POST':
        # POST 요청에서 JSON 데이터를 파싱합니다.
        data = json.loads(request.body)
        input_text = data.get('input', '')

        # Fix Typo
        sym_spell = KoSymSpell()
        sym_spell.load_korean_dictionary(decompose_korean=True, load_bigrams=True)

        fixed_text = ''
        # for suggestion in sym_spell.lookup_compound(input_text, max_edit_distance=2):
        #     fixed_text = suggestion.term

        # result = translator.translate(fixed_text, dest='en')
        # text = result.text

        texts = split_sentences(input_text)
        angry = []
        fear = []
        joy = []
        sad = []
        for text in texts: 
            # print(i)
            e_text = kor2eng(text)
            processed_text = preprocess_text(e_text)
            predicted_emotion_proba = voting_classifier.predict_proba([processed_text])
            print(f'predicted_emotion_proba  {text}  {predicted_emotion_proba}')
            print('-------------------------------')
            # print(predicted_emotion_proba[0][0])
            angry.append(predicted_emotion_proba[0][0])
            fear.append(predicted_emotion_proba[0][1])
            joy.append(predicted_emotion_proba[0][2])
            sad.append(predicted_emotion_proba[0][3])
            
        angry_mean = sum(angry)/len(angry)
        fear_mean = sum(fear)/len(fear)
        joy_mean = sum(joy)/len(joy)
        sad_mean = sum(sad)/len(sad)
        arr =[[angry_mean, fear_mean, joy_mean, sad_mean]]

        print('predict_proba 감정 :', max(arr[0]))
        print('anger :', angry_mean)
        print('fear :', fear_mean)
        print('joy :', joy_mean)
        print('sad :', sad_mean)
        predicted_emotion = voting_classifier.predict([processed_text])
        print(f"Input Text: {input_text}")
        print(f'Fixed Text: {e_text}')
        print(f"Text: {text}")
        print(f"Predicted Emotion: {predicted_emotion[0]}")

        joy_song = ['APT. - 로제 (ROSÉ), Bruno Mars','Love Lee - AKMU (악뮤)','파이팅 해야지 (Feat. 이영지) - 부석순 (SEVENTEEN)',
                    'Welcome to the Show - DAY6(데이식스)','Supernova - aespa','첫 만남은 계획대로 되지 않아 - TWS (투어스)']
        sad_song = ['헤어지자 말해요 - 박재정','운이 좋았지 - 권진아','미련하다 - 로이킴',
                    '너였다면 - 정승환', '묘해, 너와 - 어쿠스틱 콜라보', '거리에서 - 성시경']
        fear_song = ['戀人 (연인) - 박효신','바람 - 최유리','도망가자 - 선우정아',
                     '김철수 씨 이야기 - 허회경','보통의 하루 - 정승환','봄이 와도 - 로이킴']
        angry_song = ['I don\'t care - 2NE1','그건 니 생각이고 - 장기하와 얼굴들','품행제로 - 블락비 바스타즈',
                      '삐딱하게 (Crooked) - G-DRAGON', 'Shut down - BLACKPINK']

        if predicted_emotion[0] == 'joy':
            result='기분이 좋으시군요!'
            res_song=random.choice(joy_song)
        elif predicted_emotion[0] == 'sadness':
            result='슬프시군요'
            res_song=random.choice(sad_song)
        elif predicted_emotion[0] == 'fear':
            result='두려우시군요.'
            res_song=random.choice(fear_song)
        elif predicted_emotion[0] == 'anger':
            result='화가 나시는군요.'
            res_song=random.choice(angry_song)
        else:
            result='판단할 수 없습니다.'         

        response_data = {
            'result': result,
            'res_song':res_song,
            'input': correct_spacing(input_text)
        }
        return JsonResponse(response_data)

    else:
        return JsonResponse({'error': 'Invalid request method'}, status=400)
    
if __name__ == '__main__':
        app.run(debug=True)