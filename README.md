# 감정 분석 음악 추천 | Django

![1](https://github.com/user-attachments/assets/f43bda31-56a4-4e55-acac-6b8a5a1239a0)

<br>

## 프로젝트 소개

감정 분석 모델을 이용하여 나온 감정을 기반으로 음악을 추천하는 프로그램을 만들어보았습니다.

<br>

## 개발 언어

![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS-239120?&style=for-the-badge&logo=css3&logoColor=white)
![js](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=white)
![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)

<br>

## 개발 기간

2024.07.01 ~ 2024.07.14

<br>

## 멤버 및 역할

### 김종민(조장)
  - 모델 분석 및 성능 강화
  - 백엔드 개발
  - 음악 추천 기능 개발

### 김동연(조장)
  - 모델 분석 및 성능 강화
  - 백엔드 개발

### 양정윤
  - 프론트앤드 개발

### 이채은
  - 프론트앤드 개발

### 정은정
  - 프론트앤드 개발

<br>

## 기능

### 메인페이지
  - 점속하면 나오는 메인페이지이다.
  - 고양이 얼굴이 좌우로 움직인다.

<br>

![2](https://github.com/user-attachments/assets/cfa36fe8-c53b-4e10-a6d1-26741c2e6db0)

<br>

### 일기
  - 우측 상단의 today 버튼을 누르면 이동한다.

<br>

![3](https://github.com/user-attachments/assets/fede3171-c0ff-42ff-b83e-4aba928f2fe0)

<br>

  - 좌측 부분에 오늘의 기분을 적을 수 있다.

<br>

![4](https://github.com/user-attachments/assets/8c590438-3ace-4d9c-8016-e6a7c0ab468c)

<br>

### 음악 추천
  - 연필 버튼을 누르면 우측 페이지가 생성된다.
  - 입력한 내용이 원고지 형식으로 출력된다.
  - 입력한 내용을 분석해 나온 감정으로 음악을 추천해준다.

<br>

![5](https://github.com/user-attachments/assets/f43bda31-56a4-4e55-acac-6b8a5a1239a0)

<br>

## 소감
처음에 주제를 정하는 데에 시간을 많이 소비한 것 같아 아쉽다. 기대한 것에 비해 내용은 많이 없지만, 모델을 사용해보았다는 것이
좋은 경험이 되었다. 한 문장을 입력하면 감정 분석 정확도가 높게 나왔지만, 여러 문장을 한번에 입력할 시에 정확도가 낮게 나왔다.
그래서, 여러 문장을 한 문장씩 모델에 삽입하여 평균값을 최종 결과값으로 도출하니까 정확도가 높게 나왔다.
