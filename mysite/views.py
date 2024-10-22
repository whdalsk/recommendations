from django.shortcuts import render
from django.http import HttpResponse

# views.py 문서 처리함수 작성 render(요청, html문서이름, 넘기는 값)
# urls.py 문서에 맵핑 views문서이름.함수

def index(request):
    print('def index(request)')
    return render(request, 'index.html')
    # templates 폴더 아래에 있는 index.html