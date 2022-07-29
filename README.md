# kang-gui
korean ai novel generator

# 설치 방법

1. python3 설치
[https://www.python.org/downloads/](https://www.python.org/downloads/)

2. 패키지 설치(패키지 설치.bat)
```sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -q transformers~=4.12.0
pip install fastai
```

3. app 실행(app.bat)
```sh
python app.py
```

# app 설명

## 출력창
마지막 줄 문장을 input해서 나온 텍스트를 뒤에 붙이는 방식.
그 위에 있는 문장은 input에 영향은 없음.

## generate 버튼
공백을 제외한 출력창의 마지막 줄 문장을 input값으로
generator 실행

## init save.json
save.json의 내용을
```json
{
    "input": "",
    "output": "",
    "result": ""
}
```
으로 초기화

ps. 간혹 멈추면 cmd가 일시정지해서 그런 걸수도 있으니 cmd에서 Enter 몇 번 누르면 됨.

# 파인튜닝(fineTunning.py)
```sh
python fineTunning.py <start index> <end index>
```
raw 폴더에 있는 txt파일 중 start~(end-1) 까지 튜닝시킴.

* index는 0번째 부터 시작함. (ex) python fineTunning.py 0 5)
* txt 파일의 인코딩 형식은 utf-8로 통일(다른 인코딩 형식은 무시됨.)
* RAM 사용량이 많이 필요한 작업이니 개인 컴퓨터 보다는 colab에서 돌리는 것을 추천.  
[kang_FineTuning.ipynb](https://colab.research.google.com/drive/1H3MDfWQTBsMd__Szz7byFJ0nurHp9Y_-?usp=sharing)

# colab
[colab project link](https://drive.google.com/drive/folders/1_lwPOVnlnSVfekPzwttxKJkc17DZ6FhC?usp=sharing)  
본인 구글 드라이브에 사본 만들기 하여 사용.

* 실행시 본인 구글 드라이브 접근 동의 필요.
* 절대경로 사용해서 '/gdrive/MyDrive/kang-colab/'에 맞추거나 소스 수정해야함.
* 파인튜닝 시 메뉴에서 '런타임-런타임 유형 변경-하드웨어 가속기 gpu'로 설정 추천
* 하드웨어 가속기 사용 시 gpu_activate = True로 설졍해야함.
* 방식은 똑같음. raw폴더 참조해서 txt 파일들로 model.pt 학습시키는 방식.
* raw에 들어간 txt파일이 많으면 model = fineTunningLoop(main_dir+"raw/",model,0,5)에서 숫자 부분을 수정하면 됨.

# external source

* [pytorch](https://github.com/pytorch/pytorch)
* [KoGPT2](https://github.com/SKT-AI/KoGPT2)
* [transformers](https://github.com/huggingface/transformers)
* [fastai](https://github.com/fastai/fastai)