koSRBERT 모델 구축 및 학습
이 프로젝트는 BERT 기반의 koSRBERT 모델을 구축하고, 이를 이용하여 NUG, MUR, DORN 태스크를 수행하는 과정을 포함합니다. 각 태스크별로 맞춤형 모델을 사용하며, 최종적으로 세 모델의 출력값을 통합하여 최종 분류를 수행합니다.

참고 문헌
doi = "10.18653/v1/2023.findings-eacl.184"

파일 구조
models.py: koSRBERT 모델 및 관련 태스크 모델(NUG, MUR, DORN) 정의
dataset.py: 데이터셋 정의 및 전처리 코드
train.py: koSRBERT 모델 학습 및 평가 코드
requirements.txt: 필요한 패키지 목록
README.md: 프로젝트 설명 및 사용법
설치 및 실행 방법
사전 요구사항
Python 3.7 이상
PyTorch 1.7 이상
Transformers 라이브러리
기타 필요 라이브러리는 requirements.txt 파일 참조
설치
가상환경 설정 (선택사항)

bash
코드 복사
python -m venv venv
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate  # Windows
필요 패키지 설치

bash
코드 복사
pip install -r requirements.txt
데이터 준비
데이터는 데이터프레임 형식이 사용되었으며, 다음과 같은 칼럼들로 구성되어 있습니다:
text: 채팅 상담 텍스트 데이터
role: 내담자 또는 상담자 역할 분류
order: 채팅 순서
session: 세션 ID
class: SR 또는 non-SR 클래스 분류 값
모델 학습 및 평가
데이터셋 전처리 및 로드

dataset.py 파일을 사용하여 데이터셋을 전처리하고 로드합니다.
모델 학습

train.py 파일을 실행하여 koSRBERT 모델을 학습시킵니다.
bash
코드 복사
python train.py
예제 코드
train.py 파일에 포함된 예제 코드를 통해 각 태스크(NUG, MUR, DORN) 모델 학습 및 koSRBERT 모델 통합 학습을 수행할 수 있습니다.
사용법
데이터셋을 준비하고, dataset.py를 사용하여 전처리합니다.
train.py 파일을 실행하여 모델을 학습합니다.
학습된 모델을 저장하고, 필요 시 평가 또는 예측에 사용합니다.
기여
이 프로젝트에 기여하고 싶다면, 이슈를 등록하거나 풀 리퀘스트를 제출해 주세요.
라이센스
이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 LICENSE 파일을 참고하세요.
연락처
문의사항이 있으면, [1995khi@gmial.com]로 연락 바랍니다.
