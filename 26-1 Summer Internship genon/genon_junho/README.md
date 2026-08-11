필요있는 파일 목록
순서대로(필터링 및 자동 레이블링 파이프라인 -> 파인튜닝 데이터 확정 -> 파인튜닝 -> 평가)
1. /home/jhyeo/ocr_file_filter
- input은 /home/jhyeo/ocr_filter_result 여기에 두고 진행하였음.
- /home/jhyeo/ocr_file_filter/scripts/run_pipeline.sh를 활용하여 input및 output경로를 바꾸어 실행 가능


2. /home/jhyeo/finetuning
- /home/jhyeo/finetuning/finetuning_dataset
: 해당 파일에 파인튜닝할 목록을 train/val/test로 나눠서 저장하고 있으며 버전별로 진행중이다. 버전별로 변화 로직을 txt로 기록하고 있다.
    **2-1.데이터셋 시각화** -> fintuning 데이터셋 뿐아니라 eval을 위해 추론 json에 대해서도 나옴. 혹시 오류가 나면 이미지 파일이 잡히지 않은 경우이므로 원본 이미지 경로도 제일 좌단 박스에 절대경로로 붙여넣어야함.
    : /home/jhyeo/result_rendering/app.py를 포트에 서빙하여 띄우고 json파일의 절대경로를 붙여넣으면 bbox와 html렌더링 시각화하여 결과를 볼 수 있음.
- /home/jhyeo/finetuning/vlm
: 실제 파인튜닝 실행 코드(config/*.yaml, run_experiments.sh 등). jhshin/tsr_test/train/vlm 을 읽기전용으로 복사해온 것, 상세는 finetuning/README.md.


3. /home/jhyeo/finetuning/eval_318, /home/jhyeo/tsr_test
: dp500 및 tsr200 평가 파이프라인. **/home/jhyeo/tsr_test/instruction_set.md**의 명령어를 따라 실행해도 되지만 해당 파일 자체를 프롬프트화하여 만들었기 때문에 학습이 끝나면,
해당 md를 클로드 등에 태그하고 학습한 버전의 어느 ckpt들에 대해 평가돌려달라고 명령하면 같은 .md에 테이블로 업데이트됨.(기존 버전은 제일 잘 나온 것 하나만 남음)
- eval을 위한 추론 output은 /home/jhyeo/finetuning/vlm/eval_results 여기에 저장됨
- (참고) /home/jhyeo/tsr_eval 은 tsr_test 의 다른 목적 사본이라 이 파이프라인과 무관 — 헷갈리지 말 것