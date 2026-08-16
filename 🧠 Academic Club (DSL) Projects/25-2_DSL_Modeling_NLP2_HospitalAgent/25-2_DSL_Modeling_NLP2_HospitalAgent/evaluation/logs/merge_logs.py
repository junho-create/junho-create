import os
import json
import glob

# 1. 로그 파일이 있는 디렉토리 경로를 지정합니다.
# 사용자의 경로에 맞게 설정되었습니다.
log_directory = r'C:\USERS\ASAP0\ONEDRIVE\바탕 화면\연세대학교\25-2 DSL\25-2_DSL_MODELING_NLP2_HOSPITALAGENT\evaluation\logs'

# 2. 결과 파일을 저장할 경로와 파일명을 지정합니다.
# logs 폴더의 상위 폴더인 evaluation에 저장됩니다.
output_file_path = os.path.join(os.path.dirname(log_directory), 'combined_logs.json')

# 3. 모든 로그 데이터를 저장할 빈 리스트를 생성합니다.
combined_data = []

# 4. log_directory 안의 모든 .json 파일을 찾습니다.
json_files = glob.glob(os.path.join(log_directory, '*.json'))

# 5. 각 JSON 파일을 순서대로 읽습니다.
for file_path in json_files:
    try:
        # UTF-8 인코딩으로 파일을 엽니다. (한글 등 깨짐 방지)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 파일 내용이 리스트 형태이면 extend로 합치고,
            if isinstance(data, list):
                combined_data.extend(data)
            # 단일 객체이면 append로 추가합니다.
            else:
                combined_data.append(data)
                
    except json.JSONDecodeError:
        print(f"경고: '{os.path.basename(file_path)}' 파일이 비어있거나 유효한 JSON 형식이 아닙니다. 건너뜁니다.")
    except Exception as e:
        print(f"'{os.path.basename(file_path)}' 처리 중 오류 발생: {e}")

# 6. 합쳐진 데이터를 새로운 JSON 파일에 저장합니다.
with open(output_file_path, 'w', encoding='utf-8') as f:
    # ensure_ascii=False: 한글이 깨지지 않도록 설정
    # indent=4: 사람이 보기 좋게 4칸 들여쓰기 적용
    json.dump(combined_data, f, ensure_ascii=False, indent=4)

print(f"완료! 총 {len(json_files)}개의 JSON 파일을 합쳐서 '{output_file_path}'에 저장했습니다.")