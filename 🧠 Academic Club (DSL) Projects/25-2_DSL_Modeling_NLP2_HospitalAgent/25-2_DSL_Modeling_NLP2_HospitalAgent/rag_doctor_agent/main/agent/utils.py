import re
import unicodedata

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("\u200b", " ").strip().lower()
    return t

def normalize_text_preserve_symbols(text: str) -> str:
    """기호를 보존하면서 텍스트를 정규화합니다."""
    if text is None:
        return ""
    # 기본적인 공백 정리만 수행, 기호는 보존
    t = str(text).replace("\u200b", " ").strip()
    return t

def normalize_keyboard_symbols_only(text: str) -> str:
    """자판에 있는 일반적인 기호가 아닌 모든 특수 유니코드 기호를 하이픈(-)으로 대체합니다."""
    if text is None:
        return ""
    
    t = str(text).strip()
    
    # 자판에 있는 일반적인 기호들 (ASCII 범위) - 이것들은 보존
    keyboard_symbols = set("!@#$%^&*()-+={}[]|\\:;\"'<>,.?/~`")
    
    # 한글, 영문, 숫자, 공백, 자판 기호가 아닌 모든 문자를 하이픈으로 대체
    result = ""
    for char in t:
        if (char.isalnum() or  # 영문, 숫자
            '\u1100' <= char <= '\u11FF' or  # 한글 자모
            '\u3130' <= char <= '\u318F' or  # 한글 호환 자모  
            '\uAC00' <= char <= '\uD7AF' or  # 한글 완성형
            char.isspace() or  # 공백
            char in keyboard_symbols):  # 자판 기호
            result += char
        else:
            result += "-"  # 모든 특수 유니코드 기호를 하이픈으로 대체
    
    # 연속된 하이픈을 하나로 정리
    import re
    result = re.sub(r'-+', '-', result)
    
    return result


def translate_symbols_to_text(text: str) -> str:
    """특수기호를 의미가 있는 텍스트로 변환합니다."""
    if text is None:
        return ""
    
    # 먼저 인식하지 못하는 기호들을 하이픈으로 정규화
    t = normalize_keyboard_symbols_only(text)
    original = t
    
    # 직함 우선순위 관련 기호들 (화살표 + 하이픈)
    priority_symbols = ["→", "➜", "⇒", ">>", "≫", "-"]
    has_priority_symbol = any(symbol in t for symbol in priority_symbols)
    has_title = any(title in t for title in ["원장", "교수", "전문의", "센터장", "전임의", "임상강사"])
    
    if has_priority_symbol:
        if has_title:
            # 우선순위 기호들을 "다음"으로 변환
            for symbol in ["→", "➜", "⇒", ">>", "≫", "-"]:
                t = t.replace(symbol, " 다음 ")
            t += " 순으로"
    
    # 매핑 관련 화살표들
    elif any(symbol in t for symbol in ["->", "⟶", "⇨"]):
        for symbol in ["->", "⟶", "⇨"]:
            t = t.replace(symbol, "은 ")
        t += "로 연결"
    
    # 가중치 설정 관련 기호들
    elif "=" in t and any(char.isdigit() or char == "." for char in t):
        t = t.replace("=", " 가중치를 ") + "로 설정"
    
    elif ":" in t and any(char.isdigit() or char == "." for char in t.split(":")[-1]):
        t = t.replace(":", " 가중치를 ") + "로 설정"
    
    # 일반 매핑 (콜론)
    elif ":" in t:
        t = t.replace(":", "은 ") + "로 연결"
    
    # 여러 공백을 하나로 정리
    t = " ".join(t.split())
    
    return t

def tokenize_ko_en(text: str):
    t = normalize_text(text)
    tokens = re.findall(r"[A-Za-z]+|[가-힣]+|\d+", t)
    return tokens

def uniq_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out
