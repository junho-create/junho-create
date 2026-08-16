from typing import List, Dict, Tuple
from .utils import normalize_text

SYMPTOM_CANON = {
    "두통": ["headache", "지끈지끈", "편두통", "migraine", "tension headache"],
    "편두통": ["migraine", "headache", "광과민성", "photophobia", "오심", "nausea"],
    "지끈지끈": ["throbbing", "pulsating", "headache"],
    "메스꺼움": ["nausea", "오심"],
    "속 울렁거림": ["nausea", "queasy", "오심"],
    "소화불량": ["indigestion", "dyspepsia"],
    "명치 통증": ["epigastric pain", "upper abdominal pain", "epigastralgia"],
    "명치": ["epigastric", "epigastrium"],
}

DEPT_SYNONYMS = {
    "신경과": ["neurology", "neuro", "신경과학"],
    "신경외과": ["neurosurgery"],
    "소화기내과": ["gastroenterology", "gi", "hepatology"],
    "내과": ["internal medicine", "im"],
    "가정의학과": ["family medicine", "fm"],
    "응급의학과": ["emergency medicine", "er"],
}

TITLE_NORMALIZE = {
    "대표원장": ["대표원장", "대표", "ceo", "chief director"],
    "센터장": ["센터장", "center head", "director of center", "내과센터장", "신경센터장"],
    "교수": ["교수", "professor", "associate professor", "assistant professor"],
    "원장": ["원장", "director"],
    "부원장": ["부원장", "vice director", "부센터장"],
    "전문의": ["전문의", "board-certified"],
    "전임의": ["전임의", "fellow"],
    "임상강사": ["임상강사", "clinical instructor"],
}

def expand_symptoms(symptoms: List[str]) -> List[str]:
    augmented = []
    for s in symptoms:
        s0 = s.strip()
        augmented.append(s0)
        s_norm = normalize_text(s0)
        for k, vals in SYMPTOM_CANON.items():
            if normalize_text(k) in s_norm or any(normalize_text(v) in s_norm for v in vals):
                augmented.extend([k] + vals)
    seen = set(); out = []
    for x in augmented:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def expand_dept(dept: str) -> List[str]:
    dnorm = normalize_text(dept)
    out = [dept]
    for k, vals in DEPT_SYNONYMS.items():
        if normalize_text(k) == dnorm or dnorm in [normalize_text(v) for v in vals]:
            out.extend([k] + vals)
    return list(dict.fromkeys(out))

def canonical_title(title: str) -> str:
    tnorm = normalize_text(title)
    for k, vals in TITLE_NORMALIZE.items():
        if normalize_text(k) == tnorm or tnorm in [normalize_text(v) for v in vals]:
            return k
    for k, vals in TITLE_NORMALIZE.items():
        for v in vals:
            if normalize_text(v) in tnorm:
                return k
    return title.strip()
