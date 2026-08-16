# -*- coding: utf-8 -*-
"""
문서 타입 프로파일 로더 및 추정기.
- profiles.yml을 읽어 제목(1페이지 상단)로 유형 추정
"""
from pathlib import Path
from typing import Dict, List
import yaml
from app.ingest.loader_pdf import TextBlock

_PROFILES_CACHE: Dict[str, dict] = {}

def load_profiles() -> Dict[str, dict]:
    global _PROFILES_CACHE
    if _PROFILES_CACHE:
        return _PROFILES_CACHE
    path = Path(__file__).parents[1] / "resources" / "profiles.yml"
    with open(path, "r", encoding="utf-8") as f:
        _PROFILES_CACHE = yaml.safe_load(f) or {}
    return _PROFILES_CACHE

def guess_profile(blocks: List[TextBlock], profiles: Dict[str, dict]) -> str:
    # 1페이지 상단 10줄 정도를 제목으로 가정
    title = " ".join(b.text for b in blocks if b.page == 1 and b.line_no <= 10)
    for name, prof in profiles.items():
        for tok in prof.get("title_good", []) or []:
            if tok and tok in title:
                return name
    return "default"
