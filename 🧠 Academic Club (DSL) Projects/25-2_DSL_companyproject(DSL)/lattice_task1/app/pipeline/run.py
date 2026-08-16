# app/pipeline/run.py
# -*- coding: utf-8 -*-
from app.pipeline.router import process

def run_on_file(pdf_path: str) -> dict:
    """1개 파일 실행 오케스트레이션."""
    return process(pdf_path)
