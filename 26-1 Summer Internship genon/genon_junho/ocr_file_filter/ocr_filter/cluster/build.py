"""[1] cluster: layout 레코드(페이지) → ViT 임베딩 → 2단계 K-Means.

    1단계(페이지수준): 페이지 이미지 전체를 임베딩해 n_clusters_page 개로 분류.
    2단계(요소수준):   각 페이지클러스터 안의 layout_elements(bbox 크롭)를 다시
                      n_clusters_element 개로 세분화 (전역 id 로 재부여).

table_src 레코드는 "페이지" 가 아니라 독립된 표 크롭이라 이 페이지→요소 계층 구조에
안 맞음 — 이번 데모는 layout 레코드만 대상으로 한다.

    python -m ocr_filter.cli cluster build --limit 500
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ocr_filter.cluster.embed import crop_element, embed_images
from ocr_filter.cluster.kmeans import two_stage_cluster
from ocr_filter.io.schema import Record, read_jsonl


@dataclass
class ClusterStats:
    n_pages: int
    n_elements: int
    n_page_clusters: int
    n_element_clusters: int


def build_clusters(
    unified_path: str | Path,
    out_path: str | Path,
    n_clusters_page: int = 64,
    n_clusters_element: int = 32,
    embed_model: str | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> ClusterStats:
    records = [r for r in read_jsonl(unified_path) if r.source_type == "layout"]
    if limit is not None:
        records = records[:limit]

    kwargs = {"model_name": embed_model} if embed_model else {}

    # 1단계: 페이지 임베딩
    page_embeddings = embed_images([r.image_path for r in records], **kwargs)

    # 2단계(요소수준)는 n_clusters_element<=0 이면 통째로 스킵한다 — GT 없는 레코드
    # (gt=None, 예: 신규 원본 PDF 배치)는 크롭할 layout_elements 자체가 없어서 원래도
    # 자동으로 빈 결과였지만, GT 있는 데이터에서도 taster/ddas/cmcv/report 전부
    # clusters.jsonl 의 level=="page" 만 읽고 level=="element" 는 아무 데서도 안 쓰므로
    # (현재 파이프라인에서는 죽은 산출물), 필요 없을 땐 크롭+임베딩 계산 자체를 아낀다.
    element_page_index: list[int] = []
    element_crops = []
    element_meta: list[dict] = []  # {page_id, element_index, category}
    if n_clusters_element > 0:
        # 모든 페이지의 layout_elements 를 (부모 페이지 인덱스, bbox 크롭) 로 펼침
        for page_idx, record in enumerate(records):
            elements = record.gt if isinstance(record.gt, list) else []
            for elem_idx, e in enumerate(elements):
                bbox = e.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                element_page_index.append(page_idx)
                element_crops.append(crop_element(record.image_path, bbox))
                element_meta.append({
                    "page_id": record.id, "element_index": elem_idx,
                    "category": e.get("category"),
                })

    element_embeddings = embed_images(element_crops, **kwargs) if element_crops else np.zeros((0, 768))

    page_clusters, element_clusters = two_stage_cluster(
        page_embeddings, element_embeddings, np.array(element_page_index, dtype=int),
        n_clusters_page, n_clusters_element, seed=seed,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record, pc in zip(records, page_clusters):
            f.write(json.dumps({
                "id": record.id, "level": "page", "cluster": int(pc),
            }, ensure_ascii=False) + "\n")
        for meta, ec in zip(element_meta, element_clusters):
            f.write(json.dumps({
                "id": f"{meta['page_id']}#{meta['element_index']}", "level": "element",
                "cluster": int(ec), "page_id": meta["page_id"], "category": meta["category"],
            }, ensure_ascii=False) + "\n")

    return ClusterStats(
        n_pages=len(records), n_elements=len(element_crops),
        n_page_clusters=len(set(page_clusters.tolist())),
        n_element_clusters=len(set(element_clusters.tolist())),
    )
