"""ocr_file_filter 통합 CLI.

    python -m ocr_filter.cli models download          # HF 모델 받기
    python -m ocr_filter.cli models bundle out.tar.gz # 폐쇄망 이동용 tar
    python -m ocr_filter.cli models serve             # vLLM 3개 기동
    python -m ocr_filter.cli models serve --dry-run   # 실행될 명령만 출력
    python -m ocr_filter.cli models status            # 헬스체크
    python -m ocr_filter.cli models stop              # 종료

    python -m ocr_filter.cli io build                 # 원본 3소스 → ./_work/unified.jsonl
    python -m ocr_filter.cli cmcv run --limit 20       # 3모델 CMCV 채점 (일부만 시험)
    python -m ocr_filter.cli cmcv run                  # 전체 실행 (오래 걸림, 이어하기 가능)
    python -m ocr_filter.cli cluster build --limit 500 # ViT 임베딩 + 2단계 K-Means (venv/bin/python3 로 실행)
    python -m ocr_filter.cli report gallery            # cmcv 결과 HTML 갤러리 생성

옵션 --only 로 특정 모델만 대상 지정 (예: --only external_a external_b).
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MANIFEST = str(Path(__file__).resolve().parents[1] / "configs" / "models.yaml")
DEFAULT_CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


def _cmd_models(args: argparse.Namespace) -> None:
    from ocr_filter.models.download import bundle, download_all
    from ocr_filter.models.manifest import load_manifest
    from ocr_filter.models.serve import serve_all, status_all, stop_all

    manifest = load_manifest(args.manifest)
    only = args.only or None

    if args.action == "download":
        download_all(manifest, only=only)
    elif args.action == "bundle":
        bundle(manifest, args.out)
    elif args.action == "serve":
        serve_all(manifest, only=only, dry_run=args.dry_run)
    elif args.action == "status":
        status_all(manifest)
    elif args.action == "stop":
        stop_all(manifest, only=only)


def _cmd_io(args: argparse.Namespace) -> None:
    import yaml

    from ocr_filter.io import build_unified, build_unified_from_images, build_unified_from_pdfs

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    if args.action == "build":
        out_path = args.out or str(Path(paths["work_dir"]) / "unified.jsonl")
        stats = build_unified(paths, out_path)
        print(f"layout: {stats.n_layout}  table: {stats.n_table}  "
              f"total: {stats.n_total}  이미지 없음: {stats.n_missing_images}")
        print(f"→ {out_path}")
    elif args.action == "build-raw":
        if not args.input_dir:
            raise SystemExit("--input-dir 필요 (원본 PDF 들어있는 폴더, 재귀 탐색)")
        out_path = args.out or str(Path(paths["work_dir"]) / "unified.jsonl")
        images_out = args.images_out or str(Path(paths["work_dir"]) / "images")
        n = build_unified_from_pdfs(args.input_dir, images_out, out_path, dpi=args.dpi)
        print(f"PDF → 페이지 {n}건 (GT 없음, source_type=layout 고정)")
        print(f"→ {out_path}")
    elif args.action == "build-images":
        if not args.input_dir:
            raise SystemExit("--input-dir 필요 (원본 jpg/png 들어있는 폴더, 재귀 탐색)")
        out_path = args.out or str(Path(paths["work_dir"]) / "unified.jsonl")
        n = build_unified_from_images(args.input_dir, out_path)
        print(f"이미지 → {n}건 (GT 없음, source_type=layout 고정, 렌더링 없음/원본 경로 그대로)")
        print(f"→ {out_path}")


def _cmd_cmcv(args: argparse.Namespace) -> None:
    import yaml

    from ocr_filter.cmcv import run_cmcv

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    models_cfg = cfg["models"]
    thresholds = cfg.get("cmcv", {}).get("match_thresholds", {})

    if args.action == "run":
        unified_path = args.unified or str(Path(paths["work_dir"]) / "unified.jsonl")
        out_path = args.out or str(Path(paths["work_dir"]) / "cmcv_results.jsonl")
        ids = None
        if args.ids_file:
            ids = {line.strip() for line in Path(args.ids_file).read_text(encoding="utf-8")
                   .splitlines() if line.strip()}
        stats = run_cmcv(
            unified_path, out_path, models_cfg,
            limit=args.limit, source_type=args.source_type, ids=ids, workers=args.workers,
            agree_min=args.agree_min if args.agree_min is not None
            else thresholds.get("agree_min", 0.85),
            text_min=args.text_min if args.text_min is not None
            else thresholds.get("text_min", 0.90),
        )
        print(f"완료: {stats}")
        print(f"→ {out_path}")


def _cmd_rescue(args: argparse.Namespace) -> None:
    import yaml

    from ocr_filter.cmcv.rescue import run_rescue
    from ocr_filter.io.schema import read_jsonl

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    unified_path = args.unified or str(Path(paths["work_dir"]) / "unified.jsonl")
    cmcv_results = args.cmcv_results or str(Path(paths["work_dir"]) / "cmcv_results.jsonl")
    out_path = args.out or str(Path(paths["work_dir"]) / "rescue_results.jsonl")

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    id_to_path = {r.id: r.image_path for r in read_jsonl(unified_path)}
    _size_cache: dict[str, tuple[int, int] | None] = {}

    def id_to_image_size(rid: str) -> tuple[int, int] | None:
        if rid not in _size_cache:
            p = id_to_path.get(rid)
            if not p:
                _size_cache[rid] = None
            else:
                try:
                    with Image.open(p) as im:
                        _size_cache[rid] = im.size
                except Exception:
                    _size_cache[rid] = None
        return _size_cache[rid]

    stats = run_rescue(
        cmcv_results, out_path, id_to_image_size,
        tiers=tuple(args.tiers), iou_min=args.iou_min, text_min=args.text_min,
        coverage_min=args.coverage_min, limit=args.limit,
    )
    print(f"완료: {stats}")
    print(f"→ {out_path}")


def _cmd_cluster(args: argparse.Namespace) -> None:
    import yaml

    from ocr_filter.cluster import build_clusters

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    cluster_cfg = cfg.get("cluster", {})

    if args.action == "build":
        unified_path = args.unified or str(Path(paths["work_dir"]) / "unified.jsonl")
        out_path = args.out or str(Path(paths["work_dir"]) / "clusters.jsonl")
        stats = build_clusters(
            unified_path, out_path,
            n_clusters_page=args.n_clusters_page if args.n_clusters_page is not None
            else cluster_cfg.get("n_clusters_page", 64),
            n_clusters_element=args.n_clusters_element if args.n_clusters_element is not None
            else cluster_cfg.get("n_clusters_element", 32),
            embed_model=args.embed_model,
            limit=args.limit,
        )
        print(f"완료: {stats}")
        print(f"→ {out_path}")


def _cmd_ddas(args: argparse.Namespace) -> None:
    import yaml

    from ocr_filter.ddas import (
        select_additional_ids,
        select_final_ids,
        select_top_clusters_by_real_rate,
        write_selected_ids,
    )

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    if args.action == "select":
        taster_report = args.taster_report or str(Path(paths["work_dir"]) / "taster_report.json")
        clusters_path = args.clusters or str(Path(paths["work_dir"]) / "clusters.jsonl")
        out_path = args.out or str(Path(paths["work_dir"]) / "ddas_selected_ids.txt")
        selected = select_final_ids(taster_report, clusters_path,
                                     with_replacement=args.with_replacement,
                                     n_per_cluster=args.n_per_cluster)
        n = write_selected_ids(selected, out_path)
        print(f"클러스터 {len(selected)}개에서 총 {n}건 선택 → {out_path}")
        print(f"다음: python -m ocr_filter.cli cmcv run --ids-file {out_path}")
    elif args.action == "select-additional":
        clusters_path = args.clusters or str(Path(paths["work_dir"]) / "clusters.jsonl")
        cmcv_results = args.cmcv_results or str(Path(paths["work_dir"]) / "cmcv_results.jsonl")
        already_ids_path = args.already_ids or str(Path(paths["work_dir"]) / "ddas_selected_ids.txt")
        out_path = args.out or str(Path(paths["work_dir"]) / "ddas_selected_ids_additional.txt")
        already_selected = {
            line.strip() for line in Path(already_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        selected = select_additional_ids(
            cmcv_results, clusters_path, already_selected, n_total=args.n_total,
            alpha=args.ddas_alpha, beta=args.ddas_beta, seed=args.seed,
        )
        n = write_selected_ids(selected, out_path)
        print(f"실측 hard 비율 기준, 기존 {len(already_selected)}건 제외하고 "
              f"클러스터 {len(selected)}개에서 총 {n}건 추가 선택 → {out_path}")
        print(f"다음: python -m ocr_filter.cli cmcv run --ids-file {out_path}")
    elif args.action == "select-top-clusters":
        clusters_path = args.clusters or str(Path(paths["work_dir"]) / "clusters.jsonl")
        cmcv_results = args.cmcv_results or str(Path(paths["work_dir"]) / "cmcv_results.jsonl")
        already_ids_path = args.already_ids or str(Path(paths["work_dir"]) / "ddas_selected_ids.txt")
        out_path = args.out or str(Path(paths["work_dir"]) / "ddas_selected_ids_top_clusters.txt")
        already_selected = {
            line.strip() for line in Path(already_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        selected = select_top_clusters_by_real_rate(
            cmcv_results, clusters_path, already_selected,
            top_n=args.top_n, min_samples=args.min_samples,
        )
        n = write_selected_ids(selected, out_path)
        print(f"실측 medium+hard 비율 상위 {args.top_n}개 클러스터의 남은 풀 전체, "
              f"기존 {len(already_selected)}건 제외하고 {len(selected)}개 클러스터에서 총 {n}건 선택 → {out_path}")
        print(f"다음: python -m ocr_filter.cli cmcv run --ids-file {out_path}")


def _cmd_taster(args: argparse.Namespace) -> None:
    import yaml

    from ocr_filter.taster import run_taster

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    models_cfg = cfg["models"]
    thresholds = cfg.get("cmcv", {}).get("match_thresholds", {})
    ddas_cfg = cfg.get("ddas", {})
    n_per_cluster = args.n_per_cluster or cfg.get("cluster", {}).get("taster_per_cluster", 8)

    if args.action == "run":
        clusters_path = args.clusters or str(Path(paths["work_dir"]) / "clusters.jsonl")
        unified_path = args.unified or str(Path(paths["work_dir"]) / "unified.jsonl")
        cmcv_out = args.cmcv_out or str(Path(paths["work_dir"]) / "taster_cmcv_results.jsonl")
        result = run_taster(
            clusters_path, unified_path, cmcv_out, models_cfg,
            n_per_cluster=n_per_cluster, workers=args.workers,
            agree_min=args.agree_min if args.agree_min is not None
            else thresholds.get("agree_min", 0.85),
            text_min=args.text_min if args.text_min is not None
            else thresholds.get("text_min", 0.90),
            ddas_alpha=ddas_cfg.get("alpha", 1.0), ddas_beta=ddas_cfg.get("beta", 2.0),
            ddas_n_total=args.ddas_n_total,
        )
        print(f"클러스터 수: {result['n_clusters']}, taster 샘플: {result['n_taster_samples']}")
        print(f"cmcv_stats: {result['cmcv_stats']}")
        print("\n클러스터별 S_i / |C_i| / 할당 N_i (상위 10개):")
        for cid in sorted(result['scores'], key=lambda c: -result['scores'][c])[:10]:
            print(f"  cluster {cid:>4}  S_i={result['scores'][cid]:.3f}  "
                  f"|C_i|={result['sizes'][cid]:>4}  N_i={result['alloc'][cid]:>4}  "
                  f"counts={result['tier_counts'][cid]}")
        out_path = args.out or str(Path(paths["work_dir"]) / "taster_report.json")
        import json
        Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {out_path}")


def _cmd_hardcase(args: argparse.Namespace) -> None:
    import yaml

    from ocr_filter.hardcase import run_both, run_judge, run_prelabel

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    models_cfg = cfg["models"]
    hc_cfg = cfg.get("hardcase", {})
    judge_cfg = hc_cfg.get("judge_vlm", {})
    prelabel_cfg = hc_cfg.get("prelabel_vlm", judge_cfg)
    max_rounds = hc_cfg.get("max_refine_rounds", 3)

    unified_path = args.unified or str(Path(paths["work_dir"]) / "unified.jsonl")
    cmcv_results = args.cmcv_results or str(Path(paths["work_dir"]) / "cmcv_results.jsonl")
    judge_out = args.judge_out or str(Path(paths["work_dir"]) / "hardcase_judge.jsonl")
    prelabel_out = args.prelabel_out or str(Path(paths["work_dir"]) / "hardcase_prelabel.jsonl")

    if args.action == "judge":
        stats = run_judge(unified_path, cmcv_results, judge_out, models_cfg, judge_cfg,
                           max_rounds=max_rounds, workers=args.workers, limit=args.limit)
        print(f"완료: {stats}\n→ {judge_out}")
    elif args.action == "prelabel":
        stats = run_prelabel(judge_out, prelabel_out, prelabel_cfg,
                              workers=args.workers, limit=args.limit)
        print(f"완료: {stats}\n→ {prelabel_out}")
    elif args.action == "run":
        stats = run_both(unified_path, cmcv_results, judge_out, prelabel_out,
                          models_cfg, judge_cfg, prelabel_cfg,
                          max_rounds=max_rounds, workers=args.workers, limit=args.limit)
        print(f"완료: {stats}\n→ {judge_out}\n→ {prelabel_out}")
    elif args.action == "export":
        import json as _json

        from ocr_filter.hardcase.export_labeler import run_export
        # labeler 가 요구하는 config 포맷 생성 (로컬 judge 모델을 evaluator 로 재사용).
        # judge(Qwen3.5-397B) 는 thinking 모델이라 max_tokens 기본값(4096)으로 부르면
        # 사고 도중 잘려서 빈 응답이 남 (ocr_filter/hardcase 에서 이미 겪은 문제와 동일).
        # VLMClient 의 `reasoning_enabled` 는 OpenRouter 전용 포맷(`extra_body.reasoning.enabled`)
        # 이라 우리 로컬 vLLM(Qwen3 계열)에는 안 먹힘 — vLLM 은 `chat_template_kwargs.enable_thinking`
        # 이 정식 키라서 `extra_params` 로 직접 그 형태를 실어 보내야 실제로 사고를 끈다
        # (2026-07-13 실측: reasoning_enabled=False 만 줬을 때 여전히 파싱 실패 재현·확인함).
        labeler_evaluator_cfg = {
            "model": judge_cfg.get("name", "Qwen3.5-397B-A17B-NVFP4"),
            "base_url": judge_cfg.get("endpoint", "http://localhost:8004/v1"),
            "api_key_env": "DUMMY_KEY",
            "max_tokens": 16384,
            "extra_params": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        }
        labeler_cfg = {
            "evaluation": {"mode": "single"},
            "evaluator_vlm_primary": labeler_evaluator_cfg,
            "converter_vlm": labeler_evaluator_cfg,  # pipeline.py 가 vlm_client 를 강제 초기화하므로 더미로 전달
            "processing": {
                "pipeline_mode": "full",
                "parallel_workers": args.workers,
                "skip_already_passed": True,
            },
            "renderer": {
                # channel 생략 — playwright install chromium 만 돼있음(chrome 채널 없음).
                "browser": "chromium",
                "headless": True,
                "viewport_width": 1400,
                "viewport_height": 1800,
                "wait_after_load_ms": 1000,
                "full_page_screenshot": True,
                "device_scale_factor": 2,
            },
            "annotation": {
                "bbox_colors": {"Text": "#3498db", "Title": "#e74c3c", "Table": "#2ecc71"}
            }
        }
        export_out = args.export_out or str(Path(paths["work_dir"]) / "labeler_export")
        run_export(prelabel_out, export_out, export_out, labeler_cfg)
        summary_path = Path(export_out) / "summary_report.json"
        if summary_path.exists():
            summary = _json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
            print(f"완료: total={summary['total_pages']} passed={summary['passed']} "
                  f"failed={summary['failed']} errors={summary['errors']}")
        else:
            print("완료(처리 대상 0건 — hardcase_prelabel.jsonl 에 unresolved 레코드가 있는지 확인)")
        print(f"→ {export_out}/review.html")


def _cmd_report(args: argparse.Namespace) -> None:
    import yaml

    from ocr_filter.report import build_html
    from ocr_filter.report.cluster_gallery import build_cluster_gallery_html

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    if args.action == "gallery":
        unified_path = args.unified or str(Path(paths["work_dir"]) / "unified.jsonl")
        cmcv_results = args.cmcv_results or str(Path(paths["work_dir"]) / "cmcv_results.jsonl")
        out_path = args.out or str(Path(paths["work_dir"]) / "gallery.html")
        page = build_html(unified_path, cmcv_results, per_tier=args.per_tier)
        Path(out_path).write_text(page, encoding="utf-8")
        print(f"→ {out_path}")
    elif args.action == "cluster-gallery":
        unified_path = args.unified or str(Path(paths["work_dir"]) / "unified.jsonl")
        taster_results = args.taster_results or str(Path(paths["work_dir"]) / "taster_cmcv_results.jsonl")
        clusters_path = args.clusters or str(Path(paths["work_dir"]) / "clusters.jsonl")
        taster_report = args.taster_report or str(Path(paths["work_dir"]) / "taster_report.json")
        out_path = args.out or str(Path(paths["work_dir"]) / "gallery_cluster.html")
        page = build_cluster_gallery_html(unified_path, taster_results, clusters_path,
                                           taster_report_path=taster_report)
        Path(out_path).write_text(page, encoding="utf-8")
        print(f"→ {out_path}")


_RESULT_DIR = "/NHNHOME/WORKSPACE/0426030039_A/jhyeo/ocr_filter_result"
DEFAULT_FINAL_DATASET = f"{_RESULT_DIR}/finaloutput/final_dataset.jsonl"
# 주의: 프로젝트 자체 _work/unified.jsonl(16,886건, 전부 gt 있는 기존 데모셋)와는 다른 파일 —
# 이번에 새로 필터링한 63,541건의 unified.jsonl은 ocr_filter_result/ 밑에 별도로 있음(gt 없음,
# cmcv로 자동 라벨링한 신규 데이터). 2026-07-20 확인.
DEFAULT_UNIFIED_FOR_EXPORT = f"{_RESULT_DIR}/unified.jsonl"
DEFAULT_GT_HTML_OUT = str(Path(__file__).resolve().parents[1] / "filtering_result" / "gt_html_dataset.jsonl")


def _cmd_export(args: argparse.Namespace) -> None:
    from ocr_filter.export import build_gt_html_dataset

    if args.action == "gt-html":
        final_dataset = args.final_dataset or DEFAULT_FINAL_DATASET
        unified_path = args.unified or DEFAULT_UNIFIED_FOR_EXPORT
        out_path = args.out or DEFAULT_GT_HTML_OUT
        tiers = set(args.tiers) if args.tiers else None
        stats = build_gt_html_dataset(final_dataset, unified_path, out_path, tiers=tiers)
        print(f"total={stats['n_total']} written={stats['n_written']} "
              f"skipped_unresolved={stats['n_skipped_unresolved']} "
              f"skipped_wrong_tier={stats['n_skipped_wrong_tier']} "
              f"skipped_broken_table={stats['n_skipped_broken_table']} "
              f"missing_image={stats['n_missing_image']} bad_bbox={stats['n_bad_bbox']}")
        print(f"tier: {stats['tier_counts']}")
        print(f"→ {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ocr_file_filter")
    sub = p.add_subparsers(dest="group", required=True)

    m = sub.add_parser("models", help="CMCV 3모델 다운로드/서빙")
    m.add_argument("action",
                   choices=["download", "bundle", "serve", "status", "stop"])
    m.add_argument("out", nargs="?", default="_models_bundle.tar.gz",
                   help="bundle 출력 경로 (action=bundle 일 때)")
    m.add_argument("--manifest", default=DEFAULT_MANIFEST)
    m.add_argument("--only", nargs="*",
                   help="특정 모델 키만 대상 (target/external_a/external_b)")
    m.add_argument("--dry-run", action="store_true",
                   help="serve: 실행될 명령만 출력")
    m.set_defaults(func=_cmd_models)

    i = sub.add_parser("io", help="원본 3소스(데모, GT 있음) / 신규 원본 PDF / 신규 원본 이미지(GT 없음) → 통일 스키마")
    i.add_argument("action", choices=["build", "build-raw", "build-images"])
    i.add_argument("--config", default=DEFAULT_CONFIG)
    i.add_argument("--out", default=None, help="출력 JSONL 경로 (기본: work_dir/unified.jsonl)")
    i.add_argument("--input-dir", default=None,
                    help="build-raw/build-images 전용: 원본 폴더(재귀 탐색)")
    i.add_argument("--images-out", default=None,
                    help="build-raw 전용: 페이지 PNG 출력 폴더 (기본: work_dir/images)")
    i.add_argument("--dpi", type=int, default=200, help="build-raw 전용: 페이지 렌더링 DPI")
    i.set_defaults(func=_cmd_io)

    c = sub.add_parser("cmcv", help="3모델 교차검증(CMCV) 실행")
    c.add_argument("action", choices=["run"])
    c.add_argument("--config", default=DEFAULT_CONFIG)
    c.add_argument("--unified", default=None, help="입력 JSONL (기본: work_dir/unified.jsonl)")
    c.add_argument("--out", default=None, help="출력 JSONL (기본: work_dir/cmcv_results.jsonl)")
    c.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (시험용)")
    c.add_argument("--source-type", choices=["layout", "table"], default=None)
    c.add_argument("--ids-file", default=None,
                    help="한 줄에 id 하나씩인 파일만 대상 (ddas select 출력 등)")
    c.add_argument("--workers", type=int, default=4, help="동시 처리 레코드 수")
    c.add_argument("--agree-min", type=float, default=None,
                   help="쌍별 '일치' 판정 임계값 (기본: configs 의 cmcv.match_thresholds.agree_min, 없으면 0.85)")
    c.add_argument("--text-min", type=float, default=None,
                   help="Medium 텍스트 전용 게이트: dots-paddle full_text edit_distance 하한 "
                        "(기본: configs 의 cmcv.match_thresholds.text_min, 없으면 0.90)")
    c.set_defaults(func=_cmd_cmcv)

    rs = sub.add_parser("rescue", help="Hard 티어에서 요소 단위 외부합의(N:M 그룹 매칭) 추출")
    rs.add_argument("action", choices=["run"])
    rs.add_argument("--config", default=DEFAULT_CONFIG)
    rs.add_argument("--unified", default=None, help="입력 JSONL (기본: work_dir/unified.jsonl)")
    rs.add_argument("--cmcv-results", default=None, help="기본: work_dir/cmcv_results.jsonl")
    rs.add_argument("--out", default=None, help="기본: work_dir/rescue_results.jsonl")
    rs.add_argument("--tiers", nargs="+", default=["Hard"], help="대상 티어 (기본: Hard)")
    rs.add_argument("--iou-min", type=float, default=0.5, help="bbox 매칭 IoU 하한")
    rs.add_argument("--text-min", type=float, default=0.90, help="텍스트 합의 하한")
    rs.add_argument("--coverage-min", type=float, default=0.95,
                     help="채택 요소가 덮어야 하는 영역 비율 하한 (FILTER_LOGIC uncovered_ocr"
                          "≤0.05 와 정합, 기본 0.95)")
    rs.add_argument("--limit", type=int, default=None, help="앞에서 N건만 (시험용)")
    rs.set_defaults(func=_cmd_rescue)

    cl = sub.add_parser("cluster", help="ViT 임베딩 + 2단계(페이지→요소) K-Means")
    cl.add_argument("action", choices=["build"])
    cl.add_argument("--config", default=DEFAULT_CONFIG)
    cl.add_argument("--unified", default=None)
    cl.add_argument("--out", default=None, help="출력 JSONL (기본: work_dir/clusters.jsonl)")
    cl.add_argument("--limit", type=int, default=None, help="앞에서 N개 페이지만 (시험용)")
    cl.add_argument("--n-clusters-page", type=int, default=None)
    cl.add_argument("--n-clusters-element", type=int, default=None)
    cl.add_argument("--embed-model", default=None, help="기본: google/vit-base-patch16-224-in21k")
    cl.set_defaults(func=_cmd_cluster)

    dd = sub.add_parser("ddas", help="DDAS N_i 할당량 → 실제 최종 샘플 id 선택")
    dd.add_argument("action", choices=["select", "select-additional", "select-top-clusters"])
    dd.add_argument("--config", default=DEFAULT_CONFIG)
    dd.add_argument("--taster-report", default=None, help="기본: work_dir/taster_report.json")
    dd.add_argument("--clusters", default=None, help="기본: work_dir/clusters.jsonl")
    dd.add_argument("--out", default=None,
                     help="select 기본: work_dir/ddas_selected_ids.txt / "
                          "select-additional 기본: work_dir/ddas_selected_ids_additional.txt")
    dd.add_argument("--with-replacement", action="store_true",
                     help="소형 고가치 클러스터 업샘플링 허용")
    dd.add_argument("--n-per-cluster", type=int, default=None,
                     help="주면 난이도 기반 N_i 대신 클러스터마다 균일하게 이 개수만큼 선택"
                          "(스모크 테스트용, 예: 1)")
    dd.add_argument("--cmcv-results", default=None,
                     help="select-additional 전용, 기본: work_dir/cmcv_results.jsonl "
                          "(taster 노이즈 추정 대신 이 실측 결과로 클러스터별 진짜 hard 비율을 다시 계산)")
    dd.add_argument("--already-ids", default=None,
                     help="select-additional 전용, 기본: work_dir/ddas_selected_ids.txt "
                          "(이미 뽑힌 id 는 이 파일 기준으로 제외)")
    dd.add_argument("--n-total", type=int, default=None, help="select-additional 전용, 추가로 뽑을 개수")
    dd.add_argument("--ddas-alpha", type=float, default=1.0)
    dd.add_argument("--ddas-beta", type=float, default=2.0)
    dd.add_argument("--seed", type=int, default=1, help="select-additional 기본 seed(1라운드와 겹치지 않게)")
    dd.add_argument("--top-n", type=int, default=30,
                     help="select-top-clusters 전용, 실측 medium+hard 비율 상위 N개 클러스터의 "
                          "남은 풀 전체를 가져옴(부분샘플링 아님)")
    dd.add_argument("--min-samples", type=int, default=5,
                     help="select-top-clusters 전용, 실측 표본이 이보다 적은 클러스터는 순위에서 제외")
    dd.set_defaults(func=_cmd_ddas)

    ts = sub.add_parser("taster", help="클러스터별 소량 CMCV → 난이도 추정 → DDAS 예산배분")
    ts.add_argument("action", choices=["run"])
    ts.add_argument("--config", default=DEFAULT_CONFIG)
    ts.add_argument("--clusters", default=None, help="입력 (기본: work_dir/clusters.jsonl)")
    ts.add_argument("--unified", default=None)
    ts.add_argument("--cmcv-out", default=None, help="taster cmcv 결과 (기본: work_dir/taster_cmcv_results.jsonl)")
    ts.add_argument("--out", default=None, help="리포트 JSON (기본: work_dir/taster_report.json)")
    ts.add_argument("--n-per-cluster", type=int, default=None, help="기본: configs 의 cluster.taster_per_cluster")
    ts.add_argument("--workers", type=int, default=16)
    ts.add_argument("--agree-min", type=float, default=None,
                    help="쌍별 '일치' 판정 임계값 (기본: configs 의 cmcv.match_thresholds.agree_min, 없으면 0.85)")
    ts.add_argument("--text-min", type=float, default=None,
                    help="Medium 텍스트 전용 게이트 (기본: configs 의 cmcv.match_thresholds.text_min, 없으면 0.90)")
    ts.add_argument("--ddas-n-total", type=int, default=None, help="기본: 전체 페이지 수")
    ts.set_defaults(func=_cmd_taster)

    hc = sub.add_parser("hardcase", help="Hard 티어 Judge-and-Refine(1단계)+사전라벨(2단계)")
    hc.add_argument("action", choices=["judge", "prelabel", "run", "export"])
    hc.add_argument("--config", default=DEFAULT_CONFIG)
    hc.add_argument("--unified", default=None)
    hc.add_argument("--cmcv-results", default=None)
    hc.add_argument("--judge-out", default=None, help="기본: work_dir/hardcase_judge.jsonl")
    hc.add_argument("--prelabel-out", default=None, help="기본: work_dir/hardcase_prelabel.jsonl")
    hc.add_argument("--export-out", default=None, help="기본: work_dir/labeler_export")
    hc.add_argument("--workers", type=int, default=8)
    hc.add_argument("--limit", type=int, default=None)
    hc.set_defaults(func=_cmd_hardcase)

    rp = sub.add_parser("report", help="cmcv 결과 HTML 갤러리")
    rp.add_argument("action", choices=["gallery", "cluster-gallery"])
    rp.add_argument("--config", default=DEFAULT_CONFIG)
    rp.add_argument("--unified", default=None)
    rp.add_argument("--cmcv-results", default=None)
    rp.add_argument("--taster-results", default=None,
                     help="cluster-gallery 용 (기본: work_dir/taster_cmcv_results.jsonl)")
    rp.add_argument("--clusters", default=None,
                     help="cluster-gallery 용 (기본: work_dir/clusters.jsonl)")
    rp.add_argument("--taster-report", default=None,
                     help="cluster-gallery Summary 탭용 (기본: work_dir/taster_report.json)")
    rp.add_argument("--out", default=None, help="출력 HTML 경로 (기본: work_dir/gallery.html)")
    rp.add_argument("--per-tier", type=int, default=5, help="티어별 샘플 수 (gallery 용)")
    rp.set_defaults(func=_cmd_report)

    ex = sub.add_parser("export", help="큐레이션 라벨(final_dataset.jsonl) → SFT 학습 포맷(gt_html)")
    ex.add_argument("action", choices=["gt-html"])
    ex.add_argument("--final-dataset", default=None,
                     help=f"기본: {DEFAULT_FINAL_DATASET}")
    ex.add_argument("--unified", default=None,
                     help=f"기본: {DEFAULT_UNIFIED_FOR_EXPORT}")
    ex.add_argument("--out", default=None, help=f"기본: {DEFAULT_GT_HTML_OUT}")
    ex.add_argument("--tiers", nargs="+", default=None, choices=["Easy", "Medium", "Hard"],
                     help="지정한 tier만 포함(기본: 전부). 예: --tiers Medium Hard")
    ex.set_defaults(func=_cmd_export)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
