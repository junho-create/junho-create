import argparse

from src.utils import read_file, check_data_validity
from src.layout_evaluation import evaluate_layout
from src.table_evaluation import evaluate_table


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for evaluation")
    parser.add_argument(
        "--ref_path",
        type=str, required=True,
        help="Path to the ground truth file"
    )
    parser.add_argument(
        "--pred_path",
        type=str, required=True,
        help="Path to the prediction file"
    )
    parser.add_argument(
        "--ignore_classes_for_layout",
        type=list, default=["figure", "table", "chart"],
        help="List of layout classes to ignore. This is used only for layout evaluation."
    )
    parser.add_argument(
        "--convert_table_to_text_for_index",
        action="store_true",
        help=(
            "When evaluating layout, convert predicted table HTML to text if the "
            "paired reference category is index (TOC-like fallback)."
        ),
    )
    parser.add_argument(
        "--convert_table_to_text_on_mismatch",
        action="store_true",
        help=(
            "When evaluating layout, convert table HTML to text if GT/Pred are "
            "mismatched at a position (table vs non-table)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str, default="layout",
        help="Mode for evaluation (layout/table)"
    )
    parser.add_argument(
        "--disable_pred_table_header_normalization",
        action="store_true",
        help=(
            "Disable normalization of predicted table header tags (<th> -> <td>) "
            "before table evaluation."
        ),
    )
    parser.add_argument(
        "--table_match_mode",
        type=str, default="index", choices=["index", "bbox"],
        help=(
            "Table matching for TEDS. 'index'(default)=legacy per-document first "
            "table. 'bbox'=match GT/Pred tables by bbox IoU "
            "(issue #272: https://github.com/genonai/doc_parser/issues/272)."
        ),
    )
    parser.add_argument(
        "--table_bbox_iou_thr",
        type=float, default=0.5,
        help="Minimum IoU to count two tables as matched (table_match_mode=bbox).",
    )
    parser.add_argument(
        "--table_bbox_unmatched_gt",
        type=str, default="zero", choices=["zero", "skip"],
        help=(
            "How to treat GT tables with no matched prediction (bbox mode). "
            "'zero'=score 0 (option b, penalize missed detection). "
            "'skip'=exclude from scoring (option a, score only matched tables)."
        ),
    )
    parser.add_argument(
        "--gt_coord_space",
        type=str, default="extent",
        choices=["extent", "image", "grid", "fraction_tl", "fraction_bl"],
        help=(
            "Coordinate space of GT table bboxes (bbox mode). "
            "extent=self content-extent (no image needed). image=pixel coords / "
            "page size (needs --images_dir). GT(dp-bench reference) is usually "
            "'image' when comparing pipelines with different coord systems."
        ),
    )
    parser.add_argument(
        "--pred_coord_space",
        type=str, default="extent",
        choices=["extent", "image", "grid", "fraction_tl", "fraction_bl"],
        help=(
            "Coordinate space of predicted table bboxes (bbox mode). "
            "extent=self content-extent. grid=0~N grid /N (Qwen3.5 finetune JSON, "
            "N=--pred_grid, default 1024). fraction_tl=already [0,1] top-left. fraction_bl=already "
            "[0,1] bottom-left/PDF -> Y flipped (Genos doc_parser v1.3.8 / v2.0). "
            "image=pixel coords / page size (needs --images_dir)."
        ),
    )
    parser.add_argument(
        "--images_dir",
        type=str, default=None,
        help="Image folder to read page sizes when coord space is 'image'.",
    )
    parser.add_argument(
        "--pred_grid",
        type=float, default=1024.0,
        help=(
            "Grid size for pred_coord_space=grid (default 1024; Qwen finetune "
            "uses bbox_scale=1024)."
        ),
    )
    parser.add_argument(
        "--table_match_dump",
        type=str, default=None,
        help=(
            "bbox 모드에서 문서별 GT-Pred 테이블 매칭 쌍(iou 포함)을 JSON 으로 "
            "저장할 경로. 매칭 방식 비교 분석용 (issue #318 Phase 3)."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-" * 50)

    label_data = read_file(args.ref_path)
    pred_data = read_file(args.pred_path)

    check_data_validity(label_data, pred_data)

    if args.mode == "layout":
        score = evaluate_layout(
            label_data, pred_data,
            ignore_classes=args.ignore_classes_for_layout,
            convert_table_to_text_for_index=args.convert_table_to_text_for_index,
            convert_table_to_text_on_mismatch=args.convert_table_to_text_on_mismatch,
        )
        print(f"NID Score: {score:.4f}")
    elif args.mode == "table":
        teds_score, teds_s_score = evaluate_table(
            label_data,
            pred_data,
            normalize_pred_table_header_tags=not args.disable_pred_table_header_normalization,
            match_mode=args.table_match_mode,
            bbox_iou_thr=args.table_bbox_iou_thr,
            bbox_unmatched_gt=args.table_bbox_unmatched_gt,
            gt_coord_space=args.gt_coord_space,
            pred_coord_space=args.pred_coord_space,
            images_dir=args.images_dir,
            pred_grid=args.pred_grid,
            match_dump_path=args.table_match_dump,
        )
        print(f"TEDS Score: {teds_score:.4f}")
        print(f"TEDS-S Score: {teds_s_score:.4f}")
    else:
        raise ValueError(f"{args.mode} mode not supported")


if __name__ == "__main__":
    main()
