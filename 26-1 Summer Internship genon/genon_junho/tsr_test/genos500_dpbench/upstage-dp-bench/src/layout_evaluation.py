import html as html_lib
import re

from rapidfuzz import fuzz

def calc_nid(
    gt_text : list,
    pred_text : list,
) -> float:
    """Calculate the Normalized InDel score between the gt and pred text.

    Args:
        gt_text (str): The string of gt text to compare.
        pred_text (str): The string of pred text to compare.

    Returns:
        float: The nid score between gt and pred text.
    """

    # if gt and pred is empty, return 1
    if len(gt_text) == 0 and len(pred_text) == 0:
        score = 1
    # if pred is empty while gt is not, return 0
    elif len(gt_text) > 0 and len(pred_text) == 0:
        score = 0
    else:
        score = fuzz.ratio(gt_text, pred_text)

    return score


def _table_html_to_text(table_html: str) -> str:
    if not table_html:
        return ""

    text = re.sub(r"<[^>]+>", " ", table_html)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_text(
    data : dict,
    ignore_classes : list = [],
    strings_to_remove : list = ["\n"],
    paired_data: dict = None,
    convert_table_to_text_for_index: bool = False,
    convert_table_to_text_on_mismatch: bool = False,
) -> str:
    """Extract text from the dictionary data.

    Args:
        data (dict): The data to extract text from.
        ignore_classes (list): A list of classes to ignore during extraction.
        strings_to_remove (list): A list of strings to remove from the extracted text.
        paired_data (dict): Optional paired data to compare category positions.
        convert_table_to_text_for_index (bool): If True, include table HTML as text
            when paired GT category is index.
        convert_table_to_text_on_mismatch (bool): If True, include table HTML as text
            when the paired category at the same position is not table, or when the
            paired document has no table at all.

    Returns:
        str: The concatenated text extracted from the data.
    """

    ignore_classes = [x.lower() for x in ignore_classes]

    concatenated_text = ""
    paired_elements = paired_data.get("elements", []) if paired_data else []
    paired_categories = [
        (elem.get("category") or "").lower() for elem in paired_elements
    ]
    paired_has_index = "index" in paired_categories
    paired_has_table = "table" in paired_categories
    # Fallback for TOC-like pages: GT has index but no table, while pred may emit table.
    use_doc_level_index_fallback = paired_has_index and not paired_has_table
    # Fallback for mismatch mode: if paired doc has no table, convert all table blocks.
    use_doc_level_table_mismatch_fallback = not paired_has_table

    for idx, elem in enumerate(data["elements"]):
        elem_category = elem["category"].lower()

        if elem_category in ignore_classes:
            should_convert_table = False
            if elem_category == "table":
                paired_category = ""
                if idx < len(paired_elements):
                    paired_category = (
                        paired_elements[idx].get("category") or ""
                    ).lower()

                if convert_table_to_text_for_index:
                    should_convert_table = (
                        paired_category == "index" or use_doc_level_index_fallback
                    )

                if convert_table_to_text_on_mismatch:
                    should_convert_table = should_convert_table or (
                        paired_category != "table"
                        or use_doc_level_table_mismatch_fallback
                    )

            if should_convert_table:
                table_text = _table_html_to_text(elem["content"].get("html", ""))
                if table_text:
                    concatenated_text += table_text + " "
            continue

        concatenated_text += elem["content"]["text"] + ' '

    # remove unwanted strings
    for string in strings_to_remove:
        concatenated_text = concatenated_text.replace(string, '')

    return concatenated_text


def evaluate_layout(
    gt : dict,
    pred : dict,
    ignore_classes : list = [],
    convert_table_to_text_for_index: bool = False,
    convert_table_to_text_on_mismatch: bool = False,
) -> float:
    """Evaluate the layout of the gt against the pred.

    Args:
        gt (dict): The gt layout to evaluate.
        pred (dict): The pred layout to evaluate against.
        ignore_classes (list): A list of classes to ignore during evaluation.
        convert_table_to_text_for_index (bool): If True, converts predicted table
            HTML to text for positions/documents where GT is index.
        convert_table_to_text_on_mismatch (bool): If True, converts table HTML to
            text when GT/Pred categories mismatch at a position (table vs non-table).

    Returns:
        float: The layout evaluation score.
    """
    scores = []
    for image_key in gt.keys():
        gt_data = gt.get(image_key)
        pred_data = pred.get(image_key)

        gt_text = extract_text(
            gt_data,
            ignore_classes,
            paired_data=pred_data if convert_table_to_text_on_mismatch else None,
            convert_table_to_text_on_mismatch=convert_table_to_text_on_mismatch,
        )
        pred_text = extract_text(
            pred_data,
            ignore_classes,
            paired_data=gt_data,
            convert_table_to_text_for_index=convert_table_to_text_for_index,
            convert_table_to_text_on_mismatch=convert_table_to_text_on_mismatch,
        )

        score = calc_nid(gt_text, pred_text)

        scores.append(score)

    if len(scores) > 0:
        avg_score = sum(scores) / (len(scores) * 100)
    else:
        avg_score = 0

    return avg_score
