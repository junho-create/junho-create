from utils.html_utils import (
    CellInfo,
    TableStructure,
    parse_html_table,
    structure_to_html,
    normalize_html,
    extract_spans_from_html,
    get_table_dimensions,
    html_to_structure_only,
)
from utils.span_analyzer import (
    SpanStats,
    DatasetSpanProfile,
    analyze_span,
    analyze_dataset,
    categorize_by_complexity,
    compute_sampling_weights,
)
from utils.prompt_templates import (
    get_system_prompt,
    get_user_prompt,
    build_thinking_chain,
    format_assistant_response,
    build_chat_messages,
)
