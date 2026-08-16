html_template = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IFRS13 렌더링</title>

  <!-- MathJax: TeX 렌더링 -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      options: {{
        skipHtmlTags: ['script','noscript','style','textarea','pre','code']
      }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

  <style>
    :root {{
      --bg: #ffffff;
      --text: #111;
      --muted: #444;
      --border: #e5e7eb;
      --border-strong: #d1d5db;
      --soft: #f8fafc;
      --soft2: #f3f4f6;
    }}

    body {{
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      line-height: 1.7;
      padding: 24px;
      max-width: 1100px;
      margin: 0 auto;
    }}

    /* 공통 텍스트 블록 가독성 */
    .Page-Header {{
      font-size: 20px;
      font-weight: 800;
      margin: 18px 0 10px;
      padding-bottom: 8px;
      border-bottom: 2px solid var(--border);
    }}
    .Section-Header {{
      font-size: 16px;
      font-weight: 800;
      margin: 14px 0 10px;
    }}
    .Caption {{
      font-weight: 700;
      margin: 10px 0 6px;
    }}
    .Footnote, .note {{
      color: var(--muted);
      font-size: 13px;
      margin: 6px 0 10px;
    }}
    .Text {{
      margin: 6px 0;
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}

    /* 수식 블록 */
    .Equation-Block {{
      background: var(--soft);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      margin: 12px 0;
      overflow-x: auto;
    }}

    /* 페이지 구분자(--- Page N --- 같은 텍스트를 보기 좋게) */
    .page-sep {{
      margin: 24px 0;
      padding: 10px 12px;
      background: var(--soft2);
      border: 1px dashed var(--border-strong);
      border-radius: 10px;
      font-weight: 700;
      color: #222;
    }}

    /* -----------------------------
       ✅ 테이블 표시 강화
       ----------------------------- */

    /* 표가 너무 넓을 때 가로 스크롤을 제공 */
    .table-wrap {{
      width: 100%;
      overflow-x: auto;
      margin: 12px 0 18px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #fff;
    }}

    table {{
      border-collapse: separate;
      border-spacing: 0;
      width: 100%;
      min-width: 720px; /* 표가 좁게 찌그러지는 것을 방지 */
      font-size: 14px;
    }}

    th, td {{
      padding: 10px 12px;
      border-right: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      text-align: left;
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}

    tr > *:last-child {{
      border-right: none;
    }}

    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f9fafb;
      font-weight: 800;
      border-bottom: 1px solid var(--border-strong);
    }}

    tbody tr:nth-child(even) td {{
      background: #fcfcfd;
    }}

    /* 첫 행/첫 열 강조(선택적) */
    tbody td:first-child {{
      font-weight: 600;
    }}

    /* 표 캡션이 있으면 보기 좋게 */
    caption {{
      caption-side: top;
      text-align: left;
      padding: 10px 12px;
      font-weight: 800;
      color: #111;
    }}

    /* 표 내부 코드/수식이 깨지지 않도록 */
    td code, th code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 13px;
    }}

    /* -----------------------------
       data-bbox 표시 토글(선택)
       ----------------------------- */
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255,255,255,0.9);
      backdrop-filter: blur(6px);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      margin-bottom: 16px;
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .toolbar button {{
      border: 1px solid var(--border-strong);
      background: #fff;
      border-radius: 10px;
      padding: 8px 10px;
      cursor: pointer;
      font-weight: 700;
    }}
    .bbox-on [data-bbox] {{
      outline: 1px dashed rgba(239,68,68,0.55);
      position: relative;
    }}
    .bbox-on [data-bbox]::after {{
      content: attr(data-bbox);
      position: absolute;
      right: 6px;
      top: 6px;
      font-size: 11px;
      color: rgba(239,68,68,0.9);
      background: rgba(255,255,255,0.8);
      border: 1px dashed rgba(239,68,68,0.35);
      padding: 2px 6px;
      border-radius: 999px;
      pointer-events: none;
    }}
  </style>
</head>

<body>
  <div class="toolbar">
    <button type="button" id="toggleBbox">data-bbox 표시 토글</button>
    <span class="Footnote">표는 가로 스크롤(.table-wrap)과 sticky header로 가독성을 개선했습니다.</span>
  </div>

  {content}

  <script>
    // ✅ data-bbox 토글
    document.getElementById('toggleBbox')?.addEventListener('click', () => {{
      document.body.classList.toggle('bbox-on');
    }});

    // ✅ (중요) MathJax 타입셋 보장
    window.addEventListener('load', async () => {{
      if (window.MathJax && MathJax.typesetPromise) {{
        await MathJax.typesetPromise();
      }}
    }});

    // ✅ 테이블을 자동으로 .table-wrap 으로 감싸서 가로 스크롤 제공
    window.addEventListener('load', () => {{
      document.querySelectorAll('table').forEach((tbl) => {{
        if (tbl.closest('.table-wrap')) return;
        const wrap = document.createElement('div');
        wrap.className = 'table-wrap';
        tbl.parentNode.insertBefore(wrap, tbl);
        wrap.appendChild(tbl);
      }});
    }});
  </script>
</body>
</html>
"""
