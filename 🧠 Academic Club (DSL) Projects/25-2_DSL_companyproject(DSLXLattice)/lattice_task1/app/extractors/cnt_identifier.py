# -*- coding: utf-8 -*-
import re
from typing import List
from app.extractors.base import FieldExtractor, Candidate, Evidence
from app.ingest.loader_pdf import TextBlock

RE_ID = re.compile(r"(계약\s*번호|문서\s*번호|Contract\s*No\.?|Agreement\s*ID)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)")

class IdentifierExtractor(FieldExtractor):
    def extract(self, blocks: List[TextBlock]) -> List[Candidate]:
        cands: List[Candidate] = []
        for b in blocks:
            m = RE_ID.search(b.text)
            if m:
                value = m.group(2).strip()
                ev = Evidence(page=b.page, lines=[b.line_no], snippet=b.text)
                cands.append(Candidate("TEMP_KEY", value, "header", ev, {"keyword":1}))
        return cands