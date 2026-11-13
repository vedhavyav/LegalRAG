
"""
LegalRAG corpus collection configuration.

The default target is 100 judgments:
- 5 legal categories
- 2 court types
- 10 judgments per category/court combination
"""

TARGET_TOTAL = 100

COURT_TYPES = {
    "supreme_court": "Supreme Court of India",
    "high_court": "High Court",
}

CATEGORIES = {
    "bail_anticipatory_bail": {
        "label": "Bail and anticipatory bail",
        "queries": [
            "anticipatory bail",
            "regular bail section 439 CrPC",
            "principles governing grant of bail",
        ],
    },
    "criminal_procedure_arrest": {
        "label": "Criminal procedure and arrest",
        "queries": [
            "arrest under section 41 CrPC",
            "custodial interrogation",
            "quashing criminal proceedings section 482 CrPC",
        ],
    },
    "constitutional_law": {
        "label": "Constitutional law",
        "queries": [
            "Article 14 and Article 21 criminal proceedings",
            "constitutional rights and criminal justice",
            "fundamental rights criminal investigation",
        ],
    },
    "pmla_money_laundering": {
        "label": "PMLA and money laundering",
        "queries": [
            "Prevention of Money Laundering Act bail",
            "section 45 PMLA",
            "proceeds of crime money laundering",
        ],
    },
    "evidence_other_criminal_law": {
        "label": "Evidence and other criminal law",
        "queries": [
            "section 27 Indian Evidence Act",
            "circumstantial evidence criminal appeal",
            "criminal conspiracy Indian Penal Code",
        ],
    },
}

# A judgment must have one primary category.
VALID_CATEGORIES = set(CATEGORIES.keys())

# Search pagination safety limit. Increase if a bucket cannot be filled.
MAX_PAGES_PER_QUERY = 10

# Number of search results to inspect on each page.
# The API may return fewer results than this.
MAX_SEARCH_RESULTS_PER_PAGE = 100

# Request timeout is already configured in IndianKanoonClient.
# This delay reduces the chance of sending requests too quickly.
REQUEST_DELAY_SECONDS = 0.5

# Do not count statutory sections, articles, or other non-judgment records.
EXCLUDED_DOCSOURCES = {
    "Union of India - Section",
    "Constitution Article",
    "Constitution and Amendments",
}
