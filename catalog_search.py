"""
SHL Catalog Search
BM25-based retrieval over SHL Individual Test Solutions.
Includes a self-contained BM25 implementation to avoid dependency issues.
"""
import json
import logging
import math
import os
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
SEED_PATH = os.path.join(os.path.dirname(__file__), "data", "seed_catalog.json")

# Query expansion: common hiring terms → SHL assessment keywords
SYNONYMS: dict[str, list[str]] = {
    "developer": ["programming", "software", "engineering", "coding", "java", "python", "javascript"],
    "engineer": ["programming", "software", "technical", "development", "systems"],
    "analyst": ["analytical", "numerical", "data", "quantitative", "reporting"],
    "manager": ["management", "leadership", "team", "managerial", "director"],
    "personality": ["OPQ", "behavior", "trait", "work style", "character"],
    "aptitude": ["ability", "reasoning", "cognitive", "intelligence"],
    "cognitive": ["ability", "reasoning", "aptitude", "verbal", "numerical"],
    "soft skills": ["personality", "OPQ", "interpersonal", "communication", "behavior"],
    "culture fit": ["personality", "values", "motivation", "work style"],
    "graduate": ["entry level", "new hire", "junior", "campus"],
    "senior": ["experienced", "mid-level", "professional", "specialist"],
    "executive": ["director", "CXO", "VP", "leadership", "C-suite"],
    "sales": ["persuasion", "commercial", "negotiation", "revenue", "customer"],
    "customer service": ["service", "helpdesk", "support", "contact center", "empathy"],
    "data science": ["python", "SQL", "analytics", "statistical", "machine learning"],
    "frontend": ["JavaScript", "React", "Angular", "CSS", "HTML", "UI", "web"],
    "backend": ["Java", "Python", "SQL", "API", "server", "database"],
    "fullstack": ["JavaScript", "Java", "Python", "frontend", "backend", "web"],
    "tech": ["programming", "software", "developer", "technical", "coding"],
    "leadership": ["management", "OPQ", "decision making", "strategic", "vision"],
    "remote": ["remote work", "hybrid", "RemoteWorkQ", "self-management"],
    "safety": ["workplace safety", "compliance", "risk", "operations"],
    "motivation": ["MQ", "engagement", "drive", "values"],
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def expand_query(query: str) -> str:
    """Add synonym expansions to improve retrieval recall."""
    lower = query.lower()
    parts = [query]
    for key, expansions in SYNONYMS.items():
        if key in lower:
            parts.extend(expansions)
    return " ".join(parts)


class SimpleBM25:
    """Minimal BM25Okapi implementation (no external dependencies)."""
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.N, 1)

        # Build inverted index and document frequencies
        self.df: dict[str, int] = defaultdict(int)
        self.tf: list[dict[str, float]] = []

        for doc in corpus:
            freq: dict[str, int] = defaultdict(int)
            for token in doc:
                freq[token] += 1
            self.tf.append(dict(freq))
            for token in set(doc):
                self.df[token] += 1

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for token in query_tokens:
            if token not in self.df:
                continue
            idf = math.log((self.N - self.df[token] + 0.5) / (self.df[token] + 0.5) + 1)
            for i, doc in enumerate(self.corpus):
                dl = len(doc)
                tf = self.tf[i].get(token, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * num / max(den, 1e-8)
        return scores


class CatalogSearch:
    def __init__(self):
        self.assessments = self._load_catalog()
        self._valid_urls: set[str] = {a["url"] for a in self.assessments}
        self._build_index()
        logger.info(f"CatalogSearch ready: {len(self.assessments)} assessments indexed")

    def _load_catalog(self) -> list[dict[str, Any]]:
        """Load catalog; prefer scraped catalog over seed."""
        for path in [CATALOG_PATH, SEED_PATH]:
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                if data:
                    logger.info(f"Loaded {len(data)} assessments from {path}")
                    return data
        raise FileNotFoundError(
            "No catalog found. Ensure data/seed_catalog.json exists, "
            "or run scraper.py to build data/catalog.json."
        )

    def _build_index(self):
        """Build BM25 index with field weighting."""
        corpus = []
        for a in self.assessments:
            # Weight fields by importance: name > competencies > description > levels
            tokens = (
                tokenize(a["name"]) * 4
                + tokenize(" ".join(a.get("competencies", []))) * 3
                + tokenize(a.get("description", "")) * 2
                + tokenize(" ".join(a.get("job_levels", []))) * 2
                + tokenize(a.get("test_type", ""))
            )
            corpus.append(tokens)

        self.bm25 = SimpleBM25(corpus)

    def search(self, query: str, top_k: int = 12) -> list[dict[str, Any]]:
        """Search catalog using BM25 with query expansion."""
        if not query.strip():
            return self.assessments[:top_k]

        expanded = expand_query(query)
        tokens = tokenize(expanded)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = [
            self.assessments[i] for i in ranked[:top_k] if scores[i] > 0
        ]
        return results if results else self.assessments[:top_k]

    def get_all(self) -> list[dict[str, Any]]:
        return self.assessments

    def validate_url(self, url: str) -> bool:
        """Check URL exists in catalog (anti-hallucination guard)."""
        return url in self._valid_urls

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        """Find best matching assessment by name."""
        name_lower = name.lower()
        # Exact match
        for a in self.assessments:
            if a["name"].lower() == name_lower:
                return a
        # Substring match
        for a in self.assessments:
            if name_lower in a["name"].lower() or a["name"].lower() in name_lower:
                return a
        return None

    def format_for_prompt(self, assessments: list[dict]) -> str:
        """Format assessments as compact prompt text."""
        lines = []
        for a in assessments:
            line = f"[{a['test_type']}] {a['name']} | {a['url']}"
            if a.get("description"):
                line += f"\n    {a['description'][:160]}"
            if a.get("job_levels"):
                line += f"\n    Levels: {', '.join(a['job_levels'])}"
            lines.append(line)
        return "\n".join(lines)
