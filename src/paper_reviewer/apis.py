"""External API clients: Semantic Scholar and arXiv."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Optional

import httpx

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"


def search_semantic_scholar(title: str, timeout: float = 10.0) -> Optional[dict]:
    """Search Semantic Scholar for a paper by title.

    Returns the top result dict or None if not found / error.
    """
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "title,authors,year,externalIds,citationCount",
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", [])
            if results:
                return results[0]
    except httpx.HTTPStatusError:
        return None
    except httpx.RequestError:
        return None
    return None


def title_similarity(a: str, b: str) -> float:
    """Simple token-overlap similarity between two titles."""
    if not a or not b:
        return 0.0
    tokens_a = set(re.sub(r"[^\w\s]", "", a.lower()).split())
    tokens_b = set(re.sub(r"[^\w\s]", "", b.lower()).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def download_arxiv_pdf(arxiv_id: str, dest_dir: Optional[Path] = None) -> Path:
    """Download a PDF from arXiv and return its local path."""
    # Normalize ID (strip 'arxiv:' prefix if present)
    arxiv_id = re.sub(r"^arxiv:", "", arxiv_id, flags=re.IGNORECASE).strip()

    if dest_dir is None:
        dest_dir = Path(tempfile.gettempdir())
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    out_path = dest_dir / f"{arxiv_id}.pdf"
    if out_path.exists():
        return out_path

    url = f"{ARXIV_PDF_BASE}/{arxiv_id}"
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

    return out_path


def check_url_accessible(url: str, timeout: float = 10.0) -> bool:
    """Return True if the URL returns a 2xx or 3xx status."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.head(url)
            return resp.status_code < 400
    except Exception:
        return False


def check_github_repo_files(repo_url: str, timeout: float = 10.0) -> dict[str, bool]:
    """Check for common environment definition files in a GitHub repo via API."""
    # Extract owner/repo from URL
    match = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?(?:/|$)", repo_url)
    if not match:
        return {}

    repo_path = match.group(1).rstrip("/")
    api_url = f"https://api.github.com/repos/{repo_path}/contents/"
    env_files = ["requirements.txt", "pyproject.toml", "environment.yml", "setup.py", "conda.yml"]

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(api_url)
            if resp.status_code != 200:
                return {}
            contents = resp.json()
            names = {item["name"] for item in contents if isinstance(item, dict)}
            return {f: f in names for f in env_files}
    except Exception:
        return {}


def parse_arxiv_id_from_input(input_str: str) -> Optional[str]:
    """Extract arXiv ID from various input formats."""
    # Direct ID: 2501.12345 or 2501.12345v2
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", input_str):
        return input_str

    # URL: https://arxiv.org/abs/2501.12345
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", input_str)
    if match:
        return match.group(1)

    return None
