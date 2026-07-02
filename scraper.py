"""
SHL Catalog Scraper — Playwright-based (handles JavaScript rendering)
Scrapes Individual Test Solutions from https://www.shl.com/solutions/products/product-catalog/
Saves to data/catalog.json for use by the agent.

Usage:
    python scraper.py              # Scrape and save to data/catalog.json
    python scraper.py --validate   # Validate existing catalog
    python scraper.py --seed-only  # Skip scraping, use seed catalog as-is

Requirements:
    pip install playwright beautifulsoup4
    playwright install chromium
"""

import json
import logging
import os
import re
import sys
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_BASE = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
SEED_PATH = os.path.join(os.path.dirname(__file__), "data", "seed_catalog.json")

# type=1 = Individual Test Solutions; type=2 = Job Solutions (out of scope)
INDIVIDUAL_TESTS_URL = f"{CATALOG_BASE}?start={{start}}&type=1"

TEST_TYPE_KEYWORDS = {
    "A": ["ability", "aptitude", "reasoning", "cognitive", "numerical", "verbal",
          "inductive", "deductive", "spatial", "mechanical", "checking", "comprehension"],
    "P": ["personality", "behaviour", "behavior", "opq", "character", "trait"],
    "M": ["motivation", "mq", "motivational"],
    "K": ["knowledge", "skills", "technical", "java", "python", "sql", "javascript",
          "coding", "programming", "excel", "word", "c++", "c#", "php", "ruby", "scala"],
    "S": ["simulation", "situational", "sjt", "judgement", "judgment", "scenario"],
    "B": ["biodata", "biographical"],
    "C": ["competency", "competencies", "sales", "customer service"],
    "E": ["exercise", "assessment centre", "assessment center", "in-tray", "e-tray"],
}


def infer_test_type(name: str, description: str = "") -> str:
    """Infer SHL test type code from name and description."""
    text = (name + " " + description).lower()
    for code, keywords in TEST_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return code
    return "A"


def scrape_with_playwright() -> list[dict]:
    """
    Scrape SHL catalog using Playwright (handles JS rendering).
    Returns list of assessment dicts.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        logger.error("Playwright not installed. Run: pip install playwright && playwright install chromium")
        return []

    products = []
    seen_urls: set[str] = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()

        start = 0
        page_size = 12
        max_pages = 30  # safety limit

        for page_num in range(max_pages):
            url = INDIVIDUAL_TESTS_URL.format(start=start)
            logger.info(f"Fetching page {page_num + 1} (start={start}): {url}")

            try:
                page.goto(url, wait_until="networkidle", timeout=30000)
                # Wait for product rows to appear
                page.wait_for_selector(
                    ".product-catalogue__row, .product-catalogue__list-item, [class*='catalogue']",
                    timeout=10000
                )
            except PlaywrightTimeout:
                logger.warning(f"Timeout waiting for content on page {page_num + 1}")
                # Try parsing whatever loaded
                pass
            except Exception as e:
                logger.error(f"Navigation error: {e}")
                break

            content = page.content()
            page_products = _parse_catalog_page(content)

            if not page_products:
                logger.info(f"No products found on page {page_num + 1}, stopping.")
                break

            new_products = [p for p in page_products if p["url"] not in seen_urls]
            if not new_products:
                logger.info("No new products, pagination complete.")
                break

            for product in new_products:
                seen_urls.add(product["url"])
                # Optionally scrape detail page
                try:
                    detail = _scrape_detail_playwright(page, product["url"])
                    product.update(detail)
                except Exception as e:
                    logger.warning(f"Detail scrape failed for {product['name']}: {e}")
                products.append(product)
                time.sleep(0.3)

            start += page_size
            logger.info(f"  Total so far: {len(products)}")

        browser.close()

    logger.info(f"Scrape complete: {len(products)} assessments")
    return products


def _parse_catalog_page(html: str) -> list[dict]:
    """Parse products from a catalog listing page HTML."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # Try multiple selector strategies for SHL's catalog markup
    # Strategy 1: Table rows with product links
    rows = soup.select("tr.product-catalogue__row")
    if rows:
        for row in rows:
            link = row.select_one("a[href*='/product-catalog/view/']")
            if not link:
                continue
            name = link.get_text(strip=True)
            url = urljoin(BASE_URL, link["href"])
            # Get test type from row badges
            type_badges = row.select(".product-catalogue__type span, [class*='badge']")
            raw_types = [b.get_text(strip=True) for b in type_badges]
            test_type = _map_type_badges(raw_types) or infer_test_type(name)
            if name and url:
                products.append({"name": name, "url": url, "test_type": test_type,
                                 "description": "", "job_levels": [], "competencies": []})
        return products

    # Strategy 2: Generic links to /product-catalog/view/
    links = soup.find_all("a", href=re.compile(r"/solutions/products/product-catalog/view/"))
    seen = set()
    for link in links:
        href = link.get("href", "")
        url = urljoin(BASE_URL, href)
        if url in seen:
            continue
        seen.add(url)
        name = link.get_text(strip=True)
        if name and len(name) > 2:
            products.append({
                "name": name, "url": url,
                "test_type": infer_test_type(name),
                "description": "", "job_levels": [], "competencies": []
            })

    return products


def _map_type_badges(badges: list[str]) -> Optional[str]:
    """Map SHL type badge labels to single-letter codes."""
    badge_map = {
        "ability": "A", "aptitude": "A", "personality": "P",
        "motivation": "M", "knowledge": "K", "skills": "K",
        "simulation": "S", "situational": "S", "biodata": "B",
        "competency": "C", "exercise": "E",
    }
    for badge in badges:
        badge_lower = badge.lower()
        for key, code in badge_map.items():
            if key in badge_lower:
                return code
    return None


def _scrape_detail_playwright(page, url: str) -> dict:
    """Scrape individual assessment detail page."""
    detail = {"description": "", "job_levels": [], "competencies": [], "remote_testing": True}

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        content = page.content()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")

        # Description
        for sel in [".product-detail__description", ".hero__description",
                    "[class*='description']", "meta[name='description']"]:
            el = soup.select_one(sel)
            if el:
                text = el.get("content", "") or el.get_text(strip=True)
                if text and len(text) > 20:
                    detail["description"] = text[:300]
                    break

        # Job levels
        level_patterns = re.compile(
            r"(entry.?level|graduate|professional|manager|director|executive|senior|mid.?level|operator)",
            re.I
        )
        page_text = soup.get_text()
        detail["job_levels"] = list(set(
            m.group(0).title() for m in level_patterns.finditer(page_text)
        ))[:5]

        # Remote testing
        remote_text = soup.find(string=re.compile(r"remote.*testing|online.*proctoring", re.I))
        detail["remote_testing"] = bool(remote_text)

    except Exception as e:
        logger.debug(f"Detail parse error for {url}: {e}")

    return detail


def validate_catalog(path: str = OUTPUT_PATH):
    """Print summary stats for a catalog file."""
    check_path = path if os.path.exists(path) else SEED_PATH
    if not os.path.exists(check_path):
        print(f"ERROR: No catalog found at {path} or {SEED_PATH}")
        return

    with open(check_path) as f:
        data = json.load(f)

    print(f"\n{'='*50}")
    print(f"Catalog: {check_path}")
    print(f"Total assessments: {len(data)}")
    types: dict[str, int] = {}
    for a in data:
        t = a.get("test_type", "?")
        types[t] = types.get(t, 0) + 1
    print("By type:")
    for k in sorted(types):
        print(f"  [{k}] {types[k]} assessments")

    missing = [a["name"] for a in data if not a.get("url")]
    if missing:
        print(f"WARNING: {len(missing)} items missing URLs")
    print("\nSample (first 5):")
    for a in data[:5]:
        print(f"  [{a['test_type']}] {a['name']}")
        print(f"      {a['url']}")
    print("="*50)


def save_catalog(products: list[dict], path: str = OUTPUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(products, f, indent=2)
    logger.info(f"Saved {len(products)} assessments → {path}")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate_catalog()
    elif "--seed-only" in sys.argv:
        logger.info("Using seed catalog only (no scraping)")
        validate_catalog(SEED_PATH)
    else:
        logger.info("Starting SHL catalog scrape with Playwright...")
        products = scrape_with_playwright()

        if len(products) >= 5:
            # Filter to Individual Tests only
            filtered = [
                p for p in products
                if "job-solution" not in p.get("url", "").lower()
            ]
            save_catalog(filtered)
            validate_catalog()
        else:
            logger.warning(
                f"Only scraped {len(products)} products (expected 50+). "
                "Falling back to seed catalog."
            )
            logger.info(f"Seed catalog is at: {SEED_PATH}")
            validate_catalog(SEED_PATH)
