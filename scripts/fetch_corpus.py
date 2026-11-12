"""Acquire the Roads and Transport corpus from the Dubai Legislation Portal.

Applies the exact primary-category filter (ddlMainCat = "Roads and Transport"),
pages through all results, saves the raw filtered result pages as evidence,
writes a manifest, and downloads the PDF + HTML rendition of every result.

Usage:  python scripts/fetch_corpus.py [--out data] [--no-download]
Re-running skips files that already exist (idempotent).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://dlp.dubai.gov.ae"
SEARCH_URL = f"{BASE}/en/Pages/LegislationSearch.aspx"
CATEGORY = "Roads and Transport"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) corpus-fetch/1.0"
DELAY = 1.0  # polite delay between requests (seconds)


def form_fields(soup: BeautifulSoup) -> dict[str, str]:
    """Collect the ASP.NET form state needed for a postback."""
    data: dict[str, str] = {}
    form = soup.find("form", id="aspnetForm") or soup
    for inp in form.find_all("input"):
        name, typ = inp.get("name"), (inp.get("type") or "text").lower()
        if not name or typ in ("submit", "button", "image", "file"):
            continue
        if typ in ("checkbox", "radio") and not inp.has_attr("checked"):
            continue
        data[name] = inp.get("value", "")
    for sel in form.find_all("select"):
        name = sel.get("name")
        if not name:
            continue
        opt = sel.find("option", selected=True) or sel.find("option")
        data[name] = opt.get("value", "") if opt else ""
    return data


def parse_results(soup: BeautifulSoup) -> list[dict]:
    rows = []
    for li in soup.select("li[data-id]"):
        fields = {}
        for h3 in li.find_all("h3"):
            h4 = h3.find_next_sibling("h4")
            if h4:
                fields[h3.get_text(strip=True)] = h4.get_text(strip=True)
        pdf = html = arabic = None
        size = None
        for a in li.find_all("a"):
            text = a.get_text(" ", strip=True)
            if text.startswith("View as PDF"):
                pdf = a.get("href")
                small = a.find("small")
                size = small.get_text(strip=True) if small else None
            elif text.startswith("View as HTML"):
                html = a.get("href")
            elif "Arabic Legislation" in text:
                arabic = (a.get("href") or "").replace("\\", "/")
        title_el = li.select_one(".title")
        rows.append({
            "portal_id": li["data-id"],
            "title": re.sub(r"\s+", " ", title_el.get_text(" ", strip=True)) if title_el else None,
            "fields": fields,
            "pdf_href": pdf,
            "pdf_size_label": size,
            "html_href": html,
            "arabic_href": arabic,
        })
    return rows


def abs_url(href: str | None) -> str | None:
    if not href:
        return None
    return quote(urljoin(BASE + "/", href), safe=":/?&=%")


def parse_title(title: str) -> dict:
    m = re.search(r"No\.\s*\((\d+)\)\s*of\s*(\d{4})", title or "")
    return {"number": int(m.group(1)) if m else None, "year": int(m.group(2)) if m else None}


def iso_date(s: str | None) -> str | None:
    try:
        return datetime.strptime(s, "%d %b %Y").date().isoformat() if s else None
    except ValueError:
        return None


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    ev_dir, pdf_dir, html_dir = out / "evidence", out / "raw" / "pdf", out / "raw" / "html"
    for d in (ev_dir, pdf_dir, html_dir):
        d.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    s.headers["User-Agent"] = UA

    # 1. Load the search page and discover control names / the category value.
    soup = BeautifulSoup(s.get(SEARCH_URL, timeout=60).text, "html.parser")
    main_cat = soup.find("select", id="ddlMainCat")
    cat_value = next(o["value"] for o in main_cat.find_all("option")
                     if o.get_text(strip=True) == CATEGORY)
    btn_search = soup.find("input", id=re.compile(r"btnSearch$"))["name"]
    print(f"category '{CATEGORY}' -> ddlMainCat value {cat_value}")

    # 2. Postback with the primary-category filter applied.
    data = form_fields(soup)
    data[main_cat["name"]] = cat_value
    data[btn_search] = "Search"
    time.sleep(DELAY)
    resp = s.post(SEARCH_URL, data=data, timeout=60)
    soup = BeautifulSoup(resp.text, "html.parser")
    (ev_dir / "search_page_1.html").write_text(resp.text, encoding="utf-8")

    m = re.search(r"(\d+)\s+Results Found", soup.get_text(" "))
    reported = int(m.group(1)) if m else None
    rows = parse_results(soup)
    page = 1
    print(f"portal reports {reported} results; page 1 has {len(rows)}")

    # 3. "Load More" postbacks until all results are present.
    while reported and len(rows) < reported and page < 20:
        nxt = soup.find("a", id=re.compile(r"lbNext$"))
        if not nxt:
            break
        target = re.search(r'PostBackOptions\("([^"]+)"', nxt["href"]).group(1)
        data = form_fields(soup)
        data["__EVENTTARGET"], data["__EVENTARGUMENT"] = target, ""
        time.sleep(DELAY)
        resp = s.post(SEARCH_URL, data=data, timeout=60)
        soup = BeautifulSoup(resp.text, "html.parser")
        page += 1
        (ev_dir / f"search_page_{page}.html").write_text(resp.text, encoding="utf-8")
        new = parse_results(soup)
        selected = soup.find("select", id="ddlMainCat").find("option", selected=True)
        assert selected and selected["value"] == cat_value, "category filter lost during paging"
        seen = {r["portal_id"] for r in rows}
        added = [r for r in new if r["portal_id"] not in seen]
        rows.extend(added)
        print(f"page {page}: +{len(added)} (total {len(rows)})")
        if not added:
            break

    # 4. Build manifest records.
    today = date.today().isoformat()
    docs = []
    for r in rows:
        f, t = r["fields"], parse_title(r["title"])
        doc_id = f"dlp-{r['portal_id']}"
        docs.append({
            "document_id": doc_id,
            "portal_id": r["portal_id"],
            "title": r["title"],
            "source_url": SEARCH_URL,
            "content_url_pdf": abs_url(r["pdf_href"]),
            "content_url_html": abs_url(r["html_href"]),
            "arabic_url": abs_url(r["arabic_href"]),
            "primary_category": CATEGORY,
            "language": "en",
            "retrieval_date": today,
            "type": f.get("Type"),
            "number": t["number"],
            "year": t["year"],
            "issuing_authority": f.get("Issuing Authority"),
            "official_gazette_no": f.get("Official Gazette No"),
            "date_of_issuance": iso_date(f.get("Date of Issuance")),
            "source_kind": f.get("Source"),
            "status": f.get("Status"),
            "pdf_size_label": r["pdf_size_label"],
            "local_pdf": None, "local_html": None,
            "pdf_sha256": None, "html_sha256": None,
            "download_errors": [],
        })

    # 5. Download PDF + HTML renditions.
    if not args.no_download:
        for d in docs:
            for kind, url, folder, ext in (("pdf", d["content_url_pdf"], pdf_dir, "pdf"),
                                           ("html", d["content_url_html"], html_dir, "html")):
                if not url:
                    d["download_errors"].append(f"{kind}: no link on portal")
                    continue
                dest = folder / f"{d['document_id']}.{ext}"
                if not dest.exists():
                    try:
                        time.sleep(DELAY)
                        r = s.get(url, timeout=120)
                        r.raise_for_status()
                        if kind == "pdf" and not r.content.startswith(b"%PDF"):
                            raise ValueError("response is not a PDF")
                        dest.write_bytes(r.content)
                    except Exception as e:  # noqa: BLE001 - log and continue
                        d["download_errors"].append(f"{kind}: {e}")
                        print(f"  FAILED {d['document_id']} {kind}: {e}")
                        continue
                d[f"local_{kind}"] = dest.as_posix()
                d[f"{kind}_sha256"] = sha256(dest)
            print(f"ok {d['document_id']}  {d['title'][:70]}")

    manifest = {
        "source": SEARCH_URL,
        "filter": {"control": "ddlMainCat (Primary category)", "label": CATEGORY, "value": cat_value},
        "portal_reported_count": reported,
        "collected_count": len(docs),
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_files": sorted(p.as_posix() for p in ev_dir.glob("search_page_*.html")),
        "documents": docs,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = sum(1 for d in docs if d["download_errors"])
    print(f"\nmanifest: {out / 'manifest.json'}  docs={len(docs)}  with_errors={failed}")
    return 0 if len(docs) == reported else 1


if __name__ == "__main__":
    sys.exit(main())
