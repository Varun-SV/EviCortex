#!/usr/bin/env python3
"""Build the EviCortex GitHub Pages site.

Writes index.html (landing) and docs/**/*.html (one page per Markdown doc) from
tools/site/landing.html and docs/**/*.md. Requires: pip install markdown pygments
Run from the repository root:  python tools/build_site.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Varun-SV/EviCortex"
BRANCH = "main"

# (path under docs/, sidebar group)
DOCS = [
    ("index.md", "Start here"),
    ("getting-started.md", "Start here"),
    ("concepts.md", "Start here"),
    ("api-guide.md", "Start here"),
    ("faq.md", "Start here"),
    ("benchmarking.md", "Benchmarking"),
    ("benchmark-contract.md", "Benchmarking"),
    ("architecture.md", "Research notes"),
    ("phase-b.md", "Research notes"),
    ("phase-c.md", "Research notes"),
    ("decisions/0001-local-first-python-sqlite.md", "Research notes"),
    ("results/prototype-a.md", "Results"),
    ("results/prototype-b-diagnostic.md", "Results"),
    ("results/prototype-c-semantic-smoke.md", "Results"),
]
SHORT = {
    "index.md": "Overview", "getting-started.md": "Getting started", "concepts.md": "Core concepts",
    "api-guide.md": "API guide", "faq.md": "FAQ and limitations", "benchmarking.md": "Benchmarking",
    "benchmark-contract.md": "Benchmark contract", "architecture.md": "Architecture",
    "phase-b.md": "Phase B plan", "phase-c.md": "Phase C evaluation",
    "decisions/0001-local-first-python-sqlite.md": "ADR 0001: Python + SQLite",
    "results/prototype-a.md": "Prototype A", "results/prototype-b-diagnostic.md": "Prototype B diagnostic",
    "results/prototype-c-semantic-smoke.md": "Prototype C smoke",
}

GH_ICON = '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
FAVICON = "data:image/svg+xml," + re.sub(r"\s+", " ", """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%2304050d'/><circle cx='11' cy='12' r='4.5' fill='%23f6b84a'/><circle cx='22' cy='21' r='3.6' fill='none' stroke='%23ea6f95' stroke-width='1.8'/><circle cx='23' cy='8.5' r='2.4' fill='%23a9b1ff'/></svg>""")


def shell(root: str, title: str, desc: str, body_cls: str, mode: str, main: str, nav_here: str) -> str:
    on_landing = root == ""
    anchor = (lambda a: a) if on_landing else (lambda a: f"{root}index.html{a}")
    nav = [("How it works", anchor("#how")), ("Results", anchor("#results")), ("Roadmap", anchor("#roadmap")), ("Docs", f"{root}docs/index.html")]
    nav_html = "".join(
        f'<li><a href="{h}"{" aria-current=\"page\"" if label == nav_here else ""}>{label}</a></li>' for label, h in nav
    )
    mnav = [("Home", f"{root}index.html")] + nav
    mnav_html = "".join(f'<li style="--i:{i}"><a href="{h}"><small>0{i + 1}</small>{label}</a></li>' for i, (label, h) in enumerate(mnav))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
<meta name="theme-color" content="#04050d">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(desc)}">
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="{root}assets/ev.css">
<script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}</script>
</head>
<body class="{body_cls}" data-root="{root}" data-scene="{mode}">
<a class="skip" href="#main">Skip to content</a>
<canvas id="ev-canvas" aria-hidden="true"></canvas>
<div class="vignette" aria-hidden="true"></div>
<div class="focus" aria-hidden="true"></div>
<div class="grain" aria-hidden="true"></div>
<div class="veil" aria-hidden="true"></div>
<header class="hdr">
  <div class="bar">
    <a class="brand" href="{root}index.html" aria-label="EviCortex home"><span class="mk" aria-hidden="true"><i></i><i></i><i></i></span>EviCortex</a>
    <ul class="nav" id="nav">{nav_html}</ul>
    <button class="search-btn" type="button" aria-label="Search the docs"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg><kbd>⌘K</kbd></button>
    <button class="menu-btn" type="button" aria-controls="mnav" aria-expanded="false">Menu</button>
    <a class="gh" href="{REPO}" target="_blank" rel="noopener">{GH_ICON}<span>GitHub</span></a>
    <span class="prog" aria-hidden="true"></span>
  </div>
</header>
<nav class="mnav" id="mnav" aria-label="Menu">
  <ul>{mnav_html}</ul>
  <div class="mfoot"><a class="btn btn-p" href="{root}docs/getting-started.html"><span class="lbl">Get started</span></a><a class="btn btn-g" href="{REPO}" target="_blank" rel="noopener"><span class="lbl">GitHub</span></a></div>
</nav>
{main}
<footer class="foot">
  <div class="wrap">
    <div>
      <a class="brand" href="{root}index.html"><span class="mk" aria-hidden="true"><i></i><i></i><i></i></span>EviCortex</a>
      <p>Evidence-preserving, temporal memory infrastructure for long-lived AI systems. Pre-alpha research software by Varun S V. MIT licence.</p>
    </div>
    <ul><li><a href="{root}docs/getting-started.html">Getting started</a></li><li><a href="{root}docs/concepts.html">Core concepts</a></li><li><a href="{root}docs/api-guide.html">API guide</a></li></ul>
    <ul><li><a href="{root}docs/architecture.html">Architecture</a></li><li><a href="{root}docs/benchmarking.html">Benchmarking</a></li><li><a href="{REPO}" target="_blank" rel="noopener">Source on GitHub</a></li></ul>
    <div class="big" aria-hidden="true">EviCortex</div>
  </div>
</footer>
<div class="search" role="dialog" aria-modal="true" aria-label="Search documentation">
  <div class="box glass"><input type="search" placeholder="Search the docs…" aria-label="Search the docs"><ul></ul></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<script src="https://unpkg.com/lenis@1.1.13/dist/lenis.min.js"></script>
<script src="{root}assets/search-index.js"></script>
<script src="{root}assets/ev-ui.js"></script>
<script type="module" src="{root}assets/ev-scene.js"></script>
</body>
</html>
"""


def front_matter(text: str) -> tuple[dict, str]:
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        for line in text[3:end].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        text = text[end + 4:]
    return meta, text


def strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def rewrite_links(body: str, rel: str) -> str:
    here = Path("docs") / rel

    def fix(m: re.Match) -> str:
        href = m.group(1)
        if re.match(r"^(https?:|mailto:|#)", href):
            return m.group(0)
        path, _, frag = href.partition("#")
        target = (here.parent / path)
        norm = Path(*[p for p in target.parts])  # keep relative
        resolved = (ROOT / norm).resolve()
        try:
            in_repo = resolved.relative_to(ROOT)
        except ValueError:
            return m.group(0)
        if in_repo.parts[:1] == ("docs",) and in_repo.suffix == ".md":
            new = path[:-3] + ".html"
            return f'href="{new}{"#" + frag if frag else ""}"'
        if in_repo.parts[:1] == ("docs",):
            return m.group(0)  # e.g. JSON artifacts served next to the pages
        kind = "tree" if resolved.is_dir() else "blob"
        return f'href="{REPO}/{kind}/{BRANCH}/{in_repo.as_posix()}" target="_blank" rel="noopener"'

    return re.sub(r'href="([^"]+)"', fix, body)


def main() -> None:
    pages = []
    for rel, group in DOCS:
        meta, text = front_matter((ROOT / "docs" / rel).read_text(encoding="utf-8"))
        md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "toc", "attr_list", "sane_lists"],
                               extension_configs={"codehilite": {"guess_lang": False, "css_class": "codehilite"}, "toc": {"toc_depth": "2-3"}})
        body = md.convert(text)
        title = meta.get("title") or (re.search(r"<h1[^>]*>(.*?)</h1>", body) or [None, rel])[1]
        pages.append(dict(rel=rel, group=group, title=strip_tags(title), body=body, toc=md.toc_tokens))

    index = []
    for i, p in enumerate(pages):
        rel = p["rel"]
        depth = rel.count("/") + 1
        root = "../" * depth
        out = Path("docs") / rel.replace(".md", ".html")
        body = p["body"]
        body = body.replace('<div class="codehilite">', '<div class="codehilite codewrap">')
        body = re.sub(r"<pre><code>", '<div class="codewrap"><pre><code>', body)
        body = re.sub(r'(<div class="codewrap"><pre><code>.*?</code></pre>)', r"\1</div>", body, flags=re.S)
        body = re.sub(r'<h([234]) id="([^"]+)">(.*?)</h\1>', lambda m: f'<h{m[1]} id="{m[2]}">{m[3]}<a class="hl" href="#{m[2]}" aria-label="Link to this section">#</a></h{m[1]}>', body)
        body = re.sub(r"<h1([^>]*)>", r"<h1\1 data-split>", body, count=1)
        body = rewrite_links(body, rel)

        # sidebar
        side, last = [], None
        for q in pages:
            if q["group"] != last:
                side.append(f"<h4>{q['group']}</h4>")
                last = q["group"]
            href = root + "docs/" + q["rel"].replace(".md", ".html")
            cur = ' aria-current="page"' if q is p else ""
            side.append(f'<a href="{href}"{cur}>{html.escape(SHORT[q["rel"]])}</a>')
        # toc
        def toc_html(tokens):
            if not tokens:
                return ""
            items = "".join(f'<li><a href="#{t["id"]}">{html.escape(strip_tags(t["name"]))}</a>{toc_html(t["children"])}</li>' for t in tokens)
            return f"<ul>{items}</ul>"
        toks = p["toc"]
        if len(toks) == 1 and toks[0]["level"] == 1:
            toks = toks[0]["children"]
        toc = f'<nav class="toc" aria-label="On this page"><h4>On this page</h4>{toc_html(toks)}</nav>' if toks else '<div></div>'
        prev_p = pages[i - 1] if i > 0 else None
        next_p = pages[i + 1] if i + 1 < len(pages) else None
        pager = '<nav class="pager" aria-label="Previous and next">'
        if prev_p:
            pager += f'<a href="{root}docs/{prev_p["rel"].replace(".md", ".html")}"><small>Previous</small>{html.escape(SHORT[prev_p["rel"]])}</a>'
        if next_p:
            pager += f'<a class="nx" href="{root}docs/{next_p["rel"].replace(".md", ".html")}"><small>Next</small>{html.escape(SHORT[next_p["rel"]])}</a>'
        pager += "</nav>"
        main = f"""<div class="docs-shell">
<aside class="side glass" aria-label="Documentation">{''.join(side)}</aside>
<div class="side-scrim" aria-hidden="true"></div>
<main id="main" class="article glass">
  <button class="menu-btn side-btn" type="button">☰ All docs</button>
  <p class="crumbs"><a href="{root}docs/index.html">Docs</a> / {html.escape(p['group'])}</p>
  <article class="prose">{body}</article>
  <p class="crumbs" style="margin-top:40px"><a href="{REPO}/blob/{BRANCH}/docs/{rel}" target="_blank" rel="noopener">View this page’s source ↗</a></p>
  {pager}
</main>
{toc}
</div>"""
        page = shell(root, f"{p['title']} · EviCortex docs", f"EviCortex documentation: {p['title']}", "docs-page", "docs", main, "Docs")
        (ROOT / out).write_text(page, encoding="utf-8")

        # search index
        url = "docs/" + rel.replace(".md", ".html")
        first = re.search(r"<p>(.*?)</p>", body, re.S)
        index.append({"t": SHORT[rel], "h": p["title"], "u": url, "x": strip_tags(first[1])[:160] if first else ""})
        for m in re.finditer(r'<h([23]) id="([^"]+)">(.*?)<a class="hl".*?</h\1>(.*?)(?=<h[1-3] |\Z)', body, re.S):
            index.append({"t": SHORT[rel], "h": strip_tags(m[3]), "u": f"{url}#{m[2]}", "x": strip_tags(m[4])[:160], "a": 1})

    (ROOT / "assets" / "search-index.js").write_text("window.EV_SEARCH=" + json.dumps(index, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")

    landing = (ROOT / "tools" / "site" / "landing.html").read_text(encoding="utf-8").replace("{root}", "")
    (ROOT / "index.html").write_text(
        shell("", "EviCortex · evidence-preserving temporal memory", "A local-first memory kernel for long-lived AI systems: keep every observation, pack only what is still true.", "landing", "hero", landing, ""),
        encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"built index.html + {len(pages)} docs pages, {len(index)} search entries")


if __name__ == "__main__":
    main()
