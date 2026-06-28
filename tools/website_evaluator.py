#!/usr/bin/env python3
"""
website_evaluator.py — consistency evaluator for the Falcon dashboard website.

Checks the served www/ pages against the Flask API and against each other, reporting
inconsistencies a human (or CI) should fix. Pure stdlib; run on the host:

    python3 tools/website_evaluator.py            # human report
    python3 tools/website_evaluator.py --json      # machine-readable findings (for issue filing)

Checks:
  C1 BROKEN_ENDPOINT  — a page calls /api/... with no matching Flask route (static or dynamic)
  C2 ORPHAN_ENDPOINT  — an /api route no page calls (informational; not necessarily a bug)
  C3 BROKEN_LINK      — an <a href> to a page/route the server does not serve
  C4 NAV_INCONSISTENT — nav link sets differ across pages
  C5 OFF_PALETTE      — hex colors outside the molokai palette (theme drift)
  C6 NO_TITLE         — page missing a <title>
"""
import os, re, glob, json, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WWW = os.path.join(ROOT, "src", "falcon_trader", "www")
SERVER = os.path.join(ROOT, "src", "falcon_trader", "dashboard_server.py")

# Molokai palette (theme) — anything else flagged as drift. Greys/black/white tolerated.
MOLOKAI = {
    "#1b1d1e", "#161718", "#272822", "#3e3d32", "#49483e", "#75715e", "#f8f8f2",
    "#66d9ef", "#4fb3cc", "#a6e22e", "#f92672", "#f97199", "#e6db74", "#fd971f",
    "#ae81ff", "#cfcfc2", "#a59f85", "#3a6ea5", "#1b3a1b", "#3a1b1b", "#5a7a1e",
    "#4d5a1f", "#8a1538", "#a01243", "#8a5a1f",
}
GREY_OK = re.compile(r"^#(?:fff|000|[0-9a-f]{0,2})$")  # white/black short forms tolerated


def canon(path):
    """Canonicalize a URL path: drop query, map param/template segments to '*'."""
    path = path.split("?")[0].split("#")[0]
    if not path.startswith("/"):
        return None
    segs = []
    for s in path.strip("/").split("/"):
        if not s:
            continue
        if "<" in s or "${" in s or s.isdigit() or s.startswith("$"):
            segs.append("*")
        else:
            segs.append(s)
    return "/" + "/".join(segs)


def server_routes():
    routes = set()
    src = open(SERVER).read()
    for m in re.finditer(r"@app\.route\(\s*['\"]([^'\"]+)['\"]", src):
        c = canon(m.group(1))
        if c:
            routes.add(c)
    # dynamic routes registered via falcon_core.backtesting.results_api.create_api_routes
    rapi = glob.glob(os.path.join(ROOT, "..", "falcon-core", "src", "falcon_core",
                                  "backtesting", "results_api.py"))
    for f in rapi:
        for m in re.finditer(r"@app\.route\(\s*['\"]([^'\"]+)['\"]", open(f).read()):
            c = canon(m.group(1))
            if c:
                routes.add(c)
    return routes


def page_calls(html):
    calls = set()
    for m in re.finditer(r"(?:fetch|EventSource)\(\s*[`'\"]([^`'\"]+)", html):
        c = canon(m.group(1).replace("${API_BASE}", ""))
        if c and c.startswith("/api"):
            calls.add(c)
    # also count any quoted /api/... literal as a UI surface (registries, endpoint lists)
    for m in re.finditer(r"""['"`](/api/[^'"`]*)['"`]""", html):
        c = canon(m.group(1).replace("${API_BASE}", ""))
        if c and c.startswith("/api"):
            calls.add(c)
    return calls


def page_links(html):
    return {m.group(1) for m in re.finditer(r'href="([^"]+)"', html)
            if not m.group(1).startswith(("http", "#", "mailto:"))
            and "${" not in m.group(1) and "{{" not in m.group(1)}  # skip dynamic/templated hrefs


def nav_links(html):
    nav = re.search(r"<nav[^>]*>(.*?)</nav>", html, re.S)
    if not nav:
        return None
    return tuple(sorted(re.findall(r'href="([^"]+)"', nav.group(1))))


def main():
    pages = sorted(glob.glob(os.path.join(WWW, "*.html")))
    routes = server_routes()
    findings = []

    # served page paths (for link checking): "/", "/trading", "<name>.html" all map to files
    served = set(routes)
    for p in pages:
        served.add("/" + os.path.basename(p))

    all_calls = {}
    navsets = {}
    for p in pages:
        name = os.path.basename(p)
        html = open(p).read()
        # C1 broken endpoint
        calls = page_calls(html)
        all_calls[name] = calls
        for c in calls:
            # OK if exact match, or a prefix of a real route (dynamically-built URL, e.g. '/api/bot/'+action)
            if c not in routes and not any(r == c or r.startswith(c + "/") for r in routes):
                findings.append(("C1", "BROKEN_ENDPOINT", name,
                                 f"calls API `{c}` but no Flask route matches"))
        # C3 broken link
        for link in page_links(html):
            cl = canon(link) or link
            if cl not in served and ("." in os.path.basename(link) or link.startswith("/")):
                # allow served routes and existing html files
                if cl not in served and ("/" + os.path.basename(link)) not in served:
                    findings.append(("C3", "BROKEN_LINK", name,
                                     f"links to `{link}` which the server does not serve"))
        # C5 off-palette
        hexes = set(h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", html))
        for h in sorted(hexes):
            if h not in MOLOKAI and not GREY_OK.match(h):
                findings.append(("C5", "OFF_PALETTE", name, f"non-molokai color `{h}`"))
        # C6 title
        if not re.search(r"<title>.*?</title>", html, re.S):
            findings.append(("C6", "NO_TITLE", name, "missing <title>"))
        navsets[name] = nav_links(html)

    # C2 orphan endpoints
    called = set().union(*all_calls.values()) if all_calls else set()
    for r in sorted(routes):
        if r.startswith("/api") and r not in called and "*" not in r:
            findings.append(("C2", "ORPHAN_ENDPOINT", "(server)",
                             f"route `{r}` is not called by any page"))

    # C4 nav inconsistency
    present = {n: s for n, s in navsets.items() if s is not None}
    if len(set(present.values())) > 1:
        from collections import Counter
        common = Counter(present.values()).most_common(1)[0][0]
        for n, s in present.items():
            if s != common:
                missing = set(common) - set(s); extra = set(s) - set(common)
                findings.append(("C4", "NAV_INCONSISTENT", n,
                                 f"nav differs from majority — missing {sorted(missing)} extra {sorted(extra)}"))

    if "--json" in sys.argv:
        print(json.dumps([{"check": c, "type": t, "page": p, "detail": d}
                          for c, t, p, d in findings], indent=2))
        return

    print(f"FALCON WEBSITE EVALUATOR — {len(pages)} pages, {len(routes)} routes, "
          f"{len(findings)} findings\n")
    by = {}
    for c, t, p, d in findings:
        by.setdefault(t, []).append((p, d))
    for t in sorted(by):
        print(f"## {t} ({len(by[t])})")
        for p, d in by[t]:
            print(f"  - [{p}] {d}")
        print()
    sev = sum(1 for c, *_ in findings if c in ("C1", "C3", "C4"))
    print(f"actionable (broken/inconsistent): {sev} · informational (orphan/palette): {len(findings)-sev}")


if __name__ == "__main__":
    main()
