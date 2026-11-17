"""
Contract Risk Radar  -  single-file Flask application
=====================================================
Upload a contract PDF -> extract text -> split into clauses -> analyse each clause
with the Gemini API -> score the contract with OUR OWN rules -> show results.

DISCLAIMER: automated suggestions only, NOT legal advice.

SETUP
  1) pip install flask pymupdf google-genai python-dotenv gunicorn
  2) Create a file named .env next to this file:
         GEMINI_API_KEY=your_key_here
         GEMINI_MODEL=gemini-2.5-flash      # change to any Flash model your key supports
         SECRET_KEY=any_long_random_text
  3) python app.py            -> open http://127.0.0.1:5000
  4) Accuracy check:  python app.py evaluate my_labels.csv
         (CSV columns: clause_text,true_risk_level   where level is Low/Medium/High)
  5) Deploy (Render):  Start command =  gunicorn app:app
"""
import csv
import json
import logging
import os
import re
import secrets
import sqlite3
import sys
import time
from datetime import datetime

from flask import (Flask, abort, flash, g, redirect, render_template, request,
                   session, url_for)
from jinja2 import DictLoader
from werkzeug.utils import secure_filename

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pymupdf  # PyMuPDF
except ImportError:
    pymupdf = None


try:
    from google import genai
except ImportError:
    genai = None

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "contract_risk.db"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_PDF_BYTES = 5 * 1024 * 1024      # 5 MB
MAX_CLAUSES = 60                      # caps API usage
BATCH_SIZE = 8                        # clauses per Gemini request
RATE_LIMIT = 10                       # analyses per IP per minute

CATEGORIES = ["Penalty/Fees", "Auto-Renewal", "One-Sided Termination",
              "Non-Compete/Bond", "Hidden Charges", "Dispute/Jurisdiction",
              "Liability/Indemnity", "Payment Terms", "Data/Privacy",
              "Neutral/Other"]
LEVELS = ["Low", "Medium", "High"]

# OUR scoring rules (tune freely): category weight x level multiplier
CATEGORY_WEIGHTS = {"Penalty/Fees": 10, "Auto-Renewal": 8, "One-Sided Termination": 9,
                    "Non-Compete/Bond": 9, "Hidden Charges": 10, "Dispute/Jurisdiction": 5,
                    "Liability/Indemnity": 7, "Payment Terms": 3, "Data/Privacy": 3,
                    "Neutral/Other": 3}
LEVEL_MULTIPLIER = {"High": 3, "Medium": 2, "Low": 0.5}
SCORE_SCALE = 150      # raw points that equal a score of 100 (capped)

# Keyword fallback (used if the API is unavailable) : category -> (regex, level)
KEYWORD_RULES = [
    ("Penalty/Fees", r"penalt|liquidated damages|forfeit|late fee|fine of", "High"),
    ("Auto-Renewal", r"auto[- ]?renew|automatically renew|renews automatically", "High"),
    ("One-Sided Termination", r"sole discretion|terminate[^.]{0,60}without (cause|notice)|at any time without", "High"),
    ("Non-Compete/Bond", r"non[- ]?compete|service bond|bond period|shall not (work|join|engage)", "High"),
    ("Hidden Charges", r"additional charges|processing fee|administrative fee|non[- ]?refundable|surcharge", "Medium"),
    ("Dispute/Jurisdiction", r"jurisdiction|arbitration|governing law|waive[^.]{0,30}right to", "Medium"),
    ("Liability/Indemnity", r"indemnif|hold harmless|liable for all|unlimited liability", "Medium"),
    ("Payment Terms", r"payment (is )?due|interest on|deposit|invoice", "Low"),
    ("Data/Privacy", r"personal data|confidential information|privacy|share[^.]{0,30}third part", "Low"),
]

logging.basicConfig(filename=os.path.join(BASE_DIR, "app.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crr")   # NEVER log contract text or the API key

# --------------------------------------------------------------------------
# DATABASE
# --------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, created_at TEXT,
        overall_score INTEGER, risk_label TEXT, total_clauses INTEGER,
        high_count INTEGER, medium_count INTEGER, low_count INTEGER,
        is_demo INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS clauses(
        id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id INTEGER, clause_number INTEGER,
        text TEXT, category TEXT, risk_level TEXT, reason TEXT,
        suggested_question TEXT, source TEXT,
        FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE);
    """)
    con.commit()
    con.close()


def save_analysis(filename, results, score, label, is_demo=0):
    db = get_db()
    counts = {lv: sum(1 for r in results if r["risk_level"] == lv) for lv in LEVELS}
    cur = db.execute(
        "INSERT INTO analyses(filename,created_at,overall_score,risk_label,total_clauses,"
        "high_count,medium_count,low_count,is_demo) VALUES(?,?,?,?,?,?,?,?,?)",
        (filename, datetime.now().strftime("%Y-%m-%d %H:%M"), score, label, len(results),
         counts["High"], counts["Medium"], counts["Low"], is_demo))
    aid = cur.lastrowid
    db.executemany(
        "INSERT INTO clauses(analysis_id,clause_number,text,category,risk_level,reason,"
        "suggested_question,source) VALUES(?,?,?,?,?,?,?,?)",
        [(aid, r["clause_number"], r["text"], r["category"], r["risk_level"], r["reason"],
          r["suggested_question"], r["source"]) for r in results])
    db.commit()
    return aid

# --------------------------------------------------------------------------
# PDF + CLAUSE SPLITTING
# --------------------------------------------------------------------------

def extract_text(data: bytes) -> str:
    if pymupdf is None:
        raise RuntimeError(
            "PyMuPDF is not installed (pip install pymupdf)."
        )

    doc = pymupdf.open(stream=data, filetype="pdf")

    try:
        text = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()

    return text



CLAUSE_START = re.compile(
    r"^\s*(?:ARTICLE\s+[IVXLC\d]+\b|Clause\s+\d+\b|Section\s+\d+\b|\d+(?:\.\d+)*[.)]\s|"
    r"\([a-z]\)\s|\([ivx]+\)\s)", re.IGNORECASE)


def split_clauses(text: str):
    """Return a list of clause strings (max MAX_CLAUSES)."""
    clauses, current = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if CLAUSE_START.match(line) and current:
            clauses.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        clauses.append(" ".join(current))
    # Fallback: no numbering found -> split on sentences into ~300 char chunks
    if len(clauses) < 3:
        sentences = re.split(r"(?<=[.;])\s+", " ".join(text.split()))
        clauses, buf = [], ""
        for s in sentences:
            buf = (buf + " " + s).strip()
            if len(buf) >= 300:
                clauses.append(buf)
                buf = ""
        if buf:
            clauses.append(buf)
    # Merge very short fragments into the previous clause
    merged = []
    for c in clauses:
        if merged and len(c) < 40:
            merged[-1] += " " + c
        else:
            merged.append(c)
    # A short first fragment (e.g. the contract title) joins the next clause
    if len(merged) > 1 and len(merged[0]) < 40:
        merged[1] = merged[0] + " " + merged[1]
        merged = merged[1:]
    return merged[:MAX_CLAUSES]

# --------------------------------------------------------------------------
# AI ANALYSIS (Gemini) + KEYWORD FALLBACK
# --------------------------------------------------------------------------
PROMPT = f"""You are a careful contract-review assistant for non-lawyers.
For EACH clause below, return ONLY a JSON array (no markdown, no commentary) with one
object per clause:
{{"clause_number": int, "category": one of {CATEGORIES},
 "risk_level": one of {LEVELS},
 "reason": plain English, max 25 words, why it is or is not risky for the signer,
 "suggested_question": one short question the signer should ask before signing, or ""}}
Be conservative: mark High only for clearly one-sided, costly or hard-to-exit terms.
Clauses (JSON):
"""


def keyword_check(num, text):
    for cat, pattern, level in KEYWORD_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return {"clause_number": num, "text": text, "category": cat, "risk_level": level,
                    "reason": f"Keyword check: wording related to {cat.lower()} found.",
                    "suggested_question": f"Can you explain the {cat.lower()} terms in this clause?",
                    "source": "basic"}
    return {"clause_number": num, "text": text, "category": "Neutral/Other", "risk_level": "Low",
            "reason": "No risky wording detected by the basic check.", "suggested_question": "",
            "source": "basic"}


def parse_json_list(raw: str):
    """Repair common formatting problems and return a list."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array")
    return json.loads(raw[start:end + 1])


_client = None


def get_client():
    global _client
    if _client is None and genai is not None and GEMINI_API_KEY:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def call_gemini(batch):
    """batch = [(clause_number, text)] -> dict {clause_number: item} or None."""
    client = get_client()
    if client is None:
        return None
    payload = json.dumps([{"clause_number": n, "text": t[:1500]} for n, t in batch])
    for attempt in range(3):                       # 1 try + 2 retries
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=PROMPT + payload,
                config={"response_mime_type": "application/json", "temperature": 0.1})
            items = parse_json_list(resp.text)
            return {int(i["clause_number"]): i for i in items if "clause_number" in i}
        except Exception as e:                      # log the error TYPE only
            log.warning("Gemini attempt %d failed: %s", attempt + 1, type(e).__name__)
            time.sleep(1.5 * (attempt + 1))
    return None


def analyze_clauses(clauses):
    results = []
    numbered = list(enumerate(clauses, start=1))
    for i in range(0, len(numbered), BATCH_SIZE):
        batch = numbered[i:i + BATCH_SIZE]
        ai = call_gemini(batch) or {}
        for num, text in batch:
            item = ai.get(num)
            if item and item.get("category") in CATEGORIES and item.get("risk_level") in LEVELS:
                results.append({"clause_number": num, "text": text,
                                "category": item["category"], "risk_level": item["risk_level"],
                                "reason": str(item.get("reason", ""))[:300],
                                "suggested_question": str(item.get("suggested_question", ""))[:300],
                                "source": "ai"})
            else:
                results.append(keyword_check(num, text))
    return results

# --------------------------------------------------------------------------
# OUR OWN SCORING RULES
# --------------------------------------------------------------------------
def score_contract(results):
    raw, contrib = 0.0, {}
    for r in results:
        pts = CATEGORY_WEIGHTS.get(r["category"], 3) * LEVEL_MULTIPLIER[r["risk_level"]]
        raw += pts
        contrib[r["category"]] = contrib.get(r["category"], 0) + pts
    score = min(100, round(raw / SCORE_SCALE * 100))
    label = "Low" if score <= 30 else "Medium" if score <= 60 else "High"
    return score, label, sorted(contrib.items(), key=lambda kv: -kv[1])[:5]

# --------------------------------------------------------------------------
# FLASK APP
# --------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = MAX_PDF_BYTES
_hits = {}


def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


@app.before_request
def csrf_protect():
    if request.method == "POST":
        sent = request.form.get("_csrf", "")
        if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
            abort(400)


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def rate_ok(ip):
    now = time.time()
    _hits[ip] = [t for t in _hits.get(ip, []) if now - t < 60]
    if len(_hits[ip]) >= RATE_LIMIT:
        return False
    _hits[ip].append(now)
    return True


@app.get("/")
def index():
    return render_template("index.html", ai_ready=get_client() is not None)


@app.post("/analyze")
def analyze():
    if not rate_ok(request.remote_addr):
        flash("Too many requests. Please wait a minute and try again.", "error")
        return redirect(url_for("index"))
    if not request.form.get("agree"):
        flash("Please tick the box to confirm you understand this is not legal advice.", "error")
        return redirect(url_for("index"))
    f = request.files.get("pdf")
    if not f or not f.filename:
        flash("Please choose a PDF file.", "error")
        return redirect(url_for("index"))
    if not f.filename.lower().endswith(".pdf"):
        flash("Only PDF files are allowed.", "error")
        return redirect(url_for("index"))
    data = f.read()                                  # processed in memory, never saved
    if not data.startswith(b"%PDF"):
        flash("That file does not look like a real PDF.", "error")
        return redirect(url_for("index"))
    try:
        text = extract_text(data)
    except Exception as e:
        log.warning("PDF extraction failed: %s", type(e).__name__)
        flash("Could not read that PDF. It may be damaged or password protected.", "error")
        return redirect(url_for("index"))
    if len(text.strip()) < 50:
        flash("No readable text found. This may be a scanned image PDF; please upload a text-based PDF.", "error")
        return redirect(url_for("index"))
    results = analyze_clauses(split_clauses(text))
    score, label, _ = score_contract(results)
    aid = save_analysis(secure_filename(f.filename) or "contract.pdf", results, score, label)
    return redirect(url_for("results", aid=aid))


@app.get("/results/<int:aid>")
def results(aid):
    db = get_db()
    a = db.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    if a is None:
        abort(404)
    rows = [dict(r) for r in db.execute(
        "SELECT * FROM clauses WHERE analysis_id=? ORDER BY clause_number", (aid,))]
    _, _, breakdown = score_contract(rows)
    order = {"High": 0, "Medium": 1, "Low": 2}
    risky = sorted([r for r in rows if r["risk_level"] in ("High", "Medium")],
                   key=lambda r: (order[r["risk_level"]], r["clause_number"]))
    other = [r for r in rows if r["risk_level"] == "Low"]
    questions = []
    for r in risky:
        q = r["suggested_question"]
        if q and q not in questions:
            questions.append(q)
    return render_template("results.html", a=a, risky=risky, other=other,
                           questions=questions[:10], breakdown=breakdown,
                           basic=any(r["source"] == "basic" for r in rows))


@app.get("/history")
def history():
    rows = get_db().execute("SELECT * FROM analyses ORDER BY id DESC").fetchall()
    return render_template("history.html", rows=rows)


@app.post("/delete/<int:aid>")
def delete(aid):
    db = get_db()
    db.execute("DELETE FROM clauses WHERE analysis_id=?", (aid,))
    db.execute("DELETE FROM analyses WHERE id=?", (aid,))
    db.commit()
    flash("Analysis deleted.", "ok")
    return redirect(url_for("history"))


@app.get("/insights")
def insights():
    include_demo = request.args.get("demo") == "1"
    db = get_db()
    where = "" if include_demo else "WHERE is_demo=0"
    dist = {r["risk_label"]: r["n"] for r in db.execute(
        f"SELECT risk_label, COUNT(*) n FROM analyses {where} GROUP BY risk_label")}
    cats = db.execute(
        "SELECT c.category, COUNT(*) n FROM clauses c JOIN analyses a ON a.id=c.analysis_id "
        f"WHERE c.risk_level IN ('High','Medium') {'' if include_demo else 'AND a.is_demo=0'} "
        "GROUP BY c.category ORDER BY n DESC").fetchall()
    trend = db.execute(
        f"SELECT created_at, overall_score FROM analyses {where} ORDER BY id").fetchall()
    data = {"dist": [dist.get(l, 0) for l in LEVELS],
            "cat_labels": [c["category"] for c in cats], "cat_values": [c["n"] for c in cats],
            "t_labels": [t["created_at"] for t in trend], "t_values": [t["overall_score"] for t in trend]}
    return render_template("insights.html", data=data, include_demo=include_demo,
                           total=sum(data["dist"]))


@app.get("/about")
def about():
    return render_template("about.html", weights=CATEGORY_WEIGHTS, model=GEMINI_MODEL)


# ---- demo data (labelled "Demo", excluded from Insights by default) -------
DEMO_CONTRACTS = {
    "Demo - Low risk lease": [
        "1. Rent is INR 15,000 per month, payment due on the 5th of each month.",
        "2. Either party may end this agreement with 30 days written notice.",
        "3. The security deposit is fully refundable within 15 days of move-out.",
        "4. Disputes will be discussed in good faith before any legal step."],
    "Demo - Medium risk internship": [
        "1. The intern will work 40 hours per week and receive a monthly stipend.",
        "2. An administrative fee of INR 2,000 is non-refundable after joining.",
        "3. Disputes are subject to the jurisdiction of the company's home city courts.",
        "4. The intern shall keep all confidential information private."],
    "Demo - High risk employment": [
        "1. The employee must serve a service bond period of 3 years.",
        "2. A penalty of INR 3,00,000 applies if the employee leaves early.",
        "3. The company may terminate this agreement at any time without notice at its sole discretion.",
        "4. This agreement will automatically renew every year unless cancelled in writing.",
        "5. The employee shall not work for any competitor under a non-compete for 2 years."],
}


@app.post("/demo/load")
def demo_load():
    db = get_db()
    if db.execute("SELECT 1 FROM analyses WHERE is_demo=1").fetchone():
        flash("Demo data is already loaded.", "ok")
        return redirect(url_for("history"))
    for name, clauses in DEMO_CONTRACTS.items():
        results = [keyword_check(i, c) for i, c in enumerate(clauses, start=1)]
        score, label, _ = score_contract(results)
        save_analysis(name, results, score, label, is_demo=1)
    flash("Demo analyses loaded (labelled Demo).", "ok")
    return redirect(url_for("history"))


@app.post("/demo/clear")
def demo_clear():
    db = get_db()
    db.execute("DELETE FROM clauses WHERE analysis_id IN (SELECT id FROM analyses WHERE is_demo=1)")
    db.execute("DELETE FROM analyses WHERE is_demo=1")
    db.commit()
    flash("Demo data cleared.", "ok")
    return redirect(url_for("history"))


@app.errorhandler(413)
def too_big(_e):
    flash("File too large. Maximum size is 5 MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(_e):
    return render_template("404.html"), 404


# --------------------------------------------------------------------------
# TEMPLATES (kept inside this file so the whole app is ONE file)
# --------------------------------------------------------------------------
BASE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{% block title %}Contract Risk Radar{% endblock %}</title>
<style>
:root{--bg:#f5f7fb;--card:#fff;--ink:#1c2333;--mute:#5b6478;--line:#e2e6ef;--brand:#2b4acb;
--low:#16794a;--lowbg:#e3f5ec;--med:#8a5a00;--medbg:#fff2d6;--high:#b3261e;--highbg:#fde7e5}
@media (prefers-color-scheme:dark){:root{--bg:#10131c;--card:#1a1f2d;--ink:#e8ebf4;--mute:#9aa3b8;--line:#2b3247;--brand:#8da2ff;
--low:#6fdba3;--lowbg:#12301f;--med:#ffcf70;--medbg:#3a2c0b;--high:#ff8f87;--highbg:#401a17}}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink);line-height:1.5}
header{background:var(--card);border-bottom:1px solid var(--line)}
.wrap{max-width:980px;margin:0 auto;padding:0 16px}
nav{display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding:14px 0}
nav .logo{font-weight:700;font-size:1.15rem;margin-right:auto;color:var(--ink);text-decoration:none}
nav a{color:var(--mute);text-decoration:none;font-weight:500}nav a:hover{color:var(--brand)}
main{padding:24px 0 60px}h1{font-size:1.7rem;margin:.2em 0 .4em}h2{font-size:1.2rem;margin:1.4em 0 .5em}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin:12px 0}
.btn{background:var(--brand);color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:1rem;cursor:pointer;text-decoration:none;display:inline-block}
.btn.light{background:transparent;color:var(--brand);border:1px solid var(--brand)}
.btn.danger{background:transparent;color:var(--high);border:1px solid var(--high);padding:5px 10px;font-size:.85rem}
.btn:disabled{opacity:.6;cursor:wait}
.badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.8rem;font-weight:700}
.High{background:var(--highbg);color:var(--high)}.Medium{background:var(--medbg);color:var(--med)}.Low{background:var(--lowbg);color:var(--low)}
.demo{background:var(--line);color:var(--mute);margin-left:6px}
.flash{padding:10px 14px;border-radius:8px;margin:10px 0}.flash.error{background:var(--highbg);color:var(--high)}.flash.ok{background:var(--lowbg);color:var(--low)}
.disclaimer{background:var(--medbg);color:var(--med);border-radius:8px;padding:8px 12px;font-size:.9rem;margin:12px 0}
.drop{border:2px dashed var(--line);border-radius:12px;padding:34px;text-align:center;color:var(--mute);cursor:pointer}
.drop.over{border-color:var(--brand);background:var(--card)}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.stat{text-align:center}.stat b{font-size:1.8rem;display:block}
.gauge{width:150px;height:150px;border-radius:50%;display:grid;place-items:center;margin:auto}
.gauge div{width:112px;height:112px;border-radius:50%;background:var(--card);display:grid;place-items:center;font-size:2rem;font-weight:800}
details summary{cursor:pointer;color:var(--brand)}blockquote{margin:8px 0;padding:8px 12px;border-left:3px solid var(--line);color:var(--mute)}
footer{color:var(--mute);font-size:.85rem;text-align:center;padding:20px}
@media print{header,.noprint,footer{display:none}body{background:#fff;color:#000}.card{break-inside:avoid}}
</style></head><body>
<header><div class="wrap"><nav>
<a class="logo" href="{{ url_for('index') }}">&#128225; Contract Risk Radar</a>
<a href="{{ url_for('index') }}">Analyze</a><a href="{{ url_for('history') }}">History</a>
<a href="{{ url_for('insights') }}">Insights</a><a href="{{ url_for('about') }}">About</a>
</nav></div></header>
<main><div class="wrap">
{% with msgs = get_flashed_messages(with_categories=true) %}{% for cat, m in msgs %}<div class="flash {{ cat }}">{{ m }}</div>{% endfor %}{% endwith %}
<div class="disclaimer">Contract Risk Radar provides automated suggestions, not legal advice.</div>
{% block body %}{% endblock %}
</div></main>
<footer>Contract Risk Radar &middot; automated suggestions, not legal advice</footer>
</body></html>"""

INDEX_HTML = """{% extends "base.html" %}{% block body %}
<h1>Check a contract before you sign it</h1>
<p>Upload a contract PDF (rental, internship, employment or freelance). We highlight risky clauses and give a 0&ndash;100 risk score.</p>
{% if not ai_ready %}<div class="flash error">Gemini is not configured (missing key or library). The app will use the basic keyword check instead.</div>{% endif %}
<form class="card" id="f" method="post" action="{{ url_for('analyze') }}" enctype="multipart/form-data">
<input type="hidden" name="_csrf" value="{{ csrf_token() }}">
<label class="drop" id="drop" for="pdf"><div id="dtext"><b>Drag &amp; drop a PDF here</b> or click to choose (max 5 MB)</div></label>
<input type="file" id="pdf" name="pdf" accept="application/pdf,.pdf" style="position:absolute;left:-9999px">
<p><label><input type="checkbox" name="agree" required> I understand this is not legal advice and that the contract text is sent to the Gemini API for analysis.</label></p>
<button class="btn" id="go" type="submit">Analyze contract</button>
</form>
<script>
const drop=document.getElementById('drop'),inp=document.getElementById('pdf'),t=document.getElementById('dtext');
function show(){if(inp.files.length){t.textContent=inp.files[0].name}}
inp.addEventListener('change',show);
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length){inp.files=ev.dataTransfer.files;show()}});
document.getElementById('f').addEventListener('submit',()=>{const b=document.getElementById('go');b.disabled=true;b.textContent='Analyzing clauses...'});
</script>{% endblock %}"""

RESULTS_HTML = """{% extends "base.html" %}{% block title %}Results{% endblock %}{% block body %}
<h1>{{ a.filename }}{% if a.is_demo %}<span class="badge demo">Demo</span>{% endif %}</h1>
<p style="color:var(--mute)">Analysed {{ a.created_at }} &middot; {{ a.total_clauses }} clauses</p>
{% if basic %}<div class="flash error">Some clauses were checked with the basic keyword check (AI unavailable), so results are less detailed.</div>{% endif %}
<div class="card grid">
 <div class="stat"><div class="gauge" style="background:conic-gradient({{ 'var(--low)' if a.risk_label=='Low' else 'var(--med)' if a.risk_label=='Medium' else 'var(--high)' }} {{ a.overall_score*3.6 }}deg,var(--line) 0)"><div>{{ a.overall_score }}</div></div>
 <p><span class="badge {{ a.risk_label }}">{{ a.risk_label }} risk</span></p></div>
 <div class="stat"><b style="color:var(--high)">{{ a.high_count }}</b>High</div>
 <div class="stat"><b style="color:var(--med)">{{ a.medium_count }}</b>Medium</div>
 <div class="stat"><b style="color:var(--low)">{{ a.low_count }}</b>Low</div>
</div>
{% if breakdown %}<div class="card"><b>Why this score?</b> Biggest contributors:
<ul>{% for cat, pts in breakdown %}<li>{{ cat }} ({{ pts|round(1) }} points)</li>{% endfor %}</ul></div>{% endif %}
<h2>Risky clauses</h2>
{% for r in risky %}<div class="card"><span class="badge {{ r.risk_level }}">{{ r.risk_level }}</span> <b>{{ r.category }}</b> &middot; Clause {{ r.clause_number }}
{% if r.source=='basic' %}<span class="badge demo">basic check</span>{% endif %}
<p>{{ r.reason }}</p><details><summary>Show original clause</summary><blockquote>{{ r.text }}</blockquote></details></div>
{% else %}<div class="card">No medium or high risk clauses were found.</div>{% endfor %}
<h2>Questions to ask before signing</h2>
<div class="card">{% if questions %}<ol>{% for q in questions %}<li>{{ q }}</li>{% endfor %}</ol>{% else %}<p>No questions suggested.</p>{% endif %}</div>
{% if other %}<details class="card"><summary>Other clauses ({{ other|length }} low risk)</summary>
{% for r in other %}<p><span class="badge Low">Low</span> <b>{{ r.category }}</b> &middot; Clause {{ r.clause_number }}<br>{{ r.reason }}</p>{% endfor %}</details>{% endif %}
<p class="noprint"><button class="btn" onclick="window.print()">Download / print report</button>
<a class="btn light" href="{{ url_for('index') }}">Analyze another</a></p>{% endblock %}"""

HISTORY_HTML = """{% extends "base.html" %}{% block title %}History{% endblock %}{% block body %}
<h1>History</h1>
<form method="post" class="noprint" style="display:flex;gap:8px;flex-wrap:wrap">
<input type="hidden" name="_csrf" value="{{ csrf_token() }}">
<button class="btn light" formaction="{{ url_for('demo_load') }}">Load demo data</button>
<button class="btn light" formaction="{{ url_for('demo_clear') }}">Clear demo data</button></form>
<div class="card" style="overflow-x:auto">
{% if rows %}<table><tr><th>File</th><th>Date</th><th>Score</th><th>Risk</th><th></th></tr>
{% for r in rows %}<tr><td>{{ r.filename }}{% if r.is_demo %}<span class="badge demo">Demo</span>{% endif %}</td><td>{{ r.created_at }}</td>
<td>{{ r.overall_score }}</td><td><span class="badge {{ r.risk_label }}">{{ r.risk_label }}</span></td>
<td style="white-space:nowrap"><a class="btn light" style="padding:5px 10px;font-size:.85rem" href="{{ url_for('results', aid=r.id) }}">View</a>
<form method="post" action="{{ url_for('delete', aid=r.id) }}" style="display:inline" onsubmit="return confirm('Delete this analysis?')">
<input type="hidden" name="_csrf" value="{{ csrf_token() }}"><button class="btn danger">Delete</button></form></td></tr>{% endfor %}</table>
{% else %}<p>No analyses yet. Upload a contract or load the demo data.</p>{% endif %}</div>{% endblock %}"""

INSIGHTS_HTML = """{% extends "base.html" %}{% block title %}Insights{% endblock %}{% block body %}
<h1>Insights</h1>
<p><a class="btn light" href="{{ url_for('insights', demo=(0 if include_demo else 1)) }}">{{ 'Hide' if include_demo else 'Include' }} demo data</a></p>
{% if total == 0 %}<div class="card">No data yet. Analyze a contract first{% if not include_demo %} (or include demo data){% endif %}.</div>
{% else %}
<div class="card"><h2>Risk distribution</h2><canvas id="c1" height="120"></canvas></div>
<div class="card"><h2>Most common risky categories</h2><canvas id="c2" height="140"></canvas></div>
<div class="card"><h2>Score over time</h2><canvas id="c3" height="120"></canvas></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
const D={{ data|tojson }};
new Chart(document.getElementById('c1'),{type:'doughnut',data:{labels:['Low','Medium','High'],datasets:[{data:D.dist,backgroundColor:['#16794a','#d69a00','#b3261e']}]}});
new Chart(document.getElementById('c2'),{type:'bar',data:{labels:D.cat_labels,datasets:[{label:'Medium/High clauses',data:D.cat_values,backgroundColor:'#2b4acb'}]},options:{indexAxis:'y'}});
new Chart(document.getElementById('c3'),{type:'line',data:{labels:D.t_labels,datasets:[{label:'Risk score',data:D.t_values,borderColor:'#2b4acb',tension:.3}]},options:{scales:{y:{min:0,max:100}}}});
</script>{% endif %}{% endblock %}"""

ABOUT_HTML = """{% extends "base.html" %}{% block title %}About{% endblock %}{% block body %}
<h1>About</h1>
<div class="card"><h2>How it works</h2><ol>
<li>Your PDF is read in memory and the text is extracted (the file itself is not saved).</li>
<li>The text is split into clauses.</li>
<li>Each clause is analysed by the Gemini API ({{ model }}) for category, risk level, a plain-English reason and a question to ask.</li>
<li>Our own scoring rules turn those results into a 0&ndash;100 score: category weight &times; level multiplier (High &times;3, Medium &times;2, Low &times;0.5).</li></ol>
<h2>Category weights</h2><ul>{% for k, v in weights.items() %}<li>{{ k }}: {{ v }}</li>{% endfor %}</ul></div>
<div class="card"><h2>Limitations</h2><ul>
<li>This is an automated tool and can be wrong. It is not legal advice; consult a qualified professional for important contracts.</li>
<li>Scanned (image-only) PDFs are not supported. Very long contracts are limited to the first 60 clauses.</li>
<li>If the AI is unavailable, a simpler keyword check is used and clearly labelled.</li></ul></div>
<div class="card"><h2>Privacy</h2><p>Contract text is sent to the Gemini API for analysis. Only the extracted clause text and results are stored in a local database for the History page; you can delete any analysis at any time. Please use anonymised or fictional contracts for testing.</p></div>
{% endblock %}"""

NOT_FOUND_HTML = """{% extends "base.html" %}{% block body %}<h1>Page not found</h1>
<p><a class="btn" href="{{ url_for('index') }}">Back to home</a></p>{% endblock %}"""

app.jinja_env.loader = DictLoader({
    "base.html": BASE_HTML, "index.html": INDEX_HTML, "results.html": RESULTS_HTML,
    "history.html": HISTORY_HTML, "insights.html": INSIGHTS_HTML,
    "about.html": ABOUT_HTML, "404.html": NOT_FOUND_HTML})
app.jinja_env.globals["csrf_token"] = csrf_token

init_db()

# --------------------------------------------------------------------------
# ACCURACY EVALUATION:  python app.py evaluate my_labels.csv
# --------------------------------------------------------------------------
def evaluate(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [(r["clause_text"], r["true_risk_level"].strip().capitalize())
                for r in csv.DictReader(fh)]
    preds = analyze_clauses([t for t, _ in rows])
    pairs = [(true, p["risk_level"]) for (_, true), p in zip(rows, preds)]
    n = len(pairs)
    acc = sum(1 for t, p in pairs if t == p) / n
    tp = sum(1 for t, p in pairs if t == "High" and p == "High")
    fp = sum(1 for t, p in pairs if t != "High" and p == "High")
    fn = sum(1 for t, p in pairs if t == "High" and p != "High")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"Clauses: {n}\nAccuracy: {acc:.1%}\nHigh-risk precision: {prec:.1%}\nHigh-risk recall: {rec:.1%}")
    print("Source:", "Gemini" if any(p["source"] == "ai" for p in preds) else "basic keyword check only")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "evaluate":
        evaluate(sys.argv[2])
    else:
        app.run(debug=False, host="127.0.0.1", port=int(os.getenv("PORT", 5000)))