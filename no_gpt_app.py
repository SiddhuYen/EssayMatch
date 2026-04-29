import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from functools import wraps

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask, jsonify, redirect, render_template,
    request, session, url_for
)
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from openai import OpenAI

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.path.join(INSTANCE_DIR, "app.db")

DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SCHOLARSHIP_CSV = os.path.join(DATA_DIR, "scholarships_parsed.csv")
SCHOLARSHIP_EMB_NPY = os.path.join(DATA_DIR, "scholarship_embeddings.npy")

MIN_ESSAY_WORDS = 150
TOP_K = 50
GPT_RERANK_LIMIT = 10
GPT_ESSAY_CANDIDATES = 5
MIN_RESULTS_TO_SHOW = 15

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
]

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_PROJECT_NUMBER = os.getenv("GOOGLE_PROJECT_NUMBER", "")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev_secret_change_me")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

CLIENT_CONFIG = {
    "web": {
        "client_id": GOOGLE_CLIENT_ID,
        "project_id": "scholarship-matcher-local",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uris": [REDIRECT_URI],
        "javascript_origins": ["http://localhost:8000"],
    }
}

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__, template_folder="frontend", instance_relative_config=True)
app.secret_key = FLASK_SECRET_KEY


def file_exists_and_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def download_from_r2_if_needed():
    remote_to_local = {
        "scholarships_parsed.csv": SCHOLARSHIP_CSV,
        "scholarship_embeddings.npy": SCHOLARSHIP_EMB_NPY,
    }

    missing = [p for p in remote_to_local.values() if not file_exists_and_nonempty(p)]

    if not missing:
        return

    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket = os.getenv("R2_BUCKET")

    if not all([account_id, access_key, secret_key, bucket]):
        raise RuntimeError(
            "Scholarship data files not found locally and R2 env vars are missing.\n"
            "Required local files:\n"
            "- data/scholarships_parsed.csv\n"
            "- data/scholarship_embeddings.npy\n\n"
            "Or required R2 vars:\n"
            "R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET\n"
            f"Missing files: {missing}"
        )

    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    for remote_name, local_path in remote_to_local.items():
        if not file_exists_and_nonempty(local_path):
            print(f"Downloading {remote_name} from R2...")
            s3.download_file(bucket, remote_name, local_path)
            print(f"Saved to {local_path}")


def ensure_required_local_files():
    required = [SCHOLARSHIP_CSV, SCHOLARSHIP_EMB_NPY]
    missing = [p for p in required if not file_exists_and_nonempty(p)]

    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}")


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


download_from_r2_if_needed()
ensure_required_local_files()

scholarship_df = pd.read_csv(SCHOLARSHIP_CSV)
scholarship_embeddings = np.load(SCHOLARSHIP_EMB_NPY)

print(f"Scholarships loaded: {len(scholarship_df)}")
print(f"Embedding shape: {scholarship_embeddings.shape}")

if len(scholarship_df) != len(scholarship_embeddings):
    raise ValueError(
        f"Mismatch: {len(scholarship_df)} scholarships vs "
        f"{len(scholarship_embeddings)} embeddings"
    )

scholarship_embeddings = normalize_rows(scholarship_embeddings)


def get_db():
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            google_sub TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            name TEXT,
            picture TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS google_tokens (
            user_id INTEGER PRIMARY KEY,
            access_token TEXT,
            refresh_token TEXT,
            token_expiry DATETIME,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS imported_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            google_file_id TEXT NOT NULL,
            title TEXT NOT NULL,
            mime_type TEXT,
            content TEXT NOT NULL,
            imported_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, google_file_id),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()


init_db()


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


@app.after_request
def add_headers(response):
    response.headers["Cache-Control"] = "no-store"
    return response


def clean_expiry_for_storage(expiry):
    if not expiry:
        return None
    if expiry.tzinfo is not None:
        expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)
    return expiry.isoformat()


def parse_expiry_from_storage(value):
    if not value:
        return None
    expiry = datetime.fromisoformat(value)
    if expiry.tzinfo is not None:
        expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)
    return expiry


def build_flow(state=None):
    return Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=SCOPES,
        state=state,
        redirect_uri=REDIRECT_URI,
    )


def get_user_credentials(user_id):
    conn = get_db()
    row = conn.execute(
        """
        SELECT access_token, refresh_token, token_expiry
        FROM google_tokens
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()
    conn.close()

    if not row:
        raise RuntimeError("No Google token found. Please sign in again.")

    creds = Credentials(
        token=row["access_token"],
        refresh_token=row["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=SCOPES,
        expiry=parse_expiry_from_storage(row["token_expiry"]),
    )

    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest())
        conn = get_db()
        conn.execute(
            """
            UPDATE google_tokens
            SET access_token = ?, token_expiry = ?
            WHERE user_id = ?
            """,
            (creds.token, clean_expiry_for_storage(creds.expiry), user_id),
        )
        conn.commit()
        conn.close()

    if not creds.valid:
        raise RuntimeError("Google credentials invalid. Please log out and sign in again.")

    return creds


def extract_text_from_element(element):
    parts = []

    if paragraph := element.get("paragraph"):
        for pe in paragraph.get("elements", []):
            if text_run := pe.get("textRun"):
                parts.append(text_run.get("content", ""))

    if table := element.get("table"):
        for row in table.get("tableRows", []):
            for cell in row.get("tableCells", []):
                for content in cell.get("content", []):
                    parts.append(extract_text_from_element(content))

    if toc := element.get("tableOfContents"):
        for content in toc.get("content", []):
            parts.append(extract_text_from_element(content))

    return "".join(parts)


def list_google_docs_in_folder(drive, folder_id):
    docs = []

    def list_children(parent_id):
        items = []
        page_token = None

        while True:
            resp = drive.files().list(
                q=f"'{parent_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()

            items.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")

            if not page_token:
                break

        return items

    for item in list_children(folder_id):
        if item["mimeType"] == GOOGLE_DOC_MIME:
            docs.append(item)
        elif item["mimeType"] == GOOGLE_FOLDER_MIME:
            docs.extend(list_google_docs_in_folder(drive, item["id"]))

    return docs


def clean_json_value(value):
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return value


def word_count(text):
    return len(str(text).split())


def split_prompt_response(text):
    text = str(text).strip()

    q_index = text.find("?")
    if 20 <= q_index <= 600:
        return text[:q_index + 1].strip(), text[q_index + 1:].strip()

    match = re.search(
        r"\([^)]*(word|words|character|characters)[^)]*\)",
        text,
        re.IGNORECASE,
    )

    if match and match.end() <= 700:
        return text[:match.end()].strip(), text[match.end():].strip()

    return text[:300].strip(), text.strip()


def is_eligible(qualifications, profile):
    q = str(qualifications).lower()
    is_citizen = profile.get("us_citizen", True)
    level = str(profile.get("level", "undergraduate")).lower().strip()
    state = str(profile.get("state", "")).lower().strip()

    if not is_citizen:
        citizen_required = (
            "must be a u.s. citizen" in q
            or "must be united states citizen" in q
            or "citizenship: united states only" in q
            or "u.s. citizens only" in q
        )
        intl_allowed = (
            "non-u.s." in q
            or "international" in q
            or "all countries" in q
            or "non-us" in q
        )
        if citizen_required and not intl_allowed:
            return False

    if level == "undergraduate":
        if "graduate students only" in q:
            return False
        if "ph.d. required" in q or "must have a ph.d" in q or "must hold a ph.d" in q:
            return False

    if level == "graduate":
        if "undergraduate students only" in q or "undergraduates only" in q:
            return False
        if "incoming freshman only" in q or "high school seniors only" in q:
            return False

    if state:
        state_patterns = [
            r"must be a resident of ([a-z\s]+?)(?:\s+who|\s+and|\s+to|;|\.|,)",
            r"must be a ([a-z\s]+?) resident",
            r"residents of ([a-z\s]+?) only",
        ]

        for pat in state_patterns:
            match = re.search(pat, q)
            if match:
                mentioned_state = match.group(1).strip()
                if len(mentioned_state) > 4 and state not in mentioned_state:
                    return False
                break
    hard_reject_phrases = [
        "music therapy",
        "studio arts",
        "fine arts",
        "art education",
        "medical students",
        "medical student",
        "law students",
        "law student",
        "graduate students only",
        "graduate student only",
        "ph.d",
        "phd",
        "study abroad",
        "bay area high school",
        "new york city undergraduates",
        "nyc undergraduates",
    ]

    for phrase in hard_reject_phrases:
        if phrase in q:
            return False
    return True


def embed_batch(texts):
    cleaned = [str(t)[:8000] for t in texts]

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=cleaned,
    )

    return [np.array(item.embedding, dtype=np.float32) for item in response.data]


def gpt_rerank_match(profile, scholarship_row, candidate_essays):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=700,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": f"""
You are a realistic scholarship matching judge.

Decide whether this scholarship is a good fit for the student and which essay is best to adapt.

Student profile:
{json.dumps(profile, ensure_ascii=False)}

Scholarship:
Purpose: {clean_json_value(scholarship_row.get("Purpose", ""))}
Qualifications: {clean_json_value(scholarship_row.get("Qualifications", ""))}
URL: {clean_json_value(scholarship_row.get("url", ""))}

Candidate essays:
{json.dumps(candidate_essays, ensure_ascii=False)}

Rules:
- If the student is clearly ineligible, set keep to false.
- If the scholarship has a specific identity, state, school level, major, or career requirement not supported by the profile or essay, set keep to false.
- Do not reject just because the fit is not perfect.
- Score 80+ only when the essay directly supports the scholarship purpose.
- Score 60-79 for usable but not perfect matches.
- Score below 60 for generic or weak matches.
- Choose the essay that would require the least rewriting.

Return JSON only:
{{
  "keep": true,
  "score": 0,
  "best_essay_index": 0,
  "match_reason": "specific reason this essay fits",
  "eligibility_concern": "",
  "adaptation_advice": "specific advice for adapting the essay"
}}
"""
            }
        ],
    )

    return json.loads(response.choices[0].message.content)


def run_match(profile, essays_raw):
    filtered = []

    for essay in essays_raw:
        full_text = str(
            essay.get("text") or essay.get("content") or ""
        ).strip()

        prompt, response_text = split_prompt_response(full_text)

        if essay.get("prompt"):
            prompt = str(essay["prompt"]).strip()

        if essay.get("response"):
            response_text = str(essay["response"]).strip()

        if not prompt and response_text:
            prompt = response_text[:300].strip()

        if prompt and response_text and word_count(response_text) >= MIN_ESSAY_WORDS:
            filtered.append({
                "prompt": prompt,
                "response": response_text,
            })

    if not filtered:
        raise ValueError(f"No essays were at least {MIN_ESSAY_WORDS} words long.")

    profile_text = json.dumps(profile, ensure_ascii=False)

    essay_texts = [
        f"""
Student profile:
{profile_text}

Essay prompt:
{e['prompt']}

Essay response:
{e['response']}
""".strip()
        for e in filtered
    ]

    batched = embed_batch(essay_texts)

    essay_embeddings = {
        i: normalize_vector(emb)
        for i, emb in enumerate(batched)
    }

    keys = list(essay_embeddings.keys())
    keep = []

    for i, k in enumerate(keys):
        is_dup = any(
            float(np.dot(essay_embeddings[k], essay_embeddings[keys[j]])) > 0.92
            for j in keep
        )
        if not is_dup:
            keep.append(i)

    deduped_count = len(keys) - len(keep)
    if deduped_count:
        print(f"Removed {deduped_count} near-duplicate essay(s)")

    essay_embeddings = {keys[i]: essay_embeddings[keys[i]] for i in keep}
    filtered = [filtered[keys[i]] for i in keep]

    user_vecs_matrix = np.array(list(essay_embeddings.values()), dtype=np.float32)

    eligible_mask = np.array([
        is_eligible(scholarship_df.iloc[i].get("Qualifications", ""), profile)
        for i in range(len(scholarship_df))
    ], dtype=bool)

    filtered_embeddings = scholarship_embeddings[eligible_mask]
    filtered_df = scholarship_df[eligible_mask].reset_index(drop=True)

    print(f"Eligible scholarships after filter: {eligible_mask.sum()} / {len(scholarship_df)}")

    all_scores = filtered_embeddings @ user_vecs_matrix.T

    best_scores = all_scores.max(axis=1)
    top_n = min(3, all_scores.shape[1])
    top_scores = np.sort(all_scores, axis=1)[:, -top_n:]
    multi_essay_score = top_scores.mean(axis=1)

    scores_array = 0.75 * best_scores + 0.25 * multi_essay_score

    top_indices = np.argsort(scores_array)[::-1][:TOP_K]

    results = []
    fallback_results = []
    rejected_keys = set()

    for rank, idx in enumerate(top_indices):
        row = filtered_df.iloc[idx]
        base_score = float(scores_array[idx])

        essay_scores = all_scores[idx]
        top_essay_indices = essay_scores.argsort()[::-1][:min(GPT_ESSAY_CANDIDATES, len(filtered))]

        candidate_essays = [
            {
                "essay_index": int(i),
                "prompt": clean_json_value(filtered[i]["prompt"]),
                "response": clean_json_value(filtered[i]["response"])[:4000],
                "embedding_score": round(float(essay_scores[i]), 3),
            }
            for i in top_essay_indices
        ]

        default_best_real_idx = int(top_essay_indices[0])
        default_best_essay = filtered[default_best_real_idx]

        base_result = {
            "scholarship_url": str(clean_json_value(row.get("url", "")) or ""),
            "scholarship_purpose": str(clean_json_value(row.get("Purpose", "")) or ""),
            "match_score": round(base_score, 3),
            "best_essay_prompt": str(clean_json_value(default_best_essay.get("prompt", "")) or ""),
            "best_essay_response": str(clean_json_value(default_best_essay.get("response", "")) or ""),
            "match_reason": "",
            "eligibility_concern": "",
            "adaptation_advice": "",
        }

        fallback_results.append(base_result)

        # GPT rerank/advice disabled for cost savings.
        # This keeps the embedding-based match result and avoids chat completion calls.
        results.append(base_result)

    seen_urls = set()
    deduped_results = []

    for result in results:
        key = result.get("scholarship_url") or result.get("scholarship_purpose")
        if key in seen_urls or key in rejected_keys:
            continue
        seen_urls.add(key)
        deduped_results.append(result)

    if len(deduped_results) < MIN_RESULTS_TO_SHOW:
        for result in fallback_results:
            key = result.get("scholarship_url") or result.get("scholarship_purpose")
            if key in seen_urls:
                continue
            seen_urls.add(key)
            deduped_results.append(result)

            if len(deduped_results) >= MIN_RESULTS_TO_SHOW:
                break

    deduped_results = [
        r for r in deduped_results
        if float(r.get("match_score", 0)) >= 0.50
    ]

    deduped_results.sort(key=lambda x: x.get("match_score", 0), reverse=True)

    return deduped_results[:TOP_K]


@app.route("/")
def index():
    return render_template("index.html", google_config={
        "apiKey": GOOGLE_API_KEY,
        "appId": GOOGLE_PROJECT_NUMBER,
        "clientId": GOOGLE_CLIENT_ID,
    })


@app.route("/dashboard")
def dashboard():
    if not session.get("user_id"):
        return redirect(url_for("index"))

    return render_template("dashboard.html", google_config={
        "apiKey": GOOGLE_API_KEY,
        "appId": GOOGLE_PROJECT_NUMBER,
        "clientId": GOOGLE_CLIENT_ID,
    })


@app.route("/auth/login")
def auth_login():
    for key in ("state_val", "major", "citizen", "level"):
        val = request.args.get(key)
        if val is not None:
            session[f"profile_{key}"] = val

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return "Missing Google OAuth env vars.", 500

    flow = build_flow()

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        code_challenge_method="S256",
    )

    session["oauth_state"] = state
    session["code_verifier"] = flow.code_verifier

    return redirect(authorization_url)


@app.route("/auth/callback")
def auth_callback():
    state = session.get("oauth_state")
    code_verifier = session.get("code_verifier")

    if not state:
        return "Missing OAuth state.", 400

    flow = build_flow(state=state)
    flow.fetch_token(
        authorization_response=request.url,
        code_verifier=code_verifier,
    )

    credentials = flow.credentials

    oauth2 = build("oauth2", "v2", credentials=credentials, cache_discovery=False)
    user_info = oauth2.userinfo().get().execute()

    google_sub = user_info.get("id")
    email = user_info.get("email")
    name = user_info.get("name")
    picture = user_info.get("picture")

    if not google_sub or not email:
        return "Could not read Google profile.", 400

    conn = get_db()

    existing = conn.execute(
        "SELECT id FROM users WHERE google_sub = ?",
        (google_sub,),
    ).fetchone()

    if existing:
        user_id = existing["id"]
        conn.execute(
            """
            UPDATE users
            SET email = ?, name = ?, picture = ?
            WHERE id = ?
            """,
            (email, name, picture, user_id),
        )
    else:
        cur = conn.execute(
            """
            INSERT INTO users (google_sub, email, name, picture)
            VALUES (?, ?, ?, ?)
            """,
            (google_sub, email, name, picture),
        )
        user_id = cur.lastrowid

    conn.execute(
        """
        INSERT INTO google_tokens 
            (user_id, access_token, refresh_token, token_expiry)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            access_token = excluded.access_token,
            refresh_token = COALESCE(excluded.refresh_token, google_tokens.refresh_token),
            token_expiry = excluded.token_expiry
        """,
        (
            user_id,
            credentials.token,
            credentials.refresh_token,
            clean_expiry_for_storage(credentials.expiry),
        ),
    )

    conn.commit()
    conn.close()

    session["user_id"] = user_id

    return redirect(url_for("dashboard"))


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/me")
@login_required
def api_me():
    conn = get_db()

    user = conn.execute(
        """
        SELECT id, email, name, picture
        FROM users
        WHERE id = ?
        """,
        (session["user_id"],),
    ).fetchone()

    conn.close()

    if not user:
        session.clear()
        return jsonify({"error": "User not found."}), 401

    return jsonify({
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "picture": user["picture"],
        "profile": {
            "state": session.get("profile_state_val", ""),
            "major": session.get("profile_major", ""),
            "us_citizen": session.get("profile_citizen", "true") == "true",
            "level": session.get("profile_level", "undergraduate"),
        },
    })


@app.route("/api/picker-token", methods=["POST"])
@login_required
def api_picker_token():
    try:
        creds = get_user_credentials(session["user_id"])

        if not GOOGLE_API_KEY:
            return jsonify({"error": "Missing GOOGLE_API_KEY in .env"}), 500

        if not GOOGLE_PROJECT_NUMBER:
            return jsonify({"error": "Missing GOOGLE_PROJECT_NUMBER in .env"}), 500

        return jsonify({
            "accessToken": creds.token,
            "apiKey": GOOGLE_API_KEY,
            "appId": GOOGLE_PROJECT_NUMBER,
            "clientId": GOOGLE_CLIENT_ID,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/import-doc", methods=["POST"])
@login_required
def api_import_doc():
    payload = request.get_json(silent=True) or {}
    file_id = payload.get("fileId", "").strip()

    if not file_id:
        return jsonify({"error": "Missing fileId"}), 400

    try:
        creds = get_user_credentials(session["user_id"])

        drive = build("drive", "v3", credentials=creds, cache_discovery=False)
        docs_service = build("docs", "v1", credentials=creds, cache_discovery=False)

        metadata = drive.files().get(
            fileId=file_id,
            fields="id,name,mimeType",
            supportsAllDrives=True,
        ).execute()

        mime_type = metadata.get("mimeType")

        if mime_type == GOOGLE_FOLDER_MIME:
            files_to_import = list_google_docs_in_folder(drive, file_id)

            if not files_to_import:
                return jsonify({
                    "imported": [],
                    "count": 0,
                    "message": "No Google Docs found in that folder.",
                })

        elif mime_type == GOOGLE_DOC_MIME:
            files_to_import = [metadata]

        else:
            return jsonify({
                "error": "Only Google Docs or folders are supported."
            }), 400

        imported = []
        conn = get_db()

        for file_meta in files_to_import:
            document = docs_service.documents().get(
                documentId=file_meta["id"]
            ).execute()

            parts = []

            for element in document.get("body", {}).get("content", []):
                parts.append(extract_text_from_element(element))

            plain_text = "".join(parts).strip()

            conn.execute(
                """
                INSERT INTO imported_documents
                    (user_id, google_file_id, title, mime_type, content)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, google_file_id) DO UPDATE SET
                    title = excluded.title,
                    mime_type = excluded.mime_type,
                    content = excluded.content,
                    imported_at = CURRENT_TIMESTAMP
                """,
                (
                    session["user_id"],
                    file_meta["id"],
                    file_meta["name"],
                    file_meta["mimeType"],
                    plain_text,
                ),
            )

            imported.append({
                "googleFileId": file_meta["id"],
                "title": file_meta["name"],
            })

        conn.commit()
        conn.close()

        return jsonify({
            "imported": imported,
            "count": len(imported),
        })

    except HttpError as exc:
        return jsonify({"error": f"Google API error: {exc}"}), 400

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/documents")
@login_required
def api_documents():
    conn = get_db()

    rows = conn.execute(
        """
        SELECT id, title, google_file_id, content, imported_at
        FROM imported_documents
        WHERE user_id = ?
        ORDER BY imported_at DESC, id DESC
        """,
        (session["user_id"],),
    ).fetchall()

    conn.close()

    return jsonify([
        {
            "id": row["id"],
            "title": row["title"],
            "googleFileId": row["google_file_id"],
            "content": row["content"],
            "importedAt": row["imported_at"],
        }
        for row in rows
    ])


@app.route("/api/documents/<int:doc_id>", methods=["DELETE"])
@login_required
def api_delete_document(doc_id):
    conn = get_db()

    cur = conn.execute(
        """
        DELETE FROM imported_documents
        WHERE id = ? AND user_id = ?
        """,
        (doc_id, session["user_id"]),
    )

    conn.commit()
    conn.close()

    if cur.rowcount == 0:
        return jsonify({"error": "Document not found"}), 404

    return jsonify({"ok": True})


@app.route("/api/match", methods=["POST"])
@login_required
def api_match():
    try:
        payload = request.get_json(silent=True) or {}
        profile = payload.get("profile", {})

        conn = get_db()

        rows = conn.execute(
            """
            SELECT title, content
            FROM imported_documents
            WHERE user_id = ? AND content != ''
            """,
            (session["user_id"],),
        ).fetchall()

        conn.close()

        if not rows:
            return jsonify({
                "error": "No imported essays found. Please import some documents first."
            }), 400

        essays_raw = [
            {
                "text": row["content"],
                "title": row["title"],
            }
            for row in rows
            if row["content"].strip()
        ]

        results = run_match(profile, essays_raw)

        return jsonify(results)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception as exc:
        print("MATCH ERROR:", repr(exc))
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "scholarships_loaded": len(scholarship_df),
        "embeddings_shape": list(scholarship_embeddings.shape),
        "data_mode": "local_scholarship_embeddings_only",
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)