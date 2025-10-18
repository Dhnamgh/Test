# app.py
import re
import time
from datetime import datetime
import hashlib

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px

# =========================
# PAGE CONFIG + BANNER
# =========================
st.set_page_config(page_title="Hệ thống trắc nghiệm trực tuyến", layout="wide")

def render_banner():
    st.markdown(
        (
            "<div style='padding:10px 16px;border-radius:10px;"
            "background:#1e90ff;color:#ffffff;font-weight:600;"
            "display:flex;align-items:center;gap:10px;"
            "box-shadow:0 2px 5px rgba(0,0,0,0.2);'>"
            "🧪 Hệ thống trắc nghiệm trực tuyến"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

# =========================
# SECRETS HELPERS
# =========================
def sget(key, default=None):
    if key in st.secrets:
        return st.secrets[key]
    if "app" in st.secrets and key in st.secrets["app"]:
        return st.secrets["app"][key]
    return default

def srequire(key):
    val = sget(key)
    if val in (None, ""):
        st.error(f"❌ Thiếu khóa secrets: {key}. Vào Manage app → Settings → Secrets để bổ sung.")
        st.stop()
    return val

QUIZ_ID        = sget("QUIZ_ID", "PSY36")
TIME_LIMIT_MIN = int(sget("TIME_LIMIT_MIN", 20))             # Likert
MCQ_TIME_LIMIT_MIN = int(sget("MCQ_TIME_LIMIT_MIN", TIME_LIMIT_MIN))  # MCQ (mặc định theo Likert)
TEACHER_USER   = str(sget("TEACHER_USER", "teacher")).strip()
TEACHER_PASS   = str(sget("TEACHER_PASS", "teacher123")).strip()

QUESTIONS_SPREADSHEET_ID = srequire("QUESTIONS_SPREADSHEET_ID")
QUESTIONS_SHEET_NAME     = sget("QUESTIONS_SHEET_NAME", "Question")
MCQ_QUESTIONS_SHEET_NAME = sget("MCQ_QUESTIONS_SHEET_NAME", "MCQ_Questions")

RESPONSES_SPREADSHEET_ID = srequire("RESPONSES_SPREADSHEET_ID")

# =========================
# GOOGLE SHEETS HELPERS
# =========================
def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    sa = st.secrets.get("gcp_service_account")
    if not sa or "client_email" not in sa or "private_key" not in sa:
        st.error("❌ Thiếu hoặc sai khối [gcp_service_account] trong Secrets.")
        st.stop()
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    return gspread.authorize(creds)

def diagnose_gsheet_access(spreadsheet_id: str, sheet_name: str):
    sa_email = st.secrets["gcp_service_account"].get("client_email", "(unknown)")
    st.error("Không truy cập được Google Sheet (PermissionError/APIError).")
    st.info(
        "🔧 Cách sửa:\n"
        f"1) Mở file Google Sheet ID: `{spreadsheet_id}`\n"
        f"2) Share cho service account: **{sa_email}** → quyền **Editor**\n"
        f"3) Tên worksheet đúng: **{sheet_name}**\n"
        "4) Save → Rerun/Restart app."
    )

@st.cache_data(ttl=300)
def load_questions_df():
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
    except Exception:
        diagnose_gsheet_access(QUESTIONS_SPREADSHEET_ID, QUESTIONS_SHEET_NAME)
        st.stop()

    try:
        ws = sh.worksheet(QUESTIONS_SHEET_NAME)
    except gspread.WorksheetNotFound:
        st.error(f"Không thấy worksheet tên **{QUESTIONS_SHEET_NAME}**.")
        st.stop()

    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        st.warning("Worksheet câu hỏi trống.")
        return df

    if "q_index" not in df.columns:
        df["q_index"] = range(1, len(df) + 1)
    if "quiz_id" in df.columns:
        df = df[df["quiz_id"].astype(str).str.strip() == str(QUIZ_ID)].copy()

    df = df.sort_values("q_index")
    return df

@st.cache_data(ttl=300)
def load_mcq_questions_df():
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
        ws = sh.worksheet(MCQ_QUESTIONS_SHEET_NAME)
    except Exception:
        st.error(f"Không truy cập được MCQ_Questions (sheet '{MCQ_QUESTIONS_SHEET_NAME}').")
        st.stop()

    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        st.warning("Worksheet MCQ_Questions trống.")
        return df

    if "q_index" not in df.columns:
        df["q_index"] = range(1, len(df) + 1)
    if "quiz_id" in df.columns:
        df = df[df["quiz_id"].astype(str).str.strip() == str(QUIZ_ID)].copy()

    df = df.sort_values("q_index")
    return df

def _col_idx_to_letter(idx_1based: int) -> str:
    n = idx_1based
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

# =========================
# CLASS / ROSTER HELPERS
# =========================
def get_class_rosters():
    # Ưu tiên lấy từ secrets: "D25A,D25B,D25C"
    s = sget("CLASS_ROSTERS", "")
    if s:
        return [x.strip() for x in re.split(r"[,\s]+", s) if x.strip()]
    # Nếu không có, quét từ file RESPONSES: các sheet tên kiểu D25A, D25C...
    try:
        gc = get_gspread_client()
        sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
        titles = [w.title for w in sh.worksheets()]
        cands = [t for t in titles if re.match(r"^D\d+[A-Za-z]+$", t)]
        return cands or ["D25A", "D25C"]
    except Exception:
        return ["D25A", "D25C"]

CLASS_ROSTERS = get_class_rosters()

def open_roster_ws(class_code: str):
    class_code = class_code.strip()
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
        ws = sh.worksheet(class_code)
    except Exception as e:
        st.error(f"Không mở được roster lớp '{class_code}': {e}")
        st.stop()
    return ws

@st.cache_data(ttl=120)
def load_whitelist_students_by_class(class_code: str):
    ws = open_roster_ws(class_code)
    rows = ws.get_all_values()
    if not rows or len(rows) < 2:
        return {}
    header = [h.strip() for h in rows[0]]
    data = rows[1:]

    def find_idx(names):
        for n in names:
            if n in header:
                return header.index(n)
        return None

    idx_mssv = find_idx(["MSSV", "mssv"])
    idx_name = find_idx(["Họ và Tên", "Họ và tên", "Ho va Ten", "Ho va ten"])
    idx_dob  = find_idx(["NTNS", "ntns", "Ngày sinh", "DOB"])
    idx_to   = find_idx(["Tổ", "tổ", "To", "to"])

    if idx_mssv is None or idx_name is None:
        st.error("Roster lớp thiếu cột 'MSSV' hoặc 'Họ và Tên'.")
        st.stop()

    wl = {}
    for r in data:
        if len(r) <= idx_mssv:
            continue
        mssv = r[idx_mssv].strip()
        if not mssv:
            continue
        wl[mssv] = {
            "name": r[idx_name].strip() if len(r) > idx_name else "",
            "dob":  r[idx_dob].strip() if (idx_dob is not None and len(r) > idx_dob) else "",
            "to":   r[idx_to].strip()  if (idx_to is not None and len(r) > idx_to)  else "",
        }
    return wl

def _ensure_header(ws, base_cols, tail_cols):
    header = ws.row_values(1)
    changed = False
    for c in base_cols + tail_cols:
        if c not in header:
            header.append(c); changed = True
    if changed or not header:
        ws.update("A1", [header])
    return header

def open_likert_response_ws_for_class(class_code: str):
    gc = get_gspread_client()
    name = f"Likert{class_code.strip()}"
    try:
        sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
        try:
            ws = sh.worksheet(name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=name, rows=2000, cols=80)
    except Exception as e:
        st.error(f"Không mở được sheet '{name}': {e}")
        st.stop()
    base = ["TT", "MSSV", "Họ và Tên", "NTNS", "Tổ"]
    qcols = [str(i) for i in range(1, 37)]
    tail = ["submitted_at", "quiz_id", "class"]
    _ensure_header(ws, base, qcols + tail)
    return ws

def open_mcq_response_ws_for_class(class_code: str, n_questions: int):
    gc = get_gspread_client()
    name = f"MCQ{class_code.strip()}"
    try:
        sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
        try:
            ws = sh.worksheet(name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=name, rows=2000, cols=200)
    except Exception as e:
        st.error(f"Không mở được sheet '{name}': {e}")
        st.stop()
    base = ["TT", "MSSV", "Họ và Tên", "NTNS", "Tổ"]
    qcols = [str(i) for i in range(1, n_questions + 1)]
    tail = ["score", "submitted_at", "quiz_id", "class"]
    _ensure_header(ws, base, qcols + tail)
    return ws

def attempt_exists(ws, header, mssv: str) -> bool:
    try:
        col_mssv = header.index("MSSV")
    except ValueError:
        return False
    rows = ws.get_all_values()[1:]
    for r in rows:
        if len(r) > col_mssv and r[col_mssv].strip() == mssv.strip():
            if "submitted_at" in header:
                c = header.index("submitted_at")
                if len(r) > c and r[c].strip():
                    return True
            for c_name in header:
                if c_name.isdigit():
                    c_idx = header.index(c_name)
                    if len(r) > c_idx and r[c_idx].strip():
                        return True
    return False

# =========================
# SHUFFLE HELPERS
# =========================
def stable_perm(n: int, key: str) -> list:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    rng_seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(rng_seed)
    arr = np.arange(n)
    rng.shuffle(arr)
    return arr.tolist()

def _option_perm_for_student(mssv: str, qidx: int):
    key = f"MCQ|{QUIZ_ID}|{mssv}|{qidx}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    perm = np.arange(4)
    rng.shuffle(perm)
    return perm.tolist()

# =========================
# STUDENT STATE
# =========================
def init_exam_state():
    st.session_state.setdefault("sv_mssv", "")
    st.session_state.setdefault("sv_hoten", "")
    st.session_state.setdefault("sv_class", "")
    st.session_state.setdefault("sv_allow", False)

    # Likert
    st.session_state.setdefault("likert_started", False)
    st.session_state.setdefault("likert_start_time", None)
    st.session_state.setdefault("sv_order", [])
    st.session_state.setdefault("sv_cursor", 0)
    st.session_state.setdefault("sv_answers", {})

    # MCQ
    st.session_state.setdefault("mcq_started", False)
    st.session_state.setdefault("mcq_start_time", None)
    st.session_state.setdefault("mcq_cursor", 0)
    st.session_state.setdefault("mcq_answers", {})

def student_gate() -> bool:
    init_exam_state()
    if st.session_state.get("sv_allow"):
        return True

    with st.form("sv_login_unified"):
        col0, col1, col2 = st.columns([1,1,2])
        with col0:
            class_code = st.selectbox("Lớp", options=get_class_rosters(), index=0)
        with col1:
            mssv = st.text_input("MSSV", placeholder="VD: 2112345")
        with col2:
            hoten = st.text_input("Họ và Tên", placeholder="VD: Nguyễn Văn A")
        agree = st.checkbox("Tôi xác nhận thông tin trên là đúng.")
        submitted = st.form_submit_button("Đăng nhập")

    if submitted:
        if not mssv or not hoten:
            st.error("Vui lòng nhập MSSV và Họ & Tên.")
            return False
        if not agree:
            st.error("Vui lòng tích xác nhận.")
            return False
        wl = load_whitelist_students_by_class(class_code)
        if mssv.strip() not in wl:
            st.error(f"MSSV không nằm trong lớp {class_code}.")
            return False
        st.session_state["sv_class"] = class_code.strip()
        st.session_state["sv_mssv"] = mssv.strip()
        st.session_state["sv_hoten"] = hoten.strip()
        st.session_state["sv_allow"] = True
        st.success("Đăng nhập thành công.")
        st.rerun()

    st.info("Vui lòng đăng nhập để chọn loại trắc nghiệm.")
    return False

# =========================
# LIKERT EXAM
# =========================
def start_likert_exam(n_questions: int):
    mssv  = st.session_state.get("sv_mssv", "")
    hoten = st.session_state.get("sv_hoten", "")
    st.session_state["likert_started"] = True
    st.session_state["likert_start_time"] = time.time()
    key = f"{QUIZ_ID}|{mssv}|{hoten}"
    st.session_state["sv_order"] = stable_perm(n_questions, key)
    st.session_state["sv_cursor"] = 0
    st.session_state["sv_answers"] = {}

def remaining_seconds_likert():
    if not st.session_state.get("likert_started"):
        return TIME_LIMIT_MIN * 60
    spent = time.time() - (st.session_state.get("likert_start_time") or time.time())
    remain = max(0, int(TIME_LIMIT_MIN * 60 - spent))
    return remain

def render_timer_likert():
    rem = remaining_seconds_likert()
    mins, secs = divmod(rem, 60)
    st.markdown(f"⏳ **Thời gian còn lại:** {mins:02d}:{secs:02d}")

def likert36_exam():
    if not st.session_state.get("sv_allow"):
        st.info("Bạn chưa đăng nhập.")
        return

    df = load_questions_df()
    n_questions = len(df)
    st.success(f"Đề {QUIZ_ID} — {n_questions} câu (Likert 1..5)")

    # Chặn làm lại trước khi cho Start
    class_code = st.session_state.get("sv_class", "")
    ws = open_likert_response_ws_for_class(class_code)
    header = ws.row_values(1)
    if attempt_exists(ws, header, st.session_state.get("sv_mssv","")):
        st.error("Bạn đã nộp bài Likert trước đó. Chỉ được làm 1 lần.")
        return

    # Chưa bấm Start → không lộ đề
    if not st.session_state.get("likert_started"):
        with st.container():
            st.markdown("**Mô tả Likert 36:** Mỗi câu chọn mức 1..5. Có đếm ngược thời gian.")
            st.caption(f"Thời gian làm bài: {TIME_LIMIT_MIN} phút")
            if st.button("▶️ Bắt đầu bài Likert", type="primary"):
                start_likert_exam(n_questions)
                st.rerun()
        return

    # ĐÃ Start → hiển thị đề + timer
    render_timer_likert()
    if remaining_seconds_likert() <= 0:
        st.warning("⏱️ Hết thời gian — hệ thống sẽ nộp bài với các câu đã chọn.")
        do_submit_likert(df)
        return

    order = st.session_state["sv_order"] or list(range(n_questions))
    cur = st.session_state["sv_cursor"]
    cur = max(0, min(cur, n_questions - 1))
    st.session_state["sv_cursor"] = cur

    row = df.iloc[order[cur]]
    qidx = int(row["q_index"])
    qtext = str(row.get("question", f"Câu {qidx}"))

    st.markdown(f"### Câu {cur+1}/{n_questions}")
    st.write(qtext)

    current_val = st.session_state["sv_answers"].get(qidx, None)
    picked = st.radio(
        "Chọn mức độ:",
        options=[1, 2, 3, 4, 5],
        index=[1,2,3,4,5].index(current_val) if current_val in [1,2,3,4,5] else None,
        horizontal=True,
        key=f"radio_{qidx}"
    )
    if picked:
        st.session_state["sv_answers"][qidx] = int(picked)

    st.caption("**Gợi ý:** 1=Hoàn toàn không đồng ý · 2=Không đồng ý · 3=Trung lập · 4=Đồng ý · 5=Hoàn toàn đồng ý")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("⬅️ Câu trước", use_container_width=True, disabled=(cur == 0)):
            st.session_state["sv_cursor"] = max(0, cur - 1)
            st.rerun()
    with c2:
        if st.button("➡️ Câu sau", use_container_width=True, disabled=(cur == n_questions - 1)):
            st.session_state["sv_cursor"] = min(n_questions - 1, cur + 1)
            st.rerun()
    with c3:
        if st.button("📝 Nộp bài Likert", use_container_width=True):
            do_submit_likert(df)

def do_submit_likert(df_questions: pd.DataFrame):
    mssv = st.session_state.get("sv_mssv", "").strip()
    hoten = st.session_state.get("sv_hoten", "").strip()
    class_code = st.session_state.get("sv_class", "").strip()
    answers = st.session_state.get("sv_answers", {})

    if not mssv or not hoten or not class_code:
        st.error("Thiếu thông tin đăng nhập.")
        return

    if "q_index" in df_questions.columns:
        qindices = sorted(df_questions["q_index"].astype(int).tolist())
    else:
        qindices = list(range(1, 37))
    ans_map = {int(q): answers.get(int(q), "") for q in qindices}

    try:
        ws = open_likert_response_ws_for_class(class_code)
        header = ws.row_values(1)

        if attempt_exists(ws, header, mssv):
            st.error("Bạn đã nộp bài Likert trước đó. Chỉ được làm 1 lần.")
            return

        rows = ws.get_all_values()[1:]
        col_mssv = header.index("MSSV")
        found_row = None
        for idx, r in enumerate(rows, start=2):
            if len(r) > col_mssv and r[col_mssv].strip() == mssv.strip():
                found_row = idx
                break
        if not found_row:
            found_row = len(rows) + 2
            for col_name, value in {"MSSV": mssv, "Họ và Tên": hoten, "class": class_code}.items():
                if col_name in header:
                    cidx = header.index(col_name) + 1
                    ws.update_acell(f"{_col_idx_to_letter(cidx)}{found_row}", value)
            info = load_whitelist_students_by_class(class_code).get(mssv, {})
            for col_name, key in {"NTNS": "dob", "Tổ": "to"}.items():
                if col_name in header and key in info and info[key]:
                    cidx = header.index(col_name) + 1
                    ws.update_acell(f"{_col_idx_to_letter(cidx)}{found_row}", info[key])

        now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
        updates = []
        for i in range(1, 37):
            if str(i) in header:
                cidx = header.index(str(i)) + 1
                updates.append({"range": f"{_col_idx_to_letter(cidx)}{found_row}", "values": [[ans_map.get(i, "")]]})
        for col_name, value in {"submitted_at": now_iso, "quiz_id": QUIZ_ID, "class": class_code}.items():
            if col_name in header:
                cidx = header.index(col_name) + 1
                updates.append({"range": f"{_col_idx_to_letter(cidx)}{found_row}", "values": [[value]]})

        if updates:
            ws.batch_update(updates)

    except Exception as e:
        st.error(f"Lỗi ghi Responses Likert: {e}")
        return

    st.success("✅ Đã nộp bài Likert thành công!")
    for k in ["likert_started", "likert_start_time", "sv_answers", "sv_order", "sv_cursor"]:
        st.session_state.pop(k, None)

# =========================
# MCQ EXAM (có Start & Timer)
# =========================
def start_mcq_exam():
    st.session_state["mcq_started"] = True
    st.session_state["mcq_start_time"] = time.time()
    st.session_state["mcq_cursor"] = 0
    st.session_state["mcq_answers"] = {}

def remaining_seconds_mcq():
    if not st.session_state.get("mcq_started"):
        return MCQ_TIME_LIMIT_MIN * 60
    spent = time.time() - (st.session_state.get("mcq_start_time") or time.time())
    return max(0, int(MCQ_TIME_LIMIT_MIN * 60 - spent))

def render_timer_mcq():
    rem = remaining_seconds_mcq()
    mins, secs = divmod(rem, 60)
    st.markdown(f"⏳ **Thời gian còn lại (MCQ):** {mins:02d}:{secs:02d}")

def mcq_exam():
    if not st.session_state.get("sv_allow"):
        st.info("Bạn chưa đăng nhập.")
        return

    df = load_mcq_questions_df()
    if df.empty:
        st.warning("Chưa có câu hỏi MCQ.")
        return

    mssv = st.session_state.get("sv_mssv", "")
    hoten = st.session_state.get("sv_hoten", "")
    class_code = st.session_state.get("sv_class", "").strip()
    n = len(df)

    # Kiểm tra đã nộp chưa
    ws = open_mcq_response_ws_for_class(class_code, n)
    header = ws.row_values(1)
    if attempt_exists(ws, header, mssv):
        st.error("Bạn đã nộp MCQ trước đó. Chỉ được làm 1 lần.")
        return

    st.success(f"Đề MCQ {QUIZ_ID} — {n} câu (4 đáp án).")

    # Chưa ấn Start → mô tả + Start
    if not st.session_state.get("mcq_started"):
        with st.container():
            st.markdown("**Mô tả MCQ:** Mỗi câu chọn A/B/C/D. Có đếm ngược thời gian.")
            st.caption(f"Thời gian làm bài MCQ: {MCQ_TIME_LIMIT_MIN} phút")
            if st.button("▶️ Bắt đầu bài MCQ", type="primary"):
                start_mcq_exam()
                st.rerun()
        return

    # Đang làm bài → timer & hiển thị câu
    render_timer_mcq()
    if remaining_seconds_mcq() <= 0:
        st.warning("⏱️ Hết thời gian — hệ thống sẽ nộp bài với các câu đã chọn.")
        total_correct = 0
        ans = st.session_state["mcq_answers"]
        for _, r in df.iterrows():
            qi = int(r["q_index"])
            if ans.get(qi, "") == str(r["correct"]).strip().upper():
                total_correct += 1
        try:
            upsert_mcq_response(mssv, hoten, ans, total_correct, n)
            st.success(f"✅ Đã nộp MCQ. Điểm: {total_correct}/{n}")
            for k in ["mcq_cursor", "mcq_answers", "mcq_started", "mcq_start_time"]:
                st.session_state.pop(k, None)
        except Exception as e:
            st.error(f"Lỗi ghi MCQ_Responses: {e}")
        return

    order = stable_perm(n, f"MCQ_ORDER|{QUIZ_ID}|{mssv}|{hoten}")
    cur = st.session_state.get("mcq_cursor", 0)
    cur = max(0, min(cur, n-1))
    st.session_state["mcq_cursor"] = cur

    row = df.iloc[order[cur]]
    qidx = int(row["q_index"])
    qtext = str(row["question"])
    options = [str(row["optionA"]), str(row["optionB"]), str(row["optionC"]), str(row["optionD"])]

    st.markdown(f"### Câu {cur+1}/{n}")
    st.write(qtext)

    perm = _option_perm_for_student(mssv, qidx)
    shuffled_opts = [options[i] for i in perm]
    labels = ['A', 'B', 'C', 'D']
    inv = {labels[i]: ['A','B','C','D'][perm[i]] for i in range(4)}

    pick = st.radio(
        "Chọn đáp án:",
        options=[f"{labels[i]}. {shuffled_opts[i]}" for i in range(4)],
        index=None,
        key=f"mcq_{qidx}",
    )
    if pick:
        chosen_label = pick.split('.', 1)[0].strip()
        st.session_state["mcq_answers"][qidx] = inv[chosen_label]

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("⬅️ Câu trước", use_container_width=True, disabled=(cur==0)):
            st.session_state["mcq_cursor"] = max(0, cur-1); st.rerun()
    with c2:
        if st.button("➡️ Câu sau", use_container_width=True, disabled=(cur==n-1)):
            st.session_state["mcq_cursor"] = min(n-1, cur+1); st.rerun()
    with c3:
        if st.button("🧹 Xóa chọn", use_container_width=True):
            if qidx in st.session_state["mcq_answers"]:
                del st.session_state["mcq_answers"][qidx]; st.rerun()
    with c4:
        if st.button("📝 Nộp MCQ", use_container_width=True, type="primary"):
            total_correct = 0
            ans = st.session_state["mcq_answers"]
            for _, r in df.iterrows():
                qi = int(r["q_index"])
                if ans.get(qi, "") == str(r["correct"]).strip().upper():
                    total_correct += 1
            try:
                upsert_mcq_response(mssv, hoten, ans, total_correct, n)
                st.success(f"✅ Đã nộp MCQ. Điểm: {total_correct}/{n}")
                for k in ["mcq_cursor", "mcq_answers", "mcq_started", "mcq_start_time"]:
                    st.session_state.pop(k, None)
            except Exception as e:
                st.error(f"Lỗi ghi MCQ_Responses: {e}")

def upsert_mcq_response(mssv: str, hoten: str, answers: dict, total_correct: int, n_questions: int):
    class_code = st.session_state.get("sv_class", "").strip()
    ws = open_mcq_response_ws_for_class(class_code, n_questions)
    header = ws.row_values(1)

    if attempt_exists(ws, header, mssv):
        st.error("Bạn đã nộp MCQ trước đó. Chỉ được làm 1 lần.")
        return

    rows = ws.get_all_values()[1:]
    col_mssv = header.index("MSSV") if "MSSV" in header else 1
    found_row = None
    for idx, r in enumerate(rows, start=2):
        if len(r) > col_mssv and r[col_mssv].strip() == mssv.strip():
            found_row = idx
            break
    if not found_row:
        found_row = len(rows) + 2
        for col_name, value in {"MSSV": mssv, "Họ và Tên": hoten, "class": class_code}.items():
            if col_name in header:
                cidx = header.index(col_name) + 1
                ws.update_acell(f"{_col_idx_to_letter(cidx)}{found_row}", value)
        info = load_whitelist_students_by_class(class_code).get(mssv, {})
        for col_name, key in {"NTNS": "dob", "Tổ": "to"}.items():
            if col_name in header and key in info and info[key]:
                cidx = header.index(col_name) + 1
                ws.update_acell(f"{_col_idx_to_letter(cidx)}{found_row}", info[key])

    updates = []
    for q in range(1, n_questions + 1):
        if str(q) in header:
            cidx = header.index(str(q)) + 1
            updates.append({"range": f"{_col_idx_to_letter(cidx)}{found_row}", "values": [[answers.get(q, "")]]})
    for col_name, value in {
        "score": total_correct,
        "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "quiz_id": QUIZ_ID,
        "class": class_code
    }.items():
        if col_name in header:
            cidx = header.index(col_name) + 1
            updates.append({"range": f"{_col_idx_to_letter(cidx)}{found_row}", "values": [[value]]})

    if updates:
        ws.batch_update(updates)

# =========================
# TEACHER (GV) PANEL
# =========================
def teacher_login() -> bool:
    st.subheader("Đăng nhập Giảng viên")
    if st.session_state.get("is_teacher", False):
        st.success("Đã đăng nhập.")
        if st.button("🚪 Đăng xuất GV", type="secondary", key="logout_gv"):
            st.session_state["is_teacher"] = False
            st.success("Đã đăng xuất.")
            st.rerun()
        return True

    with st.form("teacher_login_form"):
        u = st.text_input("Tài khoản", value="", placeholder="teacher")
        p = st.text_input("Mật khẩu", value="", placeholder="••••••", type="password")
        ok = st.form_submit_button("Đăng nhập")

    if ok:
        if u.strip() == TEACHER_USER and p.strip() == TEACHER_PASS:
            st.session_state["is_teacher"] = True
            st.success("Đăng nhập thành công.")
            st.rerun()
        else:
            st.error("Sai tài khoản hoặc mật khẩu.")
    return False

def _diagnose_questions():
    st.markdown("#### 🔎 Kiểm tra Question sheet")
    try:
        gc = get_gspread_client()
        sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
        ws_titles = [w.title for w in sh.worksheets()]
        st.success("✅ Mở được file câu hỏi.")
        st.write("Worksheets:", ws_titles)
        if QUESTIONS_SHEET_NAME in ws_titles:
            st.info(f"Worksheet hiện hành: **{QUESTIONS_SHEET_NAME}** ✓")
        else:
            st.error(f"❌ Không thấy worksheet: **{QUESTIONS_SHEET_NAME}**")
    except Exception as e:
        st.error(f"Không mở được file câu hỏi: {e}")

def _view_questions():
    st.markdown("#### 📋 Ngân hàng câu hỏi Likert hiện tại")
    dfq = load_questions_df()
    if dfq.empty:
        st.warning("Worksheet **Question** đang trống.")
    else:
        st.dataframe(dfq, use_container_width=True, height=420)
        st.caption(f"Tổng số câu: **{len(dfq)}**")
    with st.expander("🔎 Chẩn đoán"):
        _diagnose_questions()

def push_questions(df: pd.DataFrame):
    required = {"q_index", "question"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        st.error(f"Thiếu cột bắt buộc: {missing}")
        return

    df = df.copy()
    df["q_index"] = pd.to_numeric(df["q_index"], errors="coerce").astype("Int64")

    if "quiz_id" not in df.columns:
        df["quiz_id"] = QUIZ_ID
    else:
        df["quiz_id"] = df["quiz_id"].fillna(QUIZ_ID)

    columns_order = ["quiz_id", "q_index", "facet", "question", "left_label", "right_label", "reverse"]
    for c in columns_order:
        if c not in df.columns:
            df[c] = ""
    df = df[columns_order].sort_values(["quiz_id", "q_index"], na_position="last")

    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
    except Exception as e:
        st.error(f"Không mở được file câu hỏi: {e}")
        return

    try:
        try:
            ws = sh.worksheet(QUESTIONS_SHEET_NAME)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=QUESTIONS_SHEET_NAME, rows=2000, cols=20)

        ws.append_row(list(df.columns))
        if len(df) > 0:
            ws.append_rows(df.astype(object).values.tolist())

        load_questions_df.clear()
        st.success(f"✅ Đã ghi {len[df]} dòng vào **{QUESTIONS_SHEET_NAME}**.")
    except Exception as e:
        st.error(f"Lỗi ghi dữ liệu lên sheet: {e}")

def _upload_questions():
    st.markdown("#### 📥 Tải câu hỏi Likert (CSV/XLSX)")
    st.info(
        "File nên có cột: quiz_id | q_index | facet | question | left_label | right_label | reverse. "
        "Tối thiểu: q_index, question. Nếu thiếu quiz_id, hệ thống điền mặc định."
    )
    up = st.file_uploader("Chọn file câu hỏi Likert", type=["csv", "xlsx"], key="likert_uploader")

    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df_new = pd.read_csv(up)
            else:
                import openpyxl
                df_new = pd.read_excel(up)
        except Exception as e:
            st.error(f"Không đọc được file: {e}")
            return

        st.write("Xem nhanh dữ liệu tải lên:")
        st.dataframe(df_new.head(12), use_container_width=True)
        if st.button("Ghi lên Question", type="primary", key="write_likert"):
            push_questions(df_new)

    with st.expander("🔎 Chẩn đoán"):
        _diagnose_questions()

def push_mcq_questions(df: pd.DataFrame):
    required = {"q_index", "question", "optionA", "optionB", "optionC", "optionD", "correct"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        st.error(f"Thiếu cột bắt buộc cho MCQ: {missing}")
        return

    df = df.copy()
    df["q_index"] = pd.to_numeric(df["q_index"], errors="coerce").astype("Int64")

    if "quiz_id" not in df.columns:
        df["quiz_id"] = QUIZ_ID
    else:
        df["quiz_id"] = df["quiz_id"].fillna(QUIZ_ID)

    cols = ["quiz_id","q_index","question","optionA","optionB","optionC","optionD","correct"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols].sort_values(["quiz_id","q_index"], na_position="last")

    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
    except Exception as e:
        st.error(f"Không mở được file câu hỏi: {e}")
        return

    try:
        try:
            ws = sh.worksheet(MCQ_QUESTIONS_SHEET_NAME)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=MCQ_QUESTIONS_SHEET_NAME, rows=2000, cols=30)

        ws.append_row(list(df.columns))
        if len(df) > 0:
            ws.append_rows(df.astype(object).values.tolist())

        load_mcq_questions_df.clear()
        st.success(f"✅ Đã ghi {len(df)} dòng vào **{MCQ_QUESTIONS_SHEET_NAME}**.")
    except Exception as e:
        st.error(f"Lỗi ghi dữ liệu MCQ lên sheet: {e}")

def _upload_mcq_questions():
    st.markdown("#### 🧩 Tải câu hỏi MCQ (CSV/XLSX)")
    st.info(
        "File MCQ cần cột: quiz_id | q_index | question | optionA | optionB | optionC | optionD | correct.  \n"
        "- correct: A/B/C/D (có thể để trống nếu chưa chấm ngay)."
    )
    up = st.file_uploader("Chọn file MCQ", type=["csv", "xlsx"], key="mcq_uploader")

    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df_new = pd.read_csv(up)
            else:
                import openpyxl
                df_new = pd.read_excel(up)
        except Exception as e:
            st.error(f"Không đọc được file: {e}")
            return

        st.write("Xem nhanh dữ liệu tải lên:")
        st.dataframe(df_new.head(12), use_container_width=True)
        if st.button("Ghi lên MCQ_Questions", type="primary", key="write_mcq"):
            push_mcq_questions(df_new)

# ---------- TẠO LỚP MỚI ----------
def _create_new_class_tab():
    st.markdown("#### 🏫 Tạo lớp mới")
    st.info("Upload roster lớp theo mẫu cột: **STT | MSSV | Họ và Tên | NTNS | Tổ**. Tên worksheet sẽ là **mã lớp** (ví dụ: D25B).")
    class_name = st.text_input("Tên lớp (worksheet mới)", placeholder="VD: D25B").strip()
    up = st.file_uploader("Chọn file roster (CSV/XLSX)", type=["csv", "xlsx"], key="roster_uploader")

    if st.button("Tạo lớp", type="primary", disabled=(not class_name)):
        if not class_name:
            st.error("Vui lòng nhập tên lớp.")
            return
        # Đọc file (nếu có) hoặc tạo rỗng với header mẫu
        if up is not None:
            try:
                if up.name.lower().endswith(".csv"):
                    df = pd.read_csv(up)
                else:
                    import openpyxl
                    df = pd.read_excel(up)
            except Exception as e:
                st.error(f"Không đọc được file: {e}")
                return
        else:
            df = pd.DataFrame(columns=["STT", "MSSV", "Họ và Tên", "NTNS", "Tổ"])

        # Chuẩn header
        wanted = ["STT", "MSSV", "Họ và Tên", "NTNS", "Tổ"]
        for w in wanted:
            if w not in df.columns:
                df[w] = ""
        df = df[wanted]

        # Ghi worksheet lớp
        try:
            gc = get_gspread_client()
            sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
            try:
                ws = sh.worksheet(class_name)
                ws.clear()
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title=class_name, rows=max(100, len(df)+2), cols=10)
            ws.append_row(wanted)
            if len(df) > 0:
                ws.append_rows(df.astype(object).values.tolist())

            # Cập nhật cache class rosters
            load_whitelist_students_by_class.clear()
            st.success(f"✅ Đã tạo/ghi roster lớp **{class_name}**.")
        except Exception as e:
            st.error(f"Lỗi tạo lớp: {e}")

# ---------- THỐNG KÊ MCQ ----------
def _read_mcq_sheet(class_code: str) -> pd.DataFrame:
    """Đọc MCQ<class> → DataFrame."""
    gc = get_gspread_client()
    sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
    wsname = f"MCQ{class_code}"
    try:
        ws = sh.worksheet(wsname)
    except gspread.WorksheetNotFound:
        st.warning(f"Chưa có sheet {wsname}.")
        return pd.DataFrame()
    df = pd.DataFrame(ws.get_all_records())
    return df

def _mcq_stats_tab():
    st.markdown("#### 📊 Thống kê MCQ theo câu hỏi")
    classes = get_class_rosters()
    class_code = st.selectbox("Chọn lớp", options=classes)
    df = _read_mcq_sheet(class_code)
    if df.empty:
        st.info("Chưa có dữ liệu MCQ cho lớp này.")
        return

    # Xác định số câu từ header (cột số)
    qcols = [c for c in df.columns if str(c).isdigit()]
    if not qcols:
        st.info("Không tìm thấy cột câu hỏi (1..N).")
        return

    # Chọn câu
    qnums = sorted([int(c) for c in qcols])
    q_choice = st.selectbox("Chọn câu", options=qnums, index=0)
    col = str(q_choice)

    # Thống kê A/B/C/D
    counts = df[col].astype(str).str.strip().str.upper().value_counts()
    all_labels = ["A","B","C","D"]
    data = []
    total = int(counts.sum())
    for label in all_labels:
        c = int(counts.get(label, 0))
        pct = (c/total*100) if total>0 else 0.0
        data.append({"Đáp án": label, "Số người": c, "Tỷ lệ (%)": round(pct, 2)})

    dstat = pd.DataFrame(data)
    st.dataframe(dstat, use_container_width=True, height=200)

        # Biểu đồ cột tương tác (Plotly nếu có, nếu không dùng Altair)
    if HAS_PLOTLY:
        fig = px.bar(
            dstat,
            x="Đáp án",
            y="Số người",
            color="Đáp án",
            hover_data={"Tỷ lệ (%)": True, "Số người": True, "Đáp án": False},
            text="Số người",
        )
        fig.update_traces(hovertemplate="Số người: %{y}<br>Tỷ lệ: %{customdata[0]}%")
        fig.update_layout(yaxis_title="Số người", xaxis_title="Đáp án", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = (
            alt.Chart(dstat)
            .mark_bar()
            .encode(
                x=alt.X("Đáp án:N", title="Đáp án"),
                y=alt.Y("Số người:Q", title="Số người"),
                color="Đáp án:N",
                tooltip=[alt.Tooltip("Đáp án:N"), alt.Tooltip("Số người:Q"), alt.Tooltip("Tỷ lệ (%):Q")],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    st.plotly_chart(fig, use_container_width=True)

# ---------- TRỢ LÝ AI (OFFLINE) ----------
def _ai_assistant_tab():
    st.markdown("#### 🤖 Trợ lý AI (từ khóa ngắn)")
    classes = get_class_rosters()
    class_code = st.selectbox("Chọn lớp", options=classes, key="ai_class")
    df = _read_mcq_sheet(class_code)
    if df.empty:
        st.info("Chưa có dữ liệu MCQ cho lớp này.")
        return

    # Yêu cầu tối thiểu: cần có cột 'score' và 'submitted_at'
    if "score" not in df.columns:
        st.warning("Sheet MCQ chưa có cột 'score'.")
    if "submitted_at" not in df.columns:
        st.warning("Sheet MCQ chưa có cột 'submitted_at'.")

    q = st.text_input("Nhập từ khóa (ví dụ: 'sớm nhất', 'muộn nhất', 'cao điểm', 'thấp điểm')", placeholder="sớm nhất")
    if st.button("Hỏi"):
        st.write(_ai_answer_from_df(df, q))

def _parse_ts(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.NaT

def _ai_answer_from_df(df: pd.DataFrame, query: str) -> str:
    if df.empty:
        return "Không có dữ liệu."
    q = (query or "").strip().lower()

    # Chuẩn hóa các cột cần thiết
    dfc = df.copy()
    if "score" in dfc.columns:
        dfc["score_num"] = pd.to_numeric(dfc["score"], errors="coerce")
    else:
        dfc["score_num"] = np.nan

    if "submitted_at" in dfc.columns:
        dfc["ts"] = dfc["submitted_at"].apply(_parse_ts)
    else:
        dfc["ts"] = pd.NaT

    # Ưu tiên nhận dạng theo từ khóa
    if any(k in q for k in ["sớm", "som", "sớm nhất"]):
        # người nộp sớm nhất (nhỏ nhất ts)
        dfv = dfc.dropna(subset=["ts"]).sort_values("ts")
        if len(dfv):
            r = dfv.iloc[0]
            return f"Sớm nhất: {r.get('Họ và Tên','') or r.get('MSSV','(không rõ)')} — {r.get('ts')}"
        return "Không có timestamp nộp bài."

    if any(k in q for k in ["muộn", "muon", "trễ", "tre", "muộn nhất"]):
        dfv = dfc.dropna(subset=["ts"]).sort_values("ts")
        if len(dfv):
            r = dfv.iloc[-1]
            return f"Muộn nhất: {r.get('Họ và Tên','') or r.get('MSSV','(không rõ)')} — {r.get('ts')}"
        return "Không có timestamp nộp bài."

    if any(k in q for k in ["cao điểm", "cao", "max", "highest"]):
        dfv = dfc.dropna(subset=["score_num"]).sort_values("score_num")
        if len(dfv):
            r = dfv.iloc[-1]
            return f"Cao điểm nhất: {r.get('Họ và Tên','') or r.get('MSSV','(không rõ)')} — {int(r['score_num'])}"
        return "Không có cột điểm hoặc chưa có điểm."

    if any(k in q for k in ["thấp điểm", "thấp", "min", "lowest"]):
        dfv = dfc.dropna(subset=["score_num"]).sort_values("score_num")
        if len(dfv):
            r = dfv.iloc[0]
            return f"Thấp điểm nhất: {r.get('Họ và Tên','') or r.get('MSSV','(không rõ)')} — {int(r['score_num'])}"
        return "Không có cột điểm hoặc chưa có điểm."

    # Mặc định: trả về gợi ý
    return "Từ khóa gợi ý: 'sớm nhất', 'muộn nhất', 'cao điểm', 'thấp điểm'."

def _diagnose_responses():
    st.markdown("#### ℹ️ Ghi chú Responses")
    st.info(
        "Kết quả được ghi theo từng lớp:\n"
        "- Likert: Likert<CLASS> (VD: LikertD25A, LikertD25C)\n"
        "- MCQ: MCQ<CLASS> (VD: MCQD25A, MCQD25C)\n"
        "Danh sách lớp gốc (whitelist): D25A, D25B, D25C... (cột: STT | MSSV | Họ và Tên | NTNS | Tổ)."
    )

def _view_responses():
    _diagnose_responses()

def teacher_panel():
    render_banner()
    
    if not teacher_login():
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Xem câu hỏi Likert",
        "📥 Tải câu hỏi Likert",
        "🧩 Tải câu hỏi MCQ",
        "🏫 Tạo lớp mới",
        "📊 Thống kê MCQ",
        "🤖 Trợ lý AI",
    ])
    with tab1:
        _view_questions()
    with tab2:
        _upload_questions()
    with tab3:
        _upload_mcq_questions()
    with tab4:
        _create_new_class_tab()
    with tab5:
        _mcq_stats_tab()
    with tab6:
        _ai_assistant_tab()

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.header("Chức năng")
page = st.sidebar.radio("Đi đến", ["Sinh viên", "Giảng viên", "Hướng dẫn"], index=0)

if page == "Sinh viên":
    render_banner()
    st.title("Sinh viên làm bài")

    # Đăng xuất SV
    if st.session_state.get("sv_allow") or st.session_state.get("likert_started") or st.session_state.get("mcq_started"):
        if st.button("🚪 Đăng xuất", type="secondary"):
            for k in list(st.session_state.keys()):
                if k.startswith("sv_") or k.startswith("mcq_") or k.startswith("likert_"):
                    st.session_state.pop(k, None)
            st.success("Đã đăng xuất.")
            st.stop()

    # Cổng đăng nhập SV
    if not student_gate():
        st.stop()

    # Chọn test
    mode = st.radio("Chọn loại trắc nghiệm:", ["Likert 36", "MCQ 4 đáp án"], horizontal=True)
    if mode == "Likert 36":
        likert36_exam()
    else:
        mcq_exam()

elif page == "Giảng viên":
    teacher_panel()
else:
    render_banner()
    st.title("Hướng dẫn nhanh")
    st.markdown(
        "- **Sinh viên:** đăng nhập (Lớp + MSSV + Họ & Tên) → chọn **Likert 36** hoặc **MCQ 4 đáp án** → bấm **Bắt đầu** mới hiển thị đề & **bắt giờ** → **Nộp bài**.  \n"
        "- **Giảng viên:** xem/tải ngân hàng Likert & MCQ; tạo lớp mới; xem **thống kê MCQ** (biểu đồ cột tương tác); dùng **Trợ lý AI** để hỏi nhanh về sớm/muộn, cao/ thấp điểm.  \n"
        "- **Google Sheets:**\n"
        "  - `Question`: ngân hàng Likert (`quiz_id | q_index | facet | question | left_label | right_label | reverse`)\n"
        "  - `MCQ_Questions`: ngân hàng MCQ (`quiz_id | q_index | question | optionA..D | correct`)\n"
        "  - `D25A`, `D25B`, `D25C`...: roster gốc (`STT | MSSV | Họ và Tên | NTNS | Tổ`)\n"
        "  - `Likert<CLASS>`, `MCQ<CLASS>`: kết quả theo lớp.\n"
        "- Nếu lỗi quyền, hãy **Share** file cho service account trong secrets, quyền **Editor**."
    )

st.markdown("---")
st.markdown("© Bản quyền thuộc về TS. Đào Hồng Nam - Đại học Y Dược Thành phố Hồ Chí Minh.")
