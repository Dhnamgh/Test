# app.py
import time
from datetime import datetime
import hashlib

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Hệ thống trắc nghiệm trực tuyến", layout="wide")

def render_banner():
    st.markdown(
        (
            "<div style='padding:10px 16px;border-radius:10px;"
            "background:#0f172a;color:#1e90ff;font-weight:600;"
            "display:flex;align-items:center;gap:10px'>"
            "Hệ thống trắc nghiệm trực tuyến"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

# =========================
# SECRETS HELPERS
# =========================
def sget(key, default=None):
    """Get secret by priority: top-level → [app] → default."""
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

# App config
QUIZ_ID        = sget("QUIZ_ID", "PSY36")
TIME_LIMIT_MIN = int(sget("TIME_LIMIT_MIN", 20))
TEACHER_USER   = str(sget("TEACHER_USER", "teacher")).strip()
TEACHER_PASS   = str(sget("TEACHER_PASS", "teacher123")).strip()

# Google Sheets config
QUESTIONS_SPREADSHEET_ID = srequire("QUESTIONS_SPREADSHEET_ID")
QUESTIONS_SHEET_NAME     = sget("QUESTIONS_SHEET_NAME", "Question")

RESPONSES_SPREADSHEET_ID = srequire("RESPONSES_SPREADSHEET_ID")

# =========================
# GOOGLE SHEETS HELPERS
# =========================
def get_gspread_client():
    """Authorize gspread with Sheets + Drive scopes."""
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
    """Show a friendly how-to when permission or ID is wrong."""
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
    """Đọc ngân hàng câu hỏi Likert từ worksheet Question, lọc theo QUIZ_ID nếu có."""
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

def _col_idx_to_letter(idx_1based: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA ..."""
    n = idx_1based
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

# =========================
# CLASS / ROSTER HELPERS (đa lớp, whitelist từ roster gốc)
# =========================
CLASS_ROSTERS = ["D25A", "D25C"]  # mở rộng khi cần
MCQ_QUESTIONS_SHEET_NAME = sget("MCQ_QUESTIONS_SHEET_NAME", "MCQ_Questions")

def open_roster_ws(class_code: str):
    """Mở sheet danh sách gốc (whitelist) theo lớp, ví dụ 'D25A' hoặc 'D25C'."""
    class_code = class_code.strip()
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
        ws = sh.worksheet(class_code)  # roster gốc
    except Exception as e:
        st.error(f"Không mở được roster lớp '{class_code}': {e}")
        st.stop()
    return ws

@st.cache_data(ttl=120)
def load_whitelist_students_by_class(class_code: str):
    """
    Đọc whitelist từ sheet lớp gốc (D25A/D25C). Yêu cầu cột:
    STT | MSSV | Họ và tên | NTNS | tổ
    Trả về dict {MSSV: {'name':..., 'dob':..., 'to':...}}
    """
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
    idx_name = find_idx(["Họ và tên", "Ho va ten", "Họ và Tên", "Ho va Ten"])
    idx_dob  = find_idx(["NTNS", "ntns", "Ngày sinh", "DOB"])
    idx_to   = find_idx(["tổ", "To", "to"])

    if idx_mssv is None or idx_name is None:
        st.error("Roster lớp thiếu cột 'MSSV' hoặc 'Họ và tên'.")
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
    """Đảm bảo header có đủ các cột trong base_cols + tail_cols theo thứ tự, không xóa dữ liệu cũ."""
    header = ws.row_values(1)
    changed = False
    for c in base_cols + tail_cols:
        if c not in header:
            header.append(c); changed = True
    if changed or not header:
        ws.update("A1", [header])
    return header

def open_likert_response_ws_for_class(class_code: str):
    """Mở/tạo sheet kết quả Likert theo lớp: 'LikertD25A'..."""
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
    base = ["TT", "MSSV", "Họ và Tên", "NTNS", "tổ"]
    qcols = [str(i) for i in range(1, 37)]
    tail = ["submitted_at", "quiz_id", "class"]
    _ensure_header(ws, base, qcols + tail)
    return ws

def open_mcq_response_ws_for_class(class_code: str, n_questions: int):
    """Mở/tạo sheet kết quả MCQ theo lớp: 'MCQD25A'... (tự thêm 1..N)."""
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
    base = ["TT", "MSSV", "Họ và Tên", "NTNS", "tổ"]
    qcols = [str(i) for i in range(1, n_questions + 1)]
    tail = ["score", "submitted_at", "quiz_id", "class"]
    _ensure_header(ws, base, qcols + tail)
    return ws

def attempt_exists(ws, header, mssv: str) -> bool:
    """Kiểm tra SV đã nộp (đã có submitted_at hoặc có ít nhất 1 câu trả lời)."""
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
# LOAD QUESTION BANKS
# =========================
@st.cache_data(ttl=300)
def load_mcq_questions_df():
    """Đọc ngân hàng MCQ từ worksheet MCQ_Questions."""
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

# =========================
# SHUFFLE STABLE PER STUDENT
# =========================
def stable_perm(n: int, key: str) -> list:
    """Sinh hoán vị cố định cho mỗi SV dựa trên hash(MSSV + Họ tên + QUIZ_ID)."""
    h = hashlib.sha256(key.encode("utf-8")).digest()
    rng_seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(rng_seed)
    arr = np.arange(n)
    rng.shuffle(arr)
    return arr.tolist()

def _option_perm_for_student(mssv: str, qidx: int):
    """Sinh hoán vị 4 đáp án theo từng SV cho mỗi q_index."""
    key = f"MCQ|{QUIZ_ID}|{mssv}|{qidx}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    perm = np.arange(4)
    rng.shuffle(perm)
    return perm.tolist()

# =========================
# STUDENT EXAM — COMMON STATE
# =========================
def init_exam_state():
    st.session_state.setdefault("sv_mssv", "")
    st.session_state.setdefault("sv_hoten", "")
    st.session_state.setdefault("sv_class", "")
    st.session_state.setdefault("sv_started", False)
    st.session_state.setdefault("sv_start_time", None)
    st.session_state.setdefault("sv_answers", {})      # {q_index -> 1..5}
    st.session_state.setdefault("sv_order", [])        # hoán vị câu hỏi
    st.session_state.setdefault("sv_cursor", 0)        # index đang hiển thị
    st.session_state.setdefault("sv_allow", False)     # đã đậu whitelist chưa
    st.session_state.setdefault("mcq_cursor", 0)
    st.session_state.setdefault("mcq_answers", {})     # {q_index: 'A'..'D'}

def student_gate() -> bool:
    """
    Cổng đăng nhập SV dùng chung cho cả Likert và MCQ.
    Trả về True nếu đã pass whitelist & set state (sv_allow=True).
    """
    init_exam_state()

    if st.session_state.get("sv_allow"):
        return True

    with st.form("sv_login_unified"):
        col0, col1, col2 = st.columns([1,1,2])
        with col0:
            class_code = st.selectbox("Lớp", options=CLASS_ROSTERS, index=0)
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

        # check whitelist lớp
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
# STUDENT EXAM — LIKERT
# =========================
def start_exam(mssv, hoten, n_questions):
    st.session_state["sv_mssv"] = mssv.strip()
    st.session_state["sv_hoten"] = hoten.strip()
    st.session_state["sv_started"] = True
    st.session_state["sv_start_time"] = time.time()
    key = f"{QUIZ_ID}|{mssv}|{hoten}"
    st.session_state["sv_order"] = stable_perm(n_questions, key)
    st.session_state["sv_cursor"] = 0
    st.session_state["sv_answers"] = {}

def remaining_seconds():
    if not st.session_state.get("sv_started"):
        return TIME_LIMIT_MIN * 60
    spent = time.time() - st.session_state["sv_start_time"]
    remain = max(0, int(TIME_LIMIT_MIN * 60 - spent))
    return remain

def render_timer():
    rem = remaining_seconds()
    mins = rem // 60
    secs = rem % 60
    st.markdown(f"⏳ **Thời gian còn lại:** {mins:02d}:{secs:02d}")

def likert36_exam():
    if not st.session_state.get("sv_allow"):
        st.info("Bạn chưa đăng nhập.")
        return

    df = load_questions_df()
    n_questions = len(df)
    st.success(f"Đề {QUIZ_ID} — {n_questions} câu (Likert 1..5)")

    # Nếu chưa start -> khởi tạo
    if not st.session_state.get("sv_started"):
        start_exam(st.session_state["sv_mssv"], st.session_state["sv_hoten"], n_questions)

    # Chặn làm lại trước khi làm (nếu đã có bài ở sheet lớp)
    class_code = st.session_state.get("sv_class", "")
    ws = open_likert_response_ws_for_class(class_code)
    header = ws.row_values(1)
    if attempt_exists(ws, header, st.session_state["sv_mssv"]):
        st.error("Bạn đã nộp bài Likert trước đó. Chỉ được làm 1 lần.")
        return

    # Đang làm bài
    render_timer()
    if remaining_seconds() <= 0:
        st.warning("⏱️ Hết thời gian — hệ thống sẽ nộp bài với các câu đã chọn.")
        do_submit(df)
        return

    order = st.session_state["sv_order"] or list(range(n_questions))
    cur = st.session_state["sv_cursor"]
    cur = max(0, min(cur, n_questions - 1))
    st.session_state["sv_cursor"] = cur

    row = df.iloc[order[cur]]
    qidx = int(row["q_index"])
    qtext = str(row.get("question", f"Câu {qidx}"))
    left_label = str(row.get("left_label", "Hoàn toàn không đồng ý"))
    right_label = str(row.get("right_label", "Hoàn toàn đồng ý"))

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

    help_text = (
        "**Gợi ý mức độ:**\n"
        "- 1 = Hoàn toàn không đồng ý  \n"
        "- 2 = Không đồng ý  \n"
        "- 3 = Phân vân / Trung lập  \n"
        "- 4 = Đồng ý  \n"
        "- 5 = Hoàn toàn đồng ý"
    )
    st.markdown(help_text)

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
            do_submit(df)

def do_submit(df_questions: pd.DataFrame):
    """Nộp bài Likert: ghi lên sheet Likert<CLASS> và chặn làm lại."""
    mssv = st.session_state.get("sv_mssv", "").strip()
    hoten = st.session_state.get("sv_hoten", "").strip()
    class_code = st.session_state.get("sv_class", "").strip()
    answers = st.session_state.get("sv_answers", {})

    if not mssv or not hoten or not class_code:
        st.error("Thiếu thông tin đăng nhập.")
        return

    # map 1..36 theo q_index của đề
    ans_map = {}
    if "q_index" in df_questions.columns:
        qindices = sorted(df_questions["q_index"].astype(int).tolist())
    else:
        qindices = list(range(1, 37))
    for q in qindices:
        ans_map[int(q)] = answers.get(int(q), "")

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
            # MSSV, họ tên, class
            for col_name, value in {"MSSV": mssv, "Họ và Tên": hoten, "class": class_code}.items():
                if col_name in header:
                    cidx = header.index(col_name) + 1
                    ws.update_acell(f"{_col_idx_to_letter(cidx)}{found_row}", value)
            # NTNS/tổ từ roster
            info = load_whitelist_students_by_class(class_code).get(mssv, {})
            for col_name, key in {"NTNS": "dob", "tổ": "to"}.items():
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
    for k in ["sv_started", "sv_start_time", "sv_answers", "sv_order", "sv_cursor"]:
        st.session_state.pop(k, None)

# =========================
# STUDENT EXAM — MCQ
# =========================
def mcq_exam():
    """Bài thi MCQ 4 lựa chọn, trộn thứ tự câu & đáp án theo từng SV, ghi sheet theo lớp."""
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

    # Trước khi cho làm, kiểm tra đã nộp MCQ lớp này chưa
    ws = open_mcq_response_ws_for_class(class_code, n)
    header = ws.row_values(1)
    if attempt_exists(ws, header, mssv):
        st.error("Bạn đã nộp MCQ trước đó. Chỉ được làm 1 lần.")
        return

    # Sinh trật tự câu theo SV
    order = stable_perm(n, f"MCQ_ORDER|{QUIZ_ID}|{mssv}|{hoten}")
    st.success(f"Đề MCQ {QUIZ_ID} — {n} câu (4 đáp án).")

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
    # map nhãn hiển thị -> nhãn gốc
    inv = {labels[i]: ['A','B','C','D'][perm[i]] for i in range(4)}

    current = st.session_state["mcq_answers"].get(qidx, None)
    # Không pre-select để tránh phức tạp; SV có thể xem lại được.
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
                for k in ["mcq_cursor", "mcq_answers"]:
                    st.session_state.pop(k, None)
            except Exception as e:
                st.error(f"Lỗi ghi MCQ_Responses: {e}")

def upsert_mcq_response(mssv: str, hoten: str, answers: dict, total_correct: int, n_questions: int):
    """
    Ghi MCQ vào sheet 'MCQ<CLASS>' và chặn làm lại.
    """
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
        for col_name, key in {"NTNS": "dob", "tổ": "to"}.items():
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
# TEACHER PANEL
# =========================
def teacher_login() -> bool:
    st.subheader("Đăng nhập Giảng viên")

    if st.session_state.get("is_teacher", False):
        st.success("Đã đăng nhập.")
        # Nút đăng xuất GV (luôn hiện khi đã đăng nhập)
        if st.button("🚪 Đăng xuất GV", type="secondary"):
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
    """
    Ghi ĐÈ toàn bộ worksheet câu hỏi Likert bằng dataframe cung cấp.
    Cần cột tối thiểu: q_index, question.
    """
    required = {"q_index", "question"}
    if not required.issubset(set(df.columns)):
        missing = ", ".join(sorted(required - set(df.columns)))
        st.error(f"Thiếu cột bắt buộc: {missing}")
        return

    if "q_index" in df.columns:
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
        st.success(f"✅ Đã ghi {len(df)} dòng vào **{QUESTIONS_SHEET_NAME}**.")
    except Exception as e:
        st.error(f"Lỗi ghi dữ liệu lên sheet: {e}")

def _upload_questions():
    st.markdown("#### 📥 Tải câu hỏi Likert (CSV/XLSX)")
    st.info(
        "File nên có cột: quiz_id | q_index | facet | question | left_label | right_label | reverse. "
        "Tối thiểu bắt buộc: q_index, question. Nếu thiếu quiz_id, hệ thống sẽ điền mặc định."
    )
    up = st.file_uploader("Chọn file câu hỏi", type=["csv", "xlsx"])

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
        st.dataframe(df_new.head(10), use_container_width=True)
        if st.button("Ghi lên Question", type="primary"):
            push_questions(df_new)

    with st.expander("🔎 Chẩn đoán"):
        _diagnose_questions()

def _diagnose_responses():
    st.markdown("#### ℹ️ Ghi chú Responses")
    st.info(
        "Kết quả được ghi theo từng lớp:\n"
        "- Likert: LikertD25A, LikertD25C\n"
        "- MCQ: MCQD25A, MCQD25C\n"
        "Danh sách lớp gốc (whitelist): D25A, D25C."
    )

def _view_responses():
    _diagnose_responses()

def teacher_panel():
    """UI chính của tab Giảng viên."""
    
    # Nút đăng xuất GV nếu đã đăng nhập
    if st.session_state.get("is_teacher", False):
        if st.button("🚪 Đăng xuất GV", type="secondary"):
            st.session_state["is_teacher"] = False
            st.success("Đã đăng xuất.")
            st.rerun()

    if not teacher_login():
        return

    tab1, tab2, tab3 = st.tabs(["📋 Xem câu hỏi Likert", "📥 Tải câu hỏi Likert", "📑 Ghi chú Responses"])
    with tab1:
        _view_questions()
    with tab2:
        _upload_questions()
    with tab3:
        _view_responses()

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.header("Chức năng")
page = st.sidebar.radio("Đi đến", ["Sinh viên", "Giảng viên", "Hướng dẫn"], index=0)

if page == "Sinh viên":
    render_banner()
    st.title("Sinh viên làm bài")

    # Nút Đăng xuất SV
    if st.session_state.get("sv_allow") or st.session_state.get("sv_started") or st.session_state.get("mcq_answers"):
        if st.button("🚪 Đăng xuất", type="secondary"):
            for k in list(st.session_state.keys()):
                if k.startswith("sv_") or k.startswith("mcq_"):
                    st.session_state.pop(k, None)
            st.success("Đã đăng xuất.")
            st.stop()

    # Cổng đăng nhập dùng chung
    if not student_gate():
        st.stop()

    # Đăng nhập OK → chọn mode
    mode = st.radio("Chọn loại trắc nghiệm:", ["Likert 36", "MCQ 4 đáp án"], horizontal=True)

    if mode == "Likert 36":
        likert36_exam()
    else:
        mcq_exam()

elif page == "Giảng viên":
    render_banner()
  
    teacher_panel()

else:
    render_banner()
    st.title("Hướng dẫn nhanh")
    st.markdown(
        "- **Sinh viên:** đăng nhập (Lớp + MSSV + Họ & Tên) → chọn **Likert 36** hoặc **MCQ 4 đáp án** → làm bài → **Nộp bài**.  \n"
        "  Mỗi loại bài chỉ **làm 1 lần**. Likert có **đồng hồ đếm ngược** theo `TIME_LIMIT_MIN`.\n"
        "- **Giảng viên:** xem/tải ngân hàng Likert; kết quả được ghi theo lớp: `Likert<CLASS>`, `MCQ<CLASS>`.  \n"
        "- **Google Sheets:**\n"
        "  - `Question`: ngân hàng Likert (`quiz_id | q_index | facet | question | left_label | right_label | reverse`)\n"
        "  - `D25A`, `D25C`: roster gốc (`STT | MSSV | Họ và tên | NTNS | tổ`)\n"
        "  - `LikertD25A`, `LikertD25C`, `MCQD25A`, `MCQD25C`: kết quả theo lớp.\n"
        "- Nếu lỗi quyền, hãy **Share** file cho service account trong secrets, quyền **Editor**."
    )

st.markdown("---")
st.markdown("© Bản quyền thuộc về TS. Đào Hồng Nam - Đại học Y Dược Thành phố Hồ Chí Minh.")
