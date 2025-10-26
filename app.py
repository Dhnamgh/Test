# app.py
# =========================
# IMPORTS & PAGE CONFIG
# =========================
import re, time, hashlib, unicodedata
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# Plotly (ưu tiên) → nếu thiếu dùng Altair
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    import altair as alt

st.set_page_config(page_title="Hệ thống trắc nghiệm trực tuyến", layout="wide")

# =========================
# UTILS: Secrets helpers
# =========================
def sget(key, default=None):
    """Đọc từ root secrets → nếu không có thì đọc [app]."""
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

def _normalize_credential(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    for z in ["\u200b", "\u200c", "\u200d", "\u2060"]:
        s = s.replace(z, "")
    s = s.replace("\xa0", " ")
    return s.strip()

def _get_student_password() -> str:
    """Tìm mật khẩu SV ở nhiều khóa/section; trả về chuỗi đã trim (có thể rỗng)."""
    import streamlit as st
    def norm(v): return "" if v is None else str(v).strip()

    # Khóa phẳng (root)
    for k in ["STUDENT_PASSWORD", "Student_password", "Student_pasword"]:
        if k in st.secrets:
            v = norm(st.secrets[k])
            if v: return v

    # Bên trong section phổ biến
    for sec in ["student", "app", "auth", "passwords"]:
        if sec in st.secrets:
            d = st.secrets[sec]
            for k in ["STUDENT_PASSWORD", "Student_password", "Student_pasword"]:
                if k in d:
                    v = norm(d[k])
                    if v: return v
    return ""

# =========================
# CHUẨN HÓA HỌ TÊN
# =========================
def normalize_vietnamese_name(name: str) -> str:
    """
    Chuẩn hóa họ tên tiếng Việt:
      - Bỏ khoảng trắng thừa
      - Viết hoa chữ cái đầu mỗi từ
      - Giữ nguyên dấu tiếng Việt
      - Không phân biệt chữ hoa/thường khi nhập
    """
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    parts = re.split(r"\s+", name)
    return " ".join(p.capitalize() for p in parts if p)

# =========================
# CẤU HÌNH TỪ SECRETS
# =========================
TIME_LIMIT_MIN      = int(sget("TIME_LIMIT_MIN", 20))      # Likert
MCQ_TIME_LIMIT_MIN  = int(sget("MCQ_TIME_LIMIT_MIN", 20))  # MCQ
QUIZ_ID             = sget("QUIZ_ID", "PSY36")

QUESTIONS_SPREADSHEET_ID = srequire("QUESTIONS_SPREADSHEET_ID")
RESPONSES_SPREADSHEET_ID = srequire("RESPONSES_SPREADSHEET_ID")
QUESTIONS_SHEET_NAME     = sget("QUESTIONS_SHEET_NAME", "Question")
MCQ_QUESTIONS_SHEET_NAME = sget("MCQ_QUESTIONS_SHEET_NAME", "MCQ_Questions")

# =========================
# BANNER
# =========================
def render_banner():
    st.markdown(
        (
            "<div style='padding:10px 16px;border-radius:10px;"
            "background:#1e90ff;color:#ffffff;font-weight:600;"
            "display:flex;align-items:center;gap:10px;"
            "box-shadow:0 2px 5px rgba(0,0,0,0.2);'>"
            "Hệ thống trắc nghiệm trực tuyến"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

# =========================
# GOOGLE SHEETS HELPERS
# =========================
@st.cache_resource
def get_gspread_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    sa = st.secrets.get("gcp_service_account")
    if not sa or "client_email" not in sa or "private_key" not in sa:
        st.error("❌ Thiếu [gcp_service_account] trong Secrets.")
        st.stop()
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    return gspread.authorize(creds)

def diagnose_gsheet_access(spreadsheet_id: str, sheet_name: str):
    sa_email = st.secrets["gcp_service_account"].get("client_email", "(unknown)")
    st.error("Không truy cập được Google Sheet (PermissionError/APIError).")
    st.info(
        "Cách sửa:\n"
        f"- Mở file ID: `{spreadsheet_id}`\n"
        f"- Share cho service account: **{sa_email}** (Editor)\n"
        f"- Tên worksheet cần: **{sheet_name}**\n"
        "- Lưu & Rerun app."
    )

@st.cache_data(ttl=300)
def load_questions_df():
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
        ws = sh.worksheet(QUESTIONS_SHEET_NAME)
    except Exception:
        diagnose_gsheet_access(QUESTIONS_SPREADSHEET_ID, QUESTIONS_SHEET_NAME)
        st.stop()
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return df
    if "q_index" not in df.columns:
        df["q_index"] = range(1, len(df)+1)
    if "quiz_id" in df.columns:
        df = df[df["quiz_id"].astype(str).str.strip() == str(QUIZ_ID)]
    return df.sort_values("q_index")

@st.cache_data(ttl=300)
def load_mcq_questions_df():
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
        ws = sh.worksheet(MCQ_QUESTIONS_SHEET_NAME)
    except Exception:
        diagnose_gsheet_access(QUESTIONS_SPREADSHEET_ID, MCQ_QUESTIONS_SHEET_NAME)
        st.stop()
    df = pd.DataFrame(ws.get_all_records())
    if df.empty:
        return df
    if "q_index" not in df.columns:
        df["q_index"] = range(1, len(df)+1)
    if "quiz_id" in df.columns:
        df = df[df["quiz_id"].astype(str).str.strip() == str(QUIZ_ID)]
    return df.sort_values("q_index")

def _col_idx_to_letter(idx_1based: int) -> str:
    n = idx_1based
    s = ""
    while n > 0:
        n, r = divmod(n-1, 26)
        s = chr(65+r) + s
    return s

def _row1(ws):
    rng = ws.batch_get(['1:1'])
    return rng[0][0] if (rng and rng[0]) else []

# ===== FIX 1: an toàn với dòng rỗng
def attempt_exists_fast(ws, mssv: str) -> bool:
    """
    Kiểm tra MSSV đã có trên sheet chưa.
    Dùng col_values() để tránh IndexError khi có hàng rỗng.
    """
    header = ws.row_values(1)
    if not header:
        return False
    try:
        col_mssv_idx = header.index("MSSV") + 1  # 1-based
    except ValueError:
        return False
    try:
        col_vals = ws.col_values(col_mssv_idx)[1:]  # từ dòng 2
    except Exception:
        return False
    target = str(mssv).strip()
    return any(str(v).strip() == target for v in col_vals if v is not None)

# ===== FIX 2: luôn ghi đúng dòng trống đầu tiên
def _find_row_for_write(header: list, rows: list[list], mssv: str) -> int:
    """
    Trả về số dòng (1-based) để ghi:
    - Nếu đã có MSSV → trả về dòng đó (tránh trùng).
    - Nếu chưa có → tìm dòng trống đầu tiên (các cột định danh rỗng).
    - Nếu không có dòng trống → ghi xuống dòng cuối + 1.
    """
    mssv = str(mssv).strip()
    col_mssv = header.index("MSSV") if "MSSV" in header else None

    # 1) Tồn tại MSSV
    if col_mssv is not None:
        for i, r in enumerate(rows, start=2):
            if len(r) > col_mssv and str(r[col_mssv]).strip() == mssv:
                return i

    # 2) Dòng trống đầu tiên
    id_cols = [c for c in ["MSSV","Họ và Tên","NTNS","Tổ"] if c in header]
    id_idx  = [header.index(c) for c in id_cols]
    for i, r in enumerate(rows, start=2):
        cells = [(r[j].strip() if len(r) > j else "") for j in id_idx]
        if all(c == "" for c in cells):
            return i
        if col_mssv is not None and (len(r) <= col_mssv or str(r[col_mssv]).strip() == ""):
            return i

    # 3) Thêm cuối
    return len(rows) + 2

# =========================
# LỚP (roster gốc) & Responses
# =========================
FORBIDDEN_TOKENS = ("test", "question", "likert", "mcq")
_CLASS_ALLOWED = re.compile(r"^[A-Za-z0-9]{2,20}$")

def is_roster_sheet_name(title: str) -> bool:
    if not isinstance(title, str): return False
    t = title.strip()
    if not _CLASS_ALLOWED.match(t): return False
    tl = t.lower()
    if any(tok in tl for tok in FORBIDDEN_TOKENS): return False
    if not re.search(r"[A-Za-z]", t) or not re.search(r"\d", t): return False
    return True

@st.cache_data(ttl=300)
def get_class_rosters():
    s = sget("CLASS_ROSTERS", "")
    if s:
        raw = [x.strip() for x in re.split(r"[,\s]+", s) if x.strip()]
        return [x for x in raw if is_roster_sheet_name(x)]
    try:
        gc = get_gspread_client()
        sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
        titles = [w.title for w in sh.worksheets()]
        return sorted([t for t in titles if is_roster_sheet_name(t)])
    except Exception:
        return []

def open_roster_ws(class_code: str):
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
        return sh.worksheet(class_code)
    except Exception as e:
        st.error(f"Không mở roster '{class_code}': {e}")
        st.stop()

@st.cache_data(ttl=120)
def load_whitelist_students_by_class(class_code: str):
    ws = open_roster_ws(class_code)
    rows = ws.get_all_values()
    if not rows or len(rows) < 2: return {}
    header = [h.strip() for h in rows[0]]
    data = rows[1:]

    def idx(*names):
        for n in names:
            if n in header: return header.index(n)
        return None

    i_mssv = idx("MSSV","mssv")
    i_name = idx("Họ và Tên","Họ và tên","Ho va Ten","Ho va ten")
    i_dob  = idx("NTNS","Ngày sinh","DOB")
    i_to   = idx("Tổ","to","To")

    if i_mssv is None or i_name is None:
        st.error("Roster lớp thiếu cột 'MSSV' hoặc 'Họ và Tên'."); st.stop()

    wl = {}
    for r in data:
        if len(r) <= i_mssv: continue
        m = r[i_mssv].strip()
        if not m: continue
        wl[m] = {
            "name": normalize_vietnamese_name(r[i_name].strip() if len(r)>i_name else ""),
            "dob":  r[i_dob].strip()  if (i_dob is not None and len(r)>i_dob) else "",
            "to":   r[i_to].strip()   if (i_to  is not None and len(r)>i_to)  else "",
        }
    return wl

def _ensure_header(ws, base_cols, tail_cols):
    header = ws.row_values(1)
    changed = False
    for c in base_cols + tail_cols:
        if c not in header: header.append(c); changed=True
    if changed or not header:
        ws.update("A1", [header])
    return header

def open_likert_response_ws_for_class(class_code: str):
    gc = get_gspread_client()
    name = f"Likert{class_code.strip()}"
    sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=2000, cols=80)
    base = ["TT","MSSV","Họ và Tên","NTNS","Tổ"]
    qcols = [str(i) for i in range(1,37)]
    tail  = ["submitted_at","quiz_id","class"]
    _ensure_header(ws, base, qcols+tail)
    return ws

def open_mcq_response_ws_for_class(class_code: str, n_questions: int):
    gc = get_gspread_client()
    name = f"MCQ{class_code.strip()}"
    sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=2000, cols=200)
    base = ["TT","MSSV","Họ và Tên","NTNS","Tổ"]
    qcols = [str(i) for i in range(1, n_questions+1)]
    tail  = ["score","submitted_at","quiz_id","class"]
    _ensure_header(ws, base, qcols+tail)
    return ws

# =========================
# SHUFFLE HELPERS
# =========================
def stable_perm(n: int, key: str) -> list:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    rng_seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(rng_seed)
    arr = np.arange(n); rng.shuffle(arr)
    return arr.tolist()

def _option_perm_for_student(mssv: str, qidx: int):
    key = f"MCQ|{QUIZ_ID}|{mssv}|{qidx}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    perm = np.arange(4); rng.shuffle(perm)
    return perm.tolist()

# =========================
# STUDENT STATE & LOGIN
# =========================
def init_exam_state():
    st.session_state.setdefault("sv_mssv","")
    st.session_state.setdefault("sv_hoten","")
    st.session_state.setdefault("sv_class","")
    st.session_state.setdefault("sv_allow", False)
    # Likert
    st.session_state.setdefault("likert_started", False)
    st.session_state.setdefault("likert_start_time", None)
    st.session_state.setdefault("likert_precheck_done", False)
    st.session_state.setdefault("sv_order", [])
    st.session_state.setdefault("sv_cursor", 0)
    st.session_state.setdefault("sv_answers", {})
    # MCQ
    st.session_state.setdefault("mcq_started", False)
    st.session_state.setdefault("mcq_start_time", None)
    st.session_state.setdefault("mcq_precheck_done", False)
    st.session_state.setdefault("mcq_cursor", 0)
    st.session_state.setdefault("mcq_answers", {})

def student_gate() -> bool:
    """
    Đăng nhập SV:
    - Chọn lớp (từ roster gốc)
    - Nhập MSSV, Họ & Tên (tự chuẩn hóa)
    - Kiểm tra MSSV tồn tại trong lớp; tên lưu theo roster
    """
    init_exam_state()
    if st.session_state.get("sv_allow"):
        return True

    # --- Tiêu đề + ô mật khẩu cùng hàng ---
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.subheader("Đăng nhập Sinh viên")
    with c2:
        sv_pw = st.text_input("Mật khẩu", value="", placeholder="••••••",
                              type="password", key="sv_gate_pw")

    # --- BẮT BUỘC: có secret và nhập ĐÚNG mới cho hiện form bên dưới ---
    sv_secret = _get_student_password()
    if not sv_secret:
        st.error("Trang Sinh viên đang tạm khóa. Vui lòng liên hệ giảng viên.")
        return False

    if not sv_pw:
        return False

    if sv_pw.strip() != sv_secret:
        st.error("Mật khẩu không đúng.")
        return False

    # --- Qua đây mới render form SV (Lớp / MSSV / Họ tên ...) ---
    with st.form("sv_login_unified"):
        options = get_class_rosters()
        class_code = st.selectbox("Lớp", options=options, index=0 if options else None)
        mssv = st.text_input("MSSV", placeholder="VD: 511256000").strip()
        hoten_input = st.text_input(
            "Họ và Tên (Không phân biệt chữ hoa, thường)"
        ).strip()
        agree = st.checkbox("Tôi xác nhận thông tin trên là đúng.")
        submitted = st.form_submit_button("🔑 Đăng nhập")

    # (Giữ nguyên các đoạn xử lý phía dưới của bạn)


    if not submitted:
        return False

    if not class_code:
        st.error("Chưa có danh sách lớp. Vào tab Giảng viên để tạo lớp.")
        return False
    if not mssv or not hoten_input:
        st.error("Vui lòng nhập MSSV và Họ & Tên.")
        return False
    if not agree:
        st.error("Vui lòng tích xác nhận.")
        return False

    wl = load_whitelist_students_by_class(class_code)  # {mssv: {name, dob, to}}
    if mssv not in wl:
        st.error(f"MSSV không nằm trong lớp {class_code}.")
        return False

    hoten_norm_input = normalize_vietnamese_name(hoten_input)
    roster_name = normalize_vietnamese_name(wl[mssv].get("name", ""))

    if roster_name and hoten_norm_input and hoten_norm_input != roster_name:
        st.warning(
            f"Tên bạn nhập **{hoten_norm_input}** khác với danh sách lớp: **{roster_name}**. "
            "Hệ thống sẽ dùng tên theo danh sách lớp."
        )

    st.session_state.update({
        "sv_class": class_code.strip(),
        "sv_mssv": mssv.strip(),
        "sv_hoten": roster_name or hoten_norm_input,
        "sv_allow": True
    })

    st.success(f"🎓 Xin chào **{st.session_state['sv_hoten']}** ({mssv}) – Lớp {class_code}")
    st.rerun()
    return False

# =========================
# LIKERT EXAM
# =========================
def start_likert_exam(n_questions: int):
    mssv  = st.session_state.get("sv_mssv","")
    hoten = st.session_state.get("sv_hoten","")
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
    return max(0, int(TIME_LIMIT_MIN*60 - spent))

def render_timer_likert():
    rem = remaining_seconds_likert()
    mins, secs = divmod(rem, 60)
    st.markdown(f"⏳ **Thời gian còn lại:** {mins:02d}:{secs:02d}")

def likert36_exam():
    if not st.session_state.get("sv_allow"): st.info("Bạn chưa đăng nhập."); return

    df = load_questions_df()
    n_questions = len(df)
    if n_questions == 0:
        st.warning("Chưa có câu hỏi Likert."); return
    st.success(f"Đề {QUIZ_ID} — {n_questions} câu (Likert 1..5)")

    class_code = st.session_state.get("sv_class","")
    mssv = st.session_state.get("sv_mssv","")

    if not st.session_state.get("likert_started") and not st.session_state.get("likert_precheck_done"):
        ws = open_likert_response_ws_for_class(class_code)
        if attempt_exists_fast(ws, mssv):
            st.error("Bạn đã nộp bài Likert trước đó. Chỉ được làm 1 lần."); return
        st.session_state["likert_precheck_done"] = True

    if not st.session_state.get("likert_started"):
        st.caption(f"Thời gian làm bài: {TIME_LIMIT_MIN} phút")
        if st.button("▶️ Bắt đầu bài Likert", type="primary"):
            start_likert_exam(n_questions); st.rerun()
        return

    render_timer_likert()
    if remaining_seconds_likert() <= 0:
        st.warning("⏱️ Hết thời gian — hệ thống sẽ nộp bài.")
        do_submit_likert(df); return

    order = st.session_state["sv_order"] or list(range(n_questions))
    cur = max(0, min(st.session_state["sv_cursor"], n_questions-1))
    st.session_state["sv_cursor"] = cur

    row = df.iloc[order[cur]]
    qidx = int(row["q_index"])
    qtext = str(row.get("question", f"Câu {qidx}"))

    st.markdown(f"### Câu {cur+1}/{n_questions}")
    st.write(qtext)

    current_val = st.session_state["sv_answers"].get(qidx, None)
    picked = st.radio("Chọn mức độ:",
                      options=[1,2,3,4,5],
                      index=[1,2,3,4,5].index(current_val) if current_val in [1,2,3,4,5] else None,
                      horizontal=True,
                      key=f"radio_{qidx}")
    if picked:
        st.session_state["sv_answers"][qidx] = int(picked)

    st.caption("Gợi ý: 1=Hoàn toàn không đồng ý · 2=Không đồng ý · 3=Trung lập · 4=Đồng ý · 5=Hoàn toàn đồng ý")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("⬅️ Câu trước", use_container_width=True, disabled=(cur==0)):
            st.session_state["sv_cursor"] = max(0, cur-1); st.rerun()
    with c2:
        if st.button("➡️ Câu sau", use_container_width=True, disabled=(cur==n_questions-1)):
            st.session_state["sv_cursor"] = min(n_questions-1, cur+1); st.rerun()
    with c3:
        if st.button("📝 Nộp bài Likert", use_container_width=True):
            do_submit_likert(df)

def do_submit_likert(df_questions: pd.DataFrame):
    mssv = st.session_state.get("sv_mssv","").strip()
    hoten = st.session_state.get("sv_hoten","").strip()
    class_code = st.session_state.get("sv_class","").strip()
    answers = st.session_state.get("sv_answers", {})

    if not (mssv and hoten and class_code):
        st.error("Thiếu thông tin đăng nhập."); return

    if "q_index" in df_questions.columns:
        qindices = sorted(df_questions["q_index"].astype(int).tolist())
    else:
        qindices = list(range(1,37))
    ans_map = {int(q): answers.get(int(q), "") for q in qindices}

    try:
        ws = open_likert_response_ws_for_class(class_code)
        header = ws.row_values(1)

        if attempt_exists_fast(ws, mssv):
            st.error("Bạn đã nộp bài Likert trước đó."); return

        rows = ws.get_all_values()[1:]
        target_row = _find_row_for_write(header, rows, mssv)

        # Ghi thông tin định danh
        for col_name, value in {"MSSV": mssv, "Họ và Tên": hoten, "class": class_code}.items():
            if col_name in header:
                cidx = header.index(col_name)+1
                ws.update_acell(f"{_col_idx_to_letter(cidx)}{target_row}", value)

        info = load_whitelist_students_by_class(class_code).get(mssv, {})
        for col_name, key in {"NTNS":"dob","Tổ":"to"}.items():
            if col_name in header and info.get(key, ""):
                cidx = header.index(col_name)+1
                ws.update_acell(f"{_col_idx_to_letter(cidx)}{target_row}", info[key])

        now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
        updates = []
        for i in range(1,37):
            if str(i) in header:
                cidx = header.index(str(i))+1
                updates.append({"range": f"{_col_idx_to_letter(cidx)}{target_row}", "values": [[ans_map.get(i,"")]]})
        for col_name, value in {"submitted_at": now_iso, "quiz_id": QUIZ_ID, "class": class_code}.items():
            if col_name in header:
                cidx = header.index(col_name)+1
                updates.append({"range": f"{_col_idx_to_letter(cidx)}{target_row}", "values": [[value]]})
        if updates: ws.batch_update(updates)
    except Exception as e:
        st.error(f"Lỗi ghi Likert: {e}"); return

    st.success("✅ Đã nộp bài Likert!")
    for k in ["likert_started","likert_start_time","sv_answers","sv_order","sv_cursor","likert_precheck_done"]:
        st.session_state.pop(k, None)

# =========================
# MCQ EXAM
# =========================
def start_mcq_exam():
    st.session_state.update({"mcq_started": True, "mcq_start_time": time.time(), "mcq_cursor": 0, "mcq_answers": {}})

def remaining_seconds_mcq():
    if not st.session_state.get("mcq_started"):
        return MCQ_TIME_LIMIT_MIN*60
    spent = time.time() - (st.session_state.get("mcq_start_time") or time.time())
    return max(0, int(MCQ_TIME_LIMIT_MIN*60 - spent))

def render_timer_mcq():
    rem = remaining_seconds_mcq()
    mins, secs = divmod(rem, 60)
    st.markdown(f"⏳ **Thời gian còn lại (MCQ):** {mins:02d}:{secs:02d}")

def upsert_mcq_response(mssv, hoten, answers, total_correct, n_questions):
    class_code = st.session_state.get("sv_class","").strip()
    ws = open_mcq_response_ws_for_class(class_code, n_questions)
    header = ws.row_values(1)
    updates = []

    if attempt_exists_fast(ws, mssv):
        st.error("Bạn đã nộp MCQ trước đó."); return

    rows = ws.get_all_values()[1:]
    target_row = _find_row_for_write(header, rows, mssv)

    # Ghi định danh
    for col_name, value in {"MSSV": mssv, "Họ và Tên": hoten, "class": class_code}.items():
        if col_name in header:
            cidx = header.index(col_name)+1
            ws.update_acell(f"{_col_idx_to_letter(cidx)}{target_row}", value)

    info = load_whitelist_students_by_class(class_code).get(mssv, {})
    for col_name, key in {"NTNS":"dob","Tổ":"to"}.items():
        if col_name in header and info.get(key, ""):
            cidx = header.index(col_name)+1
            ws.update_acell(f"{_col_idx_to_letter(cidx)}{target_row}", info[key])

    # Đáp án + meta
    updates = []
    for q in range(1, n_questions+1):
        if str(q) in header:
            cidx = header.index(str(q))+1
            updates.append({"range": f"{_col_idx_to_letter(cidx)}{target_row}",
                            "values": [[answers.get(q,"")]]})
    for col_name, value in {
        "score": total_correct,
        "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "quiz_id": QUIZ_ID,
        "class": class_code
    }.items():
        if col_name in header:
            cidx = header.index(col_name)+1
            updates.append({"range": f"{_col_idx_to_letter(cidx)}{target_row}",
                            "values": [[value]]})
    if updates: ws.batch_update(updates)

def mcq_exam():
    if not st.session_state.get("sv_allow"): st.info("Bạn chưa đăng nhập."); return
    df = load_mcq_questions_df()
    if df.empty: st.warning("Chưa có câu hỏi MCQ."); return

    mssv  = st.session_state.get("sv_mssv","")
    hoten = st.session_state.get("sv_hoten","")
    class_code = st.session_state.get("sv_class","").strip()
    n = len(df)
    st.success(f"Đề MCQ {QUIZ_ID} — {n} câu (4 đáp án).")

    if not st.session_state.get("mcq_started") and not st.session_state.get("mcq_precheck_done"):
        ws = open_mcq_response_ws_for_class(class_code, n)
        if attempt_exists_fast(ws, mssv):
            st.error("Bạn đã nộp MCQ trước đó."); return
        st.session_state["mcq_precheck_done"] = True

    if not st.session_state.get("mcq_started"):
        st.caption(f"Thời gian làm bài: {MCQ_TIME_LIMIT_MIN} phút")
        if st.button("▶️ Bắt đầu bài MCQ", type="primary"):
            start_mcq_exam(); st.rerun()
        return

    render_timer_mcq()
    if remaining_seconds_mcq() <= 0:
        st.warning("⏱️ Hết thời gian — hệ thống sẽ nộp bài.")
        total = 0; ans = st.session_state["mcq_answers"]
        for _, r in df.iterrows():
            qi = int(r["q_index"])
            if ans.get(qi,"") == str(r["correct"]).strip().upper():
                total += 1
        try:
            upsert_mcq_response(mssv, hoten, ans, total, n)
            st.success(f"✅ Đã nộp MCQ. Điểm: {total}/{n}")
        except Exception as e:
            st.error(f"Lỗi ghi MCQ: {e}")
        for k in ["mcq_cursor","mcq_answers","mcq_started","mcq_start_time","mcq_precheck_done"]:
            st.session_state.pop(k, None)
        return

    order = stable_perm(n, f"MCQ_ORDER|{QUIZ_ID}|{mssv}|{hoten}")
    cur = max(0, min(st.session_state.get("mcq_cursor",0), n-1))
    st.session_state["mcq_cursor"] = cur

    row = df.iloc[order[cur]]
    qidx = int(row["q_index"])
    qtext = str(row["question"])
    options = [str(row["optionA"]), str(row["optionB"]), str(row["optionC"]), str(row["optionD"])]

    st.markdown(f"### Câu {cur+1}/{n}")
    st.write(qtext)

    perm = _option_perm_for_student(mssv, qidx)
    shuffled_opts = [options[i] for i in perm]
    labels = ['A','B','C','D']
    inv = {labels[i]: ['A','B','C','D'][perm[i]] for i in range(4)}

    pick = st.radio("Chọn đáp án:",
                    options=[f"{labels[i]}. {shuffled_opts[i]}" for i in range(4)],
                    index=None,
                    key=f"mcq_{qidx}")
    if pick:
        chosen = pick.split('.',1)[0].strip()
        st.session_state["mcq_answers"][qidx] = inv[chosen]

    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("⬅️ Câu trước", use_container_width=True, disabled=(cur==0)):
            st.session_state["mcq_cursor"] = max(0, cur-1); st.rerun()
    with c2:
        if st.button("➡️ Câu sau", use_container_width=True, disabled=(cur==n-1)):
            st.session_state["mcq_cursor"] = min(n-1, cur+1); st.rerun()
    with c3:
        if st.button("🧹 Xóa chọn", use_container_width=True):
            st.session_state["mcq_answers"].pop(qidx, None); st.rerun()
    with c4:
        if st.button("📝 Nộp MCQ", use_container_width=True, type="primary"):
            total = 0; ans = st.session_state["mcq_answers"]
            for _, r in df.iterrows():
                qi = int(r["q_index"])
                if ans.get(qi,"") == str(r["correct"]).strip().upper():
                    total += 1
            upsert_mcq_response(mssv, hoten, ans, total, n)
            st.success(f"✅ Đã nộp MCQ. Điểm: {total}/{n}")
            for k in ["mcq_cursor","mcq_answers","mcq_started","mcq_start_time","mcq_precheck_done"]:
                st.session_state.pop(k, None)

# =========================
# TEACHER (GV) PANEL
# =========================
def _get_teacher_creds_strict():
    """Đọc user/pass từ Secrets (root hoặc [app]); dừng nếu thiếu."""
    def _pick(scope):
        if not scope: return None, None
        u = scope.get("TEACHER_USER"); p = scope.get("TEACHER_PASS")
        return _normalize_credential(u), _normalize_credential(p)

    u, p = _pick(st.secrets)
    if not u or not p: u, p = _pick(st.secrets.get("app", {}))
    if not u or not p:
        st.error("❌ Chưa cấu hình TEACHER_USER / TEACHER_PASS trong Secrets."); st.stop()
    return u, p

def teacher_login() -> bool:
    st.subheader("Đăng nhập Giảng viên")

    # Nếu đã đăng nhập
    if st.session_state.get("is_teacher", False):
        st.success("Đã đăng nhập.")
        if st.button("🚪 Đăng xuất GV", type="secondary", key="logout_gv_btn_simple"):
            st.session_state["is_teacher"] = False
            st.success("Đã đăng xuất."); st.rerun()
        return True

    # Username mặc định trong secrets; chỉ yêu cầu mật khẩu
    p_val = st.text_input("Mật khẩu", value="", placeholder="••••••", type="password", key="gv_pass_only")

    if st.button("Đăng nhập", type="primary", key="gv_login_btn_simple"):
        p_in = _normalize_credential(p_val)
        if not p_in:
            st.error("Vui lòng nhập mật khẩu."); return False

        # Lấy user/pass chuẩn từ secrets
        u_sec, p_sec = _get_teacher_creds_strict()
        if p_in == _normalize_credential(p_sec):
            st.session_state["is_teacher"] = True
            # Dọn state cũ (nếu có)
            for k in ("gv_pass", "gv_pass_simple", "gv_pass_only"):
                st.session_state.pop(k, None)
            st.success("Đăng nhập thành công."); st.rerun()
        else:
            st.error("Sai mật khẩu.")
            with st.expander("🔧 Chẩn đoán đăng nhập (không lộ mật khẩu)"):
                st.write({
                    "expected_pass_length": len(p_sec),
                    "input_pass_length": len(p_in),
                })
    return False

def _diagnose_questions():
    st.markdown("#### 🔎 Kiểm tra Question")
    try:
        gc = get_gspread_client()
        sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
        ws_titles = [w.title for w in sh.worksheets()]
        st.success("✅ Kết nối được file câu hỏi."); st.write("Worksheets:", ws_titles)
        if QUESTIONS_SHEET_NAME in ws_titles:
            st.info(f"Worksheet Likert: **{QUESTIONS_SHEET_NAME}**")
        if MCQ_QUESTIONS_SHEET_NAME in ws_titles:
            st.info(f"Worksheet MCQ: **{MCQ_QUESTIONS_SHEET_NAME}**")
    except Exception as e:
        st.error(f"Không mở được file câu hỏi: {e}")

def _view_questions():
    st.markdown("#### 📋 Ngân hàng câu hỏi Likert")
    dfq = load_questions_df()
    if dfq.empty: st.warning("Worksheet Likert trống.")
    else:
        st.dataframe(dfq, use_container_width=True, height=420)
        st.caption(f"Tổng: **{len(dfq)}** câu")
    with st.expander("🔎 Chẩn đoán"): _diagnose_questions()

def push_questions(df: pd.DataFrame):
    need = {"q_index","question"}
    if not need.issubset(df.columns):
        miss = ", ".join(sorted(need - set(df.columns)))
        st.error(f"Thiếu cột: {miss}"); return
    df = df.copy()
    df["q_index"] = pd.to_numeric(df["q_index"], errors="coerce").astype("Int64")
    if "quiz_id" not in df.columns: df["quiz_id"]=QUIZ_ID
    cols = ["quiz_id","q_index","facet","question","left_label","right_label","reverse"]
    for c in cols:
        if c not in df.columns: df[c] = ""
    df = df[cols].sort_values(["quiz_id","q_index"])

    gc = get_gspread_client()
    sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
    try:
        try:
            ws = sh.worksheet(QUESTIONS_SHEET_NAME); ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=QUESTIONS_SHEET_NAME, rows=2000, cols=20)
        ws.append_row(list(df.columns))
        if len(df)>0: ws.append_rows(df.astype(object).values.tolist())
        load_questions_df.clear()
        st.success(f"✅ Đã ghi {len(df)} dòng vào **{QUESTIONS_SHEET_NAME}**.")
    except Exception as e:
        st.error(f"Lỗi ghi: {e}")

def _upload_questions():
    st.markdown("#### 📥 Tải câu hỏi Likert (CSV/XLSX)")
    st.info("Cột: quiz_id | q_index | facet | question | left_label | right_label | reverse (tối thiểu: q_index, question).")
    up = st.file_uploader("Chọn file Likert", type=["csv","xlsx"], key="likert_uploader")
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                import openpyxl
                df = pd.read_excel(up)
        except Exception as e:
            st.error(f"Không đọc được file: {e}"); return
        st.dataframe(df.head(12), use_container_width=True)
        if st.button("Ghi lên Question", type="primary", key="write_likert"):
            push_questions(df)
    with st.expander("🔎 Chẩn đoán"): _diagnose_questions()

def push_mcq_questions(df: pd.DataFrame):
    need = {"q_index","question","optionA","optionB","optionC","optionD","correct"}
    if not need.issubset(df.columns):
        miss = ", ".join(sorted(need - set(df.columns)))
        st.error(f"Thiếu cột MCQ: {miss}"); return
    df = df.copy()
    df["q_index"] = pd.to_numeric(df["q_index"], errors="coerce").astype("Int64")
    if "quiz_id" not in df.columns: df["quiz_id"]=QUIZ_ID
    cols = ["quiz_id","q_index","question","optionA","optionB","optionC","optionD","correct"]
    for c in cols:
        if c not in df.columns: df[c]=""
    df = df[cols].sort_values(["quiz_id","q_index"])

    gc = get_gspread_client()
    sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
    try:
        try:
            ws = sh.worksheet(MCQ_QUESTIONS_SHEET_NAME); ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=MCQ_QUESTIONS_SHEET_NAME, rows=2000, cols=30)
        ws.append_row(list(df.columns))
        if len(df)>0: ws.append_rows(df.astype(object).values.tolist())
        load_mcq_questions_df.clear()
        st.success(f"✅ Đã ghi {len(df)} dòng vào **{MCQ_QUESTIONS_SHEET_NAME}**.")
    except Exception as e:
        st.error(f"Lỗi ghi MCQ: {e}")

def _upload_mcq_questions():
    st.markdown("#### 🧩 Tải câu hỏi MCQ (CSV/XLSX)")
    st.info("Cột: quiz_id | q_index | question | optionA..D | correct (A/B/C/D).")
    up = st.file_uploader("Chọn file MCQ", type=["csv","xlsx"], key="mcq_uploader")
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                import openpyxl
                df = pd.read_excel(up)
        except Exception as e:
            st.error(f"Không đọc được file: {e}"); return
        st.dataframe(df.head(12), use_container_width=True)
        if st.button("Ghi lên MCQ_Questions", type="primary", key="write_mcq"):
            push_mcq_questions(df)

def _create_new_class_tab():
    st.markdown("#### 🏫 Tạo lớp mới")
    st.info("Roster mẫu: **STT | MSSV | Họ và Tên | NTNS | Tổ**. Tên worksheet là mã lớp (VD: D25B).")
    class_name = st.text_input("Tên lớp", placeholder="VD: D25B").strip()
    up = st.file_uploader("Chọn file roster (CSV/XLSX)", type=["csv","xlsx"], key="roster_uploader")
    if st.button("Tạo lớp", type="primary", disabled=(not class_name)):
        # Kiểm tra tên lớp
        if not is_roster_sheet_name(class_name):
            st.error("Tên lớp không hợp lệ (chỉ chữ/số, có ≥1 chữ & ≥1 số, không chứa test/question/likert/mcq)."); return
        # Đọc dữ liệu
        if up is not None:
            try:
                if up.name.lower().endswith(".csv"): df = pd.read_csv(up)
                else:
                    import openpyxl; df = pd.read_excel(up)
            except Exception as e:
                st.error(f"Không đọc được file: {e}"); return
        else:
            df = pd.DataFrame(columns=["STT","MSSV","Họ và Tên","NTNS","Tổ"])
        for c in ["STT","MSSV","Họ và Tên","NTNS","Tổ"]:
            if c not in df.columns: df[c]=""
        df["Họ và Tên"] = df["Họ và Tên"].apply(normalize_vietnamese_name)
        df = df[["STT","MSSV","Họ và Tên","NTNS","Tổ"]]
        try:
            gc = get_gspread_client()
            sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
            try:
                ws = sh.worksheet(class_name); ws.clear()
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(title=class_name, rows=max(100,len(df)+2), cols=10)
            ws.append_row(["STT","MSSV","Họ và Tên","NTNS","Tổ"])
            if len(df)>0: ws.append_rows(df.astype(object).values.tolist())
            load_whitelist_students_by_class.clear()
            st.success(f"✅ Đã tạo/ghi roster lớp **{class_name}**.")
        except Exception as e:
            st.error(f"Lỗi tạo lớp: {e}")

def _read_mcq_sheet(class_code: str) -> pd.DataFrame:
    gc = get_gspread_client()
    sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
    wsname = f"MCQ{class_code}"
    try:
        ws = sh.worksheet(wsname)
    except gspread.WorksheetNotFound:
        st.warning(f"Chưa có sheet {wsname}."); return pd.DataFrame()
    return pd.DataFrame(ws.get_all_records())

def _mcq_stats_tab():
    st.markdown("#### 📊 Thống kê MCQ")
    classes = get_class_rosters()
    if not classes: st.info("Chưa có roster lớp."); return
    class_code = st.selectbox("Chọn lớp", options=classes)
    df = _read_mcq_sheet(class_code)
    if df.empty: st.info("Chưa có dữ liệu MCQ cho lớp này."); return
    qcols = [c for c in df.columns if str(c).isdigit()]
    if not qcols: st.info("Không thấy cột câu hỏi (1..N)."); return
    qnums = sorted([int(c) for c in qcols])
    q_choice = st.selectbox("Chọn câu", options=qnums, index=0)
    col = str(q_choice)

    counts = df[col].astype(str).str.strip().str.upper().value_counts()
    total = int(counts.sum())
    data = []
    for label in ["A","B","C","D"]:
        c = int(counts.get(label,0))
        pct = (c/total*100) if total>0 else 0.0
        data.append({"Đáp án":label,"Số người":c,"Tỷ lệ (%)":round(pct,2)})
    dstat = pd.DataFrame(data)
    st.dataframe(dstat, use_container_width=True, height=200)

    if HAS_PLOTLY:
        fig = px.bar(dstat, x="Đáp án", y="Số người", color="Đáp án",
                     hover_data={"Tỷ lệ (%)":True,"Số người":True,"Đáp án":False},
                     text="Số người")
        fig.update_layout(yaxis_title="Số người", xaxis_title="Đáp án", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart = (alt.Chart(dstat).mark_bar()
                 .encode(x=alt.X("Đáp án:N", title="Đáp án"),
                         y=alt.Y("Số người:Q", title="Số người"),
                         color="Đáp án:N",
                         tooltip=[alt.Tooltip("Đáp án:N"),
                                  alt.Tooltip("Số người:Q"),
                                  alt.Tooltip("Tỷ lệ (%):Q")])
                 .interactive())
        st.altair_chart(chart, use_container_width=True)

def _parse_ts(s):
    try: return pd.to_datetime(s)
    except Exception: return pd.NaT

def _ai_answer_from_df(df: pd.DataFrame, query: str) -> str:
    if df.empty: return "Không có dữ liệu."
    q = (query or "").strip().lower()
    dfc = df.copy()
    dfc["score_num"] = pd.to_numeric(dfc.get("score", np.nan), errors="coerce")
    dfc["ts"] = dfc.get("submitted_at", pd.Series([np.nan]*len(dfc))).apply(_parse_ts)

    if any(k in q for k in ["sớm","som","sớm nhất"]):
        dfv = dfc.dropna(subset=["ts"]).sort_values("ts")
        if len(dfv): r=dfv.iloc[0]; who=r.get('Họ và Tên','') or r.get('MSSV','?'); return f"Sớm nhất: {who} — {r.get('ts')}"
        return "Không có timestamp."
    if any(k in q for k in ["muộn","muon","trễ","tre","muộn nhất"]):
        dfv = dfc.dropna(subset=["ts"]).sort_values("ts")
        if len(dfv): r=dfv.iloc[-1]; who=r.get('Họ và Tên','') or r.get('MSSV','?'); return f"Muộn nhất: {who} — {r.get('ts')}"
        return "Không có timestamp."
    if any(k in q for k in ["cao điểm","cao","max","highest"]):
        dfv = dfc.dropna(subset=["score_num"]).sort_values("score_num")
        if len(dfv): r=dfv.iloc[-1]; who=r.get('Họ và Tên','') or r.get('MSSV','?'); return f"Cao điểm nhất: {who} — {int(r['score_num'])}"
        return "Chưa có điểm."
    if any(k in q for k in ["thấp điểm","thấp","min","lowest"]):
        dfv = dfc.dropna(subset=["score_num"]).sort_values("score_num")
        if len(dfv): r=dfv.iloc[0]; who=r.get('Họ và Tên','') or r.get('MSSV','?'); return f"Thấp điểm nhất: {who} — {int(r['score_num'])}"
        return "Chưa có điểm."
    return "Từ khóa gợi ý: sớm nhất, muộn nhất, cao điểm, thấp điểm."

def _ai_assistant_tab():
    st.markdown("#### 🤖 Trợ lý AI (từ khóa ngắn)")
    classes = get_class_rosters()
    if not classes: st.info("Chưa có roster lớp."); return
    class_code = st.selectbox("Chọn lớp", options=classes, key="ai_class")
    df = _read_mcq_sheet(class_code)
    if df.empty: st.info("Chưa có dữ liệu MCQ."); return
    if "score" not in df.columns: st.warning("Sheet MCQ chưa có cột 'score'.")
    if "submitted_at" not in df.columns: st.warning("Sheet MCQ chưa có cột 'submitted_at'.")
    q = st.text_input("Nhập từ khóa (vd: sớm nhất / muộn nhất / cao điểm / thấp điểm)")
    if st.button("Hỏi"):
        st.write(_ai_answer_from_df(df, q))

def _diagnose_responses():
    st.markdown("#### ℹ️ Ghi chú Responses")
    st.info("Kết quả ghi theo lớp: Likert<CLASS> / MCQ<CLASS> (VD: LikertD25A, MCQD25A).")

def teacher_panel():
    if not teacher_login(): return
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Xem Likert",
        "📥 Tải Likert",
        "🧩 Tải MCQ",
        "🏫 Tạo lớp",
        "📊 Thống kê MCQ",
        "🤖 Trợ lý AI",
    ])
    with tab1: _view_questions()
    with tab2: _upload_questions()
    with tab3: _upload_mcq_questions()
    with tab4: _create_new_class_tab()
    with tab5: _mcq_stats_tab()
    with tab6: _ai_assistant_tab()

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.header("Chức năng")
page = st.sidebar.radio("Đi đến", ["Sinh viên", "Giảng viên", "Hướng dẫn"], index=0)

if page == "Sinh viên":
    render_banner()
    
    # Đăng xuất SV
    if st.session_state.get("sv_allow") or st.session_state.get("likert_started") or st.session_state.get("mcq_started"):
        if st.button("🚪 Đăng xuất", type="secondary"):
            for k in list(st.session_state.keys()):
                if k.startswith("sv_") or k.startswith("mcq_") or k.startswith("likert_"):
                    st.session_state.pop(k, None)
            st.success("Đã đăng xuất."); st.stop()

    if not student_gate(): st.stop()

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
        "- **Sinh viên**: đăng nhập (Lớp + MSSV + Họ & Tên) → chọn **Likert** hoặc **MCQ** → Bắt đầu (bắt giờ) → Nộp bài.\n"
        "- **Giảng viên**: xem/tải ngân hàng **Likert/MCQ**, **tạo lớp**, **thống kê MCQ**, **trợ lý AI**.\n"
        "- Kết quả ghi vào sheet: **Likert<CLASS>**, **MCQ<CLASS>** trong file Responses."
    )

st.markdown("---")
st.markdown("© Bản quyền thuộc về TS...")

st.markdown("---")
st.markdown("© Bản quyền thuộc về TS...")
