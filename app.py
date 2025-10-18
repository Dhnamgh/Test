# app.py
import time
from datetime import datetime, timedelta, timezone
import hashlib
import json

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Trắc nghiệm Likert 36", layout="wide")

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
TEACHER_USER = str(sget("TEACHER_USER", "teacher")).strip()
TEACHER_PASS = str(sget("TEACHER_PASS", "teacher123")).strip()

# Google Sheets config
QUESTIONS_SPREADSHEET_ID = srequire("QUESTIONS_SPREADSHEET_ID")
QUESTIONS_SHEET_NAME     = sget("QUESTIONS_SHEET_NAME", "Question")

RESPONSES_SPREADSHEET_ID = srequire("RESPONSES_SPREADSHEET_ID")
RESPONSES_SHEET_NAME     = sget("RESPONSES_SHEET_NAME", "D25Atest")

# ------------------ RESPONSES HELPERS (ghi bài SV không xóa sheet) ------------------

def open_responses_ws():
    """Mở worksheet Responses (D25Atest)."""
    gc = get_gspread_client()
    try:
        sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
    except Exception:
        diagnose_gsheet_access(RESPONSES_SPREADSHEET_ID, RESPONSES_SHEET_NAME)
        st.stop()
    try:
        ws = sh.worksheet(RESPONSES_SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=RESPONSES_SHEET_NAME, rows=1000, cols=50)
    ensure_responses_header(ws)
    return ws


def ensure_responses_header(ws):
    """Đảm bảo worksheet Responses có header đúng nhưng KHÔNG xóa dữ liệu cũ."""
    header = ws.row_values(1)
    base_header = ["TT", "MSSV", "Họ và Tên", "NTNS"]
    question_cols = [str(i) for i in range(1, 37)]
    extra_cols = ["submitted_at", "quiz_id"]

    # Nếu thiếu cột nào thì thêm vào cuối header
    for col in base_header + question_cols + extra_cols:
        if col not in header:
            header.append(col)

    # Ghi lại header (chỉ hàng 1)
    ws.update("A1", [header])


def upsert_response(mssv: str, hoten: str, dob: str, answers: dict):
    """
    Ghi hoặc cập nhật bài làm SV theo MSSV mà không xóa dữ liệu cũ.
    Ghi dữ liệu các câu hỏi (1..36), submitted_at, quiz_id.
    """
    ws = open_responses_ws()
    values = ws.get_all_values()
    if not values:
        st.error("Sheet Responses trống — cần có header trước.")
        return

    header = values[0]
    data = values[1:]

    # Tìm vị trí cột MSSV
    if "MSSV" not in header:
        st.error("Không tìm thấy cột 'MSSV' trong sheet Responses.")
        return
    mssv_col_index = header.index("MSSV")

    # Tìm dòng có MSSV khớp
    found_row = None
    for idx, row in enumerate(data, start=2):
        if len(row) > mssv_col_index and row[mssv_col_index].strip() == mssv.strip():
            found_row = idx
            break

    # Nếu không có → thêm dòng mới cuối bảng
    target_row = found_row or len(data) + 2

    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # Tạo dict giá trị cần ghi
    update_dict = {
        "Họ và Tên": hoten,
        "NTNS": dob,
        "submitted_at": now_iso,
        "quiz_id": QUIZ_ID,
    }
    for i in range(1, 37):
        update_dict[str(i)] = answers.get(i, "")

    # Ghi từng ô theo cột có tên trùng
    for col_name, value in update_dict.items():
        if col_name not in header:
            continue
        col_index = header.index(col_name) + 1  # 1-based
        col_letter = ""
        # chuyển số cột sang chữ (A, B, ..., AA, AB...)
        n = col_index
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            col_letter = chr(65 + remainder) + col_letter
        cell_label = f"{col_letter}{target_row}"
        ws.update_acell(cell_label, value)

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

# =========================
# STUDENT EXAM UI
# =========================
def init_exam_state():
    st.session_state.setdefault("sv_mssv", "")
    st.session_state.setdefault("sv_hoten", "")
    st.session_state.setdefault("sv_dob", "")
    st.session_state.setdefault("sv_started", False)
    st.session_state.setdefault("sv_start_time", None)
    st.session_state.setdefault("sv_answers", {})      # {q_index -> 1..5}
    st.session_state.setdefault("sv_order", [])        # hoán vị câu hỏi
    st.session_state.setdefault("sv_cursor", 0)        # index đang hiển thị

def start_exam(mssv, hoten, dob, n_questions):
    init_exam_state()
    st.session_state["sv_mssv"] = mssv.strip()
    st.session_state["sv_hoten"] = hoten.strip()
    st.session_state["sv_dob"] = dob.strip()
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
    # Tự refresh mỗi giây
    st.autorefresh(interval=1000, key="timer_refresh")

def likert36_exam():
    init_exam_state()
    df = load_questions_df()
    # Kỳ vọng cột: quiz_id | q_index | question | left_label | right_label | reverse
    # Lọc theo QUIZ_ID nếu có cột quiz_id
    if "quiz_id" in df.columns:
        df = df[df["quiz_id"].astype(str).str.strip() == str(QUIZ_ID)].copy()
    # Nếu không có q_index, tạo từ 1..len
    if "q_index" not in df.columns:
        df["q_index"] = range(1, len(df) + 1)
    df = df.sort_values("q_index")
    n_questions = len(df)

    st.success(f"Đề {QUIZ_ID} — {n_questions} câu (Likert 1..5)")

    # ---- Đăng nhập SV / bắt đầu ----
    if not st.session_state.get("sv_started"):
        with st.form("sv_login"):
            col1, col2, col3 = st.columns([1, 1, 1.2])
            with col1:
                mssv = st.text_input("MSSV", placeholder="VD: 2112345")
            with col2:
                hoten = st.text_input("Họ và Tên", placeholder="VD: Nguyễn Văn A")
            with col3:
                dob = st.text_input("NTNS (tùy chọn)", placeholder="VD: 01/01/2000")
            agree = st.checkbox("Tôi xác nhận thông tin trên là đúng.")
            submitted = st.form_submit_button("Bắt đầu làm bài")
        if submitted:
            if not mssv or not hoten:
                st.error("Vui lòng nhập MSSV và Họ & Tên.")
            elif not agree:
                st.error("Vui lòng tích xác nhận.")
            else:
                start_exam(mssv, hoten, dob, n_questions)
                st.rerun()
        st.info("SV chỉ tiếp cận tab **Sinh viên**. Sau khi bắt đầu sẽ có đồng hồ đếm ngược.")
        return

    # ---- Đang làm bài ----
    render_timer()
    if remaining_seconds() <= 0:
        st.warning("⏱️ Hết thời gian — hệ thống sẽ nộp bài với các câu đã chọn.")
        do_submit(df)
        return

    # Ánh xạ q_index theo hoán vị
    order = st.session_state["sv_order"] or list(range(n_questions))
    cur = st.session_state["sv_cursor"]
    cur = max(0, min(cur, n_questions - 1))
    st.session_state["sv_cursor"] = cur

    # Lấy dòng câu hỏi hiện tại
    row = df.iloc[order[cur]]
    qidx = int(row["q_index"])
    qtext = str(row.get("question", f"Câu {qidx}"))
    left_label = str(row.get("left_label", "Hoàn toàn không đồng ý"))
    right_label = str(row.get("right_label", "Hoàn toàn đồng ý"))

    st.markdown(f"### Câu {cur+1}/{n_questions}")
    st.write(qtext)

    # Giá trị hiện có
    current_val = st.session_state["sv_answers"].get(qidx, None)
    # Radio 1..5
    picked = st.radio(
        "Chọn mức độ:",
        options=[1, 2, 3, 4, 5],
        index=[1,2,3,4,5].index(current_val) if current_val in [1,2,3,4,5] else None,
        horizontal=True,
        key=f"radio_{qidx}"
    )
    # Ghi tạm
    if picked:
        st.session_state["sv_answers"][qidx] = int(picked)

    # Nhãn trái/phải
    cL, cR = st.columns(2)
    with cL:
        st.caption(f"1 = {left_label}")
    with cR:
        st.caption(f"5 = {right_label}")

    # Điều hướng
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
        st.button("📝 Nộp bài", use_container_width=True, on_click=lambda: None)
        # Để xử lý nộp bài có xác nhận:
        if st.session_state.get("submit_clicked_once") is None:
            st.session_state["submit_clicked_once"] = False
        if st.button("✅ Xác nhận nộp"):
            st.session_state["submit_clicked_once"] = True
            do_submit(df)

def do_submit(df_questions: pd.DataFrame):
    """Nộp bài: ghi lên sheet Responses."""
    mssv = st.session_state.get("sv_mssv", "").strip()
    hoten = st.session_state.get("sv_hoten", "").strip()
    dob = st.session_state.get("sv_dob", "").strip()
    answers = st.session_state.get("sv_answers", {})

    if not mssv or not hoten:
        st.error("Thiếu MSSV hoặc Họ & Tên.")
        return

    # Đảm bảo 36 cột 1..36
    ans_map = {}
    # Nếu df có q_index thì lấy theo đó
    if "q_index" in df_questions.columns:
        qindices = sorted(df_questions["q_index"].astype(int).tolist())
    else:
        qindices = list(range(1, 37))
    for q in qindices:
        ans_map[int(q)] = answers.get(int(q), "")

    try:
        upsert_response(mssv, hoten, dob, ans_map)
    except Exception as e:
        st.error(f"Lỗi ghi Responses: {e}")
        return

    st.success("✅ Đã nộp bài thành công!")
    # Khóa bài thi: reset trạng thái
    for k in ["sv_started", "sv_start_time", "sv_answers", "sv_order", "sv_cursor"]:
        st.session_state.pop(k, None)

# =========================
# TEACHER PANEL (FULL)
# =========================

# Lấy user/pass từ secrets và loại khoảng trắng vô tình
TEACHER_USER = str(sget("TEACHER_USER", "teacher")).strip()
TEACHER_PASS = str(sget("TEACHER_PASS", "teacher123")).strip()

def teacher_login() -> bool:
    """Đăng nhập giảng viên đơn giản bằng secrets."""
    st.subheader("Đăng nhập Giảng viên")

    # Đã login trong session
    if st.session_state.get("is_teacher", False):
        st.success("Đã đăng nhập.")
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Đăng xuất", use_container_width=True):
                st.session_state["is_teacher"] = False
                st.rerun()
        return True

    # Form đăng nhập
    with st.form("teacher_login_form", clear_on_submit=False):
        u = st.text_input("Tài khoản", value="", placeholder="teacher")
        p = st.text_input("Mật khẩu", value="", placeholder="••••••", type="password")
        ok = st.form_submit_button("Đăng nhập")

    if ok:
        if u.strip() == TEACHER_USER and p == TEACHER_PASS:
            st.session_state["is_teacher"] = True
            st.success("Đăng nhập thành công.")
            st.rerun()
        else:
            st.error("Sai tài khoản hoặc mật khẩu.")
    return False


def _diagnose_questions():
    """Chẩn đoán nhanh kết nối Question sheet."""
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


def _diagnose_responses():
    """Chẩn đoán nhanh kết nối Responses sheet."""
    st.markdown("#### 🔎 Kiểm tra Responses sheet")
    try:
        ws = open_responses_ws()
        st.success(f"✅ Mở được worksheet: **{RESPONSES_SHEET_NAME}** (file ID {RESPONSES_SPREADSHEET_ID})")
        st.write("Số dòng hiện có (kể cả header):", ws.row_count)
    except Exception as e:
        st.error(f"Không mở được Responses: {e}")


def _view_questions():
    st.markdown("#### 📋 Ngân hàng câu hỏi hiện tại")
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
    Ghi ĐÈ toàn bộ worksheet câu hỏi bằng dataframe cung cấp.
    Cần cột tối thiểu: q_index, question.
    Khuyến nghị cột đầy đủ: quiz_id | q_index | facet | question | left_label | right_label | reverse
    """
    # Kiểm tra cột
    required = {"q_index", "question"}
    if not required.issubset(set(df.columns)):
        missing = ", ".join(sorted(required - set(df.columns)))
        st.error(f"Thiếu cột bắt buộc: {missing}")
        return

    # Ép kiểu cơ bản
    if "q_index" in df.columns:
        df["q_index"] = pd.to_numeric(df["q_index"], errors="coerce").astype("Int64")

    # Mặc định quiz_id nếu thiếu
    if "quiz_id" not in df.columns:
        df["quiz_id"] = QUIZ_ID
    else:
        df["quiz_id"] = df["quiz_id"].fillna(QUIZ_ID)

    # Trật tự cột gợi ý
    columns_order = ["quiz_id", "q_index", "facet", "question", "left_label", "right_label", "reverse"]
    for c in columns_order:
        if c not in df.columns:
            df[c] = ""
    df = df[columns_order].sort_values(["quiz_id", "q_index"], na_position="last")

    # Ghi lên Google Sheets
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

        # Viết header + dữ liệu
        ws.append_row(list(df.columns))
        if len(df) > 0:
            ws.append_rows(df.astype(object).values.tolist())

        # Làm mới cache câu hỏi
        load_questions_df.clear()
        st.success(f"✅ Đã ghi {len(df)} dòng vào **{QUESTIONS_SHEET_NAME}**.")
    except Exception as e:
        st.error(f"Lỗi ghi dữ liệu lên sheet: {e}")


def _upload_questions():
    st.markdown("#### 📥 Tải câu hỏi (CSV/XLSX)")
    st.info(
        "File nên có cột: **quiz_id | q_index | facet | question | left_label | right_label | reverse**.\n"
        "Tối thiểu bắt buộc: **q_index, question**. Nếu thiếu `quiz_id`, hệ thống sẽ điền mặc định."
    )
    up = st.file_uploader("Chọn file câu hỏi", type=["csv", "xlsx"])

    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df_new = pd.read_csv(up)
            else:
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


def _view_responses():
    st.markdown("#### 📑 Xem bài làm (Responses)")
    try:
        ws = open_responses_ws()
        # Lấy ~1000 dòng để xem nhanh
        rows = ws.get_all_values()
        if not rows:
            st.info("Sheet trống.")
            return
        df = pd.DataFrame(rows[1:], columns=rows[0])
        st.dataframe(df, use_container_width=True, height=420)
        st.caption(f"Số bài ghi nhận: **{len(df)}** (không tính header)")
    except Exception as e:
        st.error(f"Lỗi đọc Responses: {e}")

    with st.expander("🔎 Chẩn đoán"):
        _diagnose_responses()


def teacher_panel():
    """UI chính của tab Giảng viên."""
    if not teacher_login():
        return

    st.header("Bảng điều khiển Giảng viên")

    # Tabs chức năng
    tab1, tab2, tab3 = st.tabs([
        "📋 Xem câu hỏi",
        "📥 Tải câu hỏi",
        "📑 Xem bài làm"
    ])

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
    st.title("Sinh viên làm bài")
    likert36_exam()

elif page == "Giảng viên":
    st.title("Khu vực Giảng viên")
    teacher_panel()

else:
    st.title("Hướng dẫn nhanh")
    st.markdown(
        """
- **Sinh viên:** nhập MSSV + Họ & Tên → Bắt đầu → làm lần lượt 36 câu (Likert 1..5) → **Nộp bài**.  
  Có **đồng hồ đếm ngược** theo `TIME_LIMIT_MIN`.
- **Giảng viên:** đăng nhập để xem câu hỏi đang dùng và có thể **tải (CSV/XLSX)** để cập nhật ngân hàng câu hỏi.
- **Google Sheets:**
  - **Question**: ngân hàng câu hỏi (cột gợi ý: `quiz_id | q_index | question | left_label | right_label | reverse`)
  - **D25Atest**: nơi lưu bài làm; nếu trống app sẽ tự tạo header:  
    `TT | MSSV | Họ và Tên | NTNS | 1..36 | submitted_at | quiz_id`
- Nếu gặp lỗi quyền, hãy **Share** file cho service account trong secrets, quyền **Editor**.
        """
    )

st.markdown("---")
st.markdown("© Bản quyền thuộc về TS. Đào Hồng Nam - Đại học Y Dược Thành phố Hồ Chí Minh")
