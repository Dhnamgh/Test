import os
import time
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# ------------------ App config ------------------
st.set_page_config(page_title="Trắc nghiệm Likert 36", layout="wide")

QUIZ_ID = st.secrets.get("QUIZ_ID", "PSY36")
TIME_LIMIT_MIN = int(st.secrets.get("TIME_LIMIT_MIN", 20))

QUESTIONS_SPREADSHEET_ID = st.secrets["QUESTIONS_SPREADSHEET_ID"]
QUESTIONS_SHEET_NAME = st.secrets.get("QUESTIONS_SHEET_NAME", "PSY36_Questions")

RESPONSES_SPREADSHEET_ID = st.secrets["RESPONSES_SPREADSHEET_ID"]
RESPONSES_SHEET_NAME = st.secrets.get("RESPONSES_SHEET_NAME", "Sheet1")

TEACHER_USER = st.secrets.get("TEACHER_USER", "teacher")
TEACHER_PASS = st.secrets.get("TEACHER_PASS", "teacher123")

# ------------------ Google Sheets helpers ------------------
def get_gspread_client():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds_info = st.secrets["gcp_service_account"]
    credentials = Credentials.from_service_account_info(creds_info, scopes=scopes)
    return gspread.authorize(credentials)

@st.cache_data(ttl=300)
def load_questions_df():
    gc = get_gspread_client()
    sh = gc.open_by_key(QUESTIONS_SPREADSHEET_ID)
    ws = sh.worksheet(QUESTIONS_SHEET_NAME)
    df = pd.DataFrame(ws.get_all_records())
    # Chuẩn hoá cột
    needed = ["quiz_id", "q_index", "question", "left_label", "right_label", "reverse"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""
    df = df[df["quiz_id"] == QUIZ_ID].copy()
    df["q_index"] = df["q_index"].astype(int)
    df = df.sort_values("q_index")
    # Kiểm tra đủ 36 câu
    miss = [i for i in range(1,37) if i not in df["q_index"].tolist()]
    if miss:
        raise ValueError(f"Thiếu q_index: {miss}")
    return df.reset_index(drop=True)

def get_responses_ws():
    gc = get_gspread_client()
    sh = gc.open_by_key(RESPONSES_SPREADSHEET_ID)
    try:
        ws = sh.worksheet(RESPONSES_SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(RESPONSES_SHEET_NAME, rows=1000, cols=50)
    # Tạo header nếu trống
    vals = ws.get_all_values()
    if not vals:
        header = ["TT", "MSSV", "Họ và Tên", "NTNS"] + [str(i) for i in range(1,37)] + ["submitted_at", "quiz_id"]
        ws.append_row(header, value_input_option="USER_ENTERED")
    return ws

def find_or_create_row_by_mssv(ws, mssv, fullname=""):
    vals = ws.get_all_values()
    header = vals[0]
    col_mssv = header.index("MSSV") + 1

    for r in range(2, len(vals)+1):
        if ws.cell(r, col_mssv).value == mssv:
            return r
    # chưa có → thêm
    tt = len(vals)  # số thứ tự tiếp theo
    row = [tt, mssv, fullname, ""] + [""]*36 + ["", QUIZ_ID]
    ws.append_row(row, value_input_option="USER_ENTERED")
    return len(vals) + 1

def write_answers_row(ws, row_idx, answers_dict, fullname=None):
    # ghi 36 cột 1..36
    values = [answers_dict.get(i, "") for i in range(1,37)]
    start_col = 5; end_col = 40
    r1c1 = gspread.utils.rowcol_to_a1(row_idx, start_col)
    r2c2 = gspread.utils.rowcol_to_a1(row_idx, end_col)
    ws.update(f"{r1c1}:{r2c2}", [values], value_input_option="USER_ENTERED")

    # cập nhật submitted_at, quiz_id, fullname (nếu muốn)
    header = ws.row_values(1)
    col_fullname = header.index("Họ và Tên") + 1 if "Họ và Tên" in header else None
    col_sub_at = header.index("submitted_at") + 1 if "submitted_at" in header else None
    col_quiz = header.index("quiz_id") + 1 if "quiz_id" in header else None

    updates = []
    if col_fullname and fullname:
        updates.append((col_fullname, fullname))
    if col_sub_at:
        updates.append((col_sub_at, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    if col_quiz:
        updates.append((col_quiz, QUIZ_ID))

    if updates:
        rng = []
        vals = []
        for (c, v) in updates:
            rng.append(gspread.utils.rowcol_to_a1(row_idx, c))
            vals.append([v])
        ws.update(rng, vals, value_input_option="USER_ENTERED")

# ------------------ UI parts ------------------
def render_timer(end_ts, total_seconds):
    remaining = max(0, int(end_ts - time.time()))
    mins, secs = divmod(remaining, 60)
    used = total_seconds - remaining
    pct = min(max(used / total_seconds if total_seconds else 1.0, 0.0), 1.0)

    c1, c2 = st.columns([1,3])
    with c1:
        st.metric("Thời gian còn lại", f"{mins:02d}:{secs:02d}")
    with c2:
        st.progress(pct, text="Tiến độ thời gian")

    # auto refresh 1s
    st.markdown("<script>setTimeout(()=>window.parent.location.reload(),1000);</script>",
                unsafe_allow_html=True)
    return remaining

def nav_grid(ans_dict, cur_q):
    cols = st.columns(12)
    for i in range(1,37):
        btn_col = cols[(i-1)%12]
        label = f"{i}"
        style = "secondary"
        if i == cur_q:
            style = "primary"
        elif ans_dict.get(i):
            style = "success"
        with btn_col:
            if st.button(label, key=f"jump_{i}"):
                st.session_state.cur_q = i

def question_block(row, current_value):
    left = row["left_label"] if str(row["left_label"]).strip() else "Hoàn toàn không đồng ý"
    right = row["right_label"] if str(row["right_label"]).strip() else "Hoàn toàn đồng ý"
    st.write(row["question"])
    st.caption(f"_{left} ⟶ {right}_")
    return st.slider("Chọn mức (1–5)", 1, 5, int(current_value), key=f"slider_q{row['q_index']}")

def likert36_exam():
    # lấy câu hỏi
    qdf = load_questions_df()

    # login SV
    st.subheader("Đăng nhập Sinh viên")
    mssv = st.text_input("MSSV")
    fullname = st.text_input("Họ và Tên")

    if st.button("Bắt đầu làm bài"):
        if not mssv or not fullname:
            st.error("Vui lòng nhập MSSV và Họ tên.")
        else:
            st.session_state.started = True
            st.session_state.mssv = mssv.strip()
            st.session_state.fullname = fullname.strip()
            total_seconds = TIME_LIMIT_MIN * 60
            st.session_state.end_ts = time.time() + total_seconds
            st.session_state.total_seconds = total_seconds
            st.session_state.cur_q = 1
            st.session_state.ans = {}     # {q_index: 1..5}
            st.session_state.auto_submitted = False

    if not st.session_state.get("started"):
        st.info(f"Bài {QUIZ_ID}: 36 câu Likert. Thời gian **{TIME_LIMIT_MIN} phút**.")
        return

    # header & đồng hồ
    st.success(f"Bài: {QUIZ_ID} — MSSV: {st.session_state.mssv} — Họ tên: {st.session_state.fullname}")
    remaining = render_timer(st.session_state.end_ts, st.session_state.total_seconds)

    # thanh tiến độ & grid
    answered = sum(1 for v in st.session_state.ans.values() if v is not None)
    st.progress(answered/36)
    st.caption(f"Đã trả lời: {answered}/36")
    nav_grid(st.session_state.ans, st.session_state.cur_q)
    st.divider()

    # câu hiện tại
    i = st.session_state.cur_q
    row = qdf[qdf["q_index"] == i].iloc[0]
    st.markdown(f"### Câu {i}/36")
    cur_val = st.session_state.ans.get(i, 3)
    new_val = question_block(row, cur_val)
    st.session_state.ans[i] = int(new_val)

    # điều hướng
    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("⬅️ Câu trước", use_container_width=True, disabled=(i==1)):
            st.session_state.cur_q = max(1, i-1)
    with next_col:
        if st.button("Câu sau ➡️", use_container_width=True, disabled=(i==36)):
            st.session_state.cur_q = min(36, i+1)

    st.divider()
    st.caption("Bạn có thể đi tới/lui để kiểm tra trước khi nộp.")

    def do_submit(auto=False):
        try:
            ws = get_responses_ws()
            row_idx = find_or_create_row_by_mssv(ws, st.session_state.mssv, st.session_state.fullname)
            write_answers_row(ws, row_idx, st.session_state.ans, fullname=st.session_state.fullname)
            if auto:
                st.warning("Hết giờ — hệ thống đã tự động nộp.")
            else:
                st.success("Đã nộp bài thành công.")
        except Exception as e:
            st.error(f"Lỗi ghi Google Sheet: {e}")
            return
        # reset
        st.session_state.started = False
        st.session_state.ans = {}
        st.session_state.cur_q = 1
        st.session_state.auto_submitted = False

    # auto-submit khi hết giờ
    if remaining == 0 and not st.session_state.get("auto_submitted", False):
        st.session_state.auto_submitted = True
        do_submit(auto=True)
        return

    # nút nộp
    all_answered = len(st.session_state.ans) == 36 and all(st.session_state.ans.get(k) for k in range(1,37))
    st.button("✅ Nộp bài", type="primary", on_click=lambda: do_submit(auto=False), disabled=not all_answered)

# ------------------ Teacher page ------------------
def teacher_page():
    st.subheader("Đăng nhập Giảng viên")
    u = st.text_input("Tài khoản")
    p = st.text_input("Mật khẩu", type="password")
    if st.button("Đăng nhập"):
        if u == TEACHER_USER and p == TEACHER_PASS:
            st.session_state.teacher = True
        else:
            st.error("Sai tài khoản/mật khẩu.")

    if not st.session_state.get("teacher"):
        st.info("Nhập tài khoản GV.")
        return

    st.success("Đăng nhập GV thành công.")
    st.markdown("### Ngân hàng câu hỏi")
    try:
        df = load_questions_df()
        st.dataframe(df)
        st.caption("Nguồn: Google Sheet (QuestionBank)")
    except Exception as e:
        st.error(f"Không tải được câu hỏi: {e}")

    st.markdown("### Cấu hình")
    st.write(f"- QUIZ_ID: `{QUIZ_ID}`")
    st.write(f"- TIME_LIMIT_MIN: `{TIME_LIMIT_MIN}` phút")
    st.write(f"- Questions Sheet: `{QUESTIONS_SHEET_NAME}`")
    st.write(f"- Responses Sheet: `{RESPONSES_SHEET_NAME}`")

# ------------------ App main ------------------
st.title("Trắc nghiệm Likert 36 (Google Sheets)")

tab_gv, tab_sv, tab_hdsd = st.tabs(["Giảng viên", "Sinh viên", "Hướng dẫn"])
with tab_gv:
    teacher_page()
with tab_sv:
    likert36_exam()
with tab_hdsd:
    st.markdown("""
**Cách dùng nhanh**
1) Thêm 36 câu vào Google Sheet *QuestionBank* (tab `PSY36_Questions`) theo cột:  
`quiz_id | q_index | question | left_label | right_label | reverse`
2) Tạo Google Sheet *Responses* với header:  
`TT | MSSV | Họ và Tên | NTNS | 1 | 2 | ... | 36 | submitted_at | quiz_id`
3) Dán Secrets (service account + IDs) vào Streamlit Cloud.  
4) SV vào tab **Sinh viên** → nhập MSSV + Họ tên → làm bài → nộp (hoặc tự nộp khi hết giờ).
""")
