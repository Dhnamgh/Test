import streamlit as st
import pandas as pd
import numpy as np
import os
import hashlib
from datetime import datetime

DATA_DIR = "data"
QUESTIONS_CSV = os.path.join(DATA_DIR, "questions.csv")
SUBMISSIONS_CSV = os.path.join(DATA_DIR, "submissions.csv")

# ---------- Utilities ----------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(QUESTIONS_CSV):
        pd.DataFrame(columns=[
            "question_id","quiz_id","qtype","question",
            "option_a","option_b","option_c","option_d",
            "answer","likert_left_label","likert_right_label"
        ]).to_csv(QUESTIONS_CSV, index=False, encoding="utf-8")
    if not os.path.exists(SUBMISSIONS_CSV):
        pd.DataFrame(columns=[
            "timestamp","quiz_id","mssv","fullname","score_mcq","total_mcq",
            "likert_sum","likert_count","answers_json"
        ]).to_csv(SUBMISSIONS_CSV, index=False, encoding="utf-8")

def load_questions():
    ensure_data_dir()
    try:
        df = pd.read_csv(QUESTIONS_CSV, dtype=str).fillna("")
    except Exception:
        df = pd.DataFrame()
    # Normalize columns
    required = {
        "question_id","quiz_id","qtype","question",
        "option_a","option_b","option_c","option_d",
        "answer","likert_left_label","likert_right_label"
    }
    for c in required:
        if c not in df.columns:
            df[c] = ""
    # auto-generate question_id if missing
    if "question_id" in df.columns:
        mask = df["question_id"].astype(str).str.strip() == ""
        df.loc[mask, "question_id"] = [f"q_{i+1}" for i in range(mask.sum())]
    return df[sorted(required)]

def save_questions(df):
    df.to_csv(QUESTIONS_CSV, index=False, encoding="utf-8")

def load_submissions():
    ensure_data_dir()
    try:
        return pd.read_csv(SUBMISSIONS_CSV, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame(columns=[
            "timestamp","quiz_id","mssv","fullname","score_mcq","total_mcq",
            "likert_sum","likert_count","answers_json"
        ])

def append_submission(row_dict):
    df = load_submissions()
    df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    df.to_csv(SUBMISSIONS_CSV, index=False, encoding="utf-8")

def stable_seed(*parts) -> int:
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16)  # large but within Python int

def shuffle_with_seed(items, seed):
    rng = np.random.default_rng(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx], idx  # return new order + index mapping

def list_quiz_ids(df):
    vals = sorted([q for q in df["quiz_id"].unique().tolist() if q.strip() != ""])
    return vals

# ---------- UI Helpers ----------
def mcq_block(qrow, seed):
    # Build option list and shuffle
    options = [("A", qrow["option_a"]), ("B", qrow["option_b"]),
               ("C", qrow["option_c"]), ("D", qrow["option_d"])]
    # filter out empty options to be safe
    options = [(k,v) for k,v in options if str(v).strip() != ""]
    letters = [k for k,_ in options]
    opt_texts = [v for _,v in options]
    shuffled_opts, idx_map = shuffle_with_seed(opt_texts, seed)
    # map back to original letters
    shuffled_letters = [letters[i] for i in idx_map]
    # radio
    choice = st.radio(
        "Chọn đáp án:",
        options=[f"{i+1}. {opt}" for i,opt in enumerate(shuffled_opts)],
        index=None,
        key=f"mcq_{qrow['question_id']}"
    )
    # decode back to letter
    picked_letter = None
    if choice:
        # "1. text ..." -> index 0
        try:
            ix = int(choice.split(".")[0]) - 1
            picked_letter = shuffled_letters[ix]
        except Exception:
            picked_letter = None
    return picked_letter

def likert_block(qrow):
    left = qrow.get("likert_left_label", "") or "Hoàn toàn không đồng ý"
    right = qrow.get("likert_right_label", "") or "Hoàn toàn đồng ý"
    st.write(f"_{left} ⟶ {right}_")
    val = st.slider("Mức độ (1–5)", min_value=1, max_value=5, value=3, step=1,
                    key=f"likert_{qrow['question_id']}")
    return val

# ---------- Pages ----------
def page_teacher():
    st.subheader("Đăng nhập Giảng viên")
    # Demo auth — đổi sau
    u = st.text_input("Tài khoản", value="", type="default")
    p = st.text_input("Mật khẩu", value="", type="password")
    if st.button("Đăng nhập"):
        if u == "teacher" and p == "teacher123":
            st.session_state["teacher_logged_in"] = True
            st.success("Đăng nhập thành công.")
        else:
            st.error("Sai tài khoản hoặc mật khẩu.")

    if not st.session_state.get("teacher_logged_in"):
        st.info("Nhập tài khoản/mật khẩu demo: teacher / teacher123 (đổi sau).")
        return

    st.markdown("### Quản lý ngân hàng câu hỏi")
    dfq = load_questions()

    # Upload CSV
    up = st.file_uploader("Tải lên CSV câu hỏi", type=["csv"], accept_multiple_files=False)
    if up is not None:
        try:
            newdf = pd.read_csv(up, dtype=str).fillna("")
            # normalize cols
            needed = [
                "question_id","quiz_id","qtype","question",
                "option_a","option_b","option_c","option_d",
                "answer","likert_left_label","likert_right_label"
            ]
            for c in needed:
                if c not in newdf.columns:
                    newdf[c] = ""
            # auto gen question_id if empty
            mask = newdf["question_id"].astype(str).str.strip() == ""
            newdf.loc[mask, "question_id"] = [f"new_{i+1}" for i in range(mask.sum())]
            # Append (MVP) — có thể chuyển sang hợp nhất/ghi đè theo question_id
            dfq = pd.concat([dfq, newdf[needed]], ignore_index=True)
            save_questions(dfq)
            st.success(f"Tải {len(newdf)} câu hỏi thành công.")
        except Exception as e:
            st.error(f"Lỗi đọc CSV: {e}")

    # Hiển thị preview & danh sách quiz
    st.write("**Danh sách quiz_id khả dụng:**", ", ".join(list_quiz_ids(dfq)) or "(chưa có)")
    st.dataframe(dfq.head(100))

    st.markdown("### Kết quả nộp bài")
    subs = load_submissions()
    if len(subs):
        st.dataframe(subs)
        st.download_button(
            "Tải kết quả CSV",
            data=subs.to_csv(index=False).encode("utf-8"),
            file_name=f"submissions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Chưa có bài nộp.")

def page_student():
    st.subheader("Đăng nhập Sinh viên")
    mssv = st.text_input("MSSV")
    fullname = st.text_input("Họ và tên")
    dfq = load_questions()
    quiz_ids = list_quiz_ids(dfq)
    quiz_id = st.selectbox("Chọn bài (quiz_id)", options=quiz_ids if quiz_ids else ["(chưa có)"])

    if st.button("Bắt đầu làm bài"):
        if not mssv or not fullname or not quiz_id or quiz_id == "(chưa có)":
            st.error("Vui lòng nhập MSSV, Họ tên và chọn bài.")
        else:
            st.session_state["exam_started"] = True
            st.session_state["exam_ctx"] = {
                "mssv": mssv.strip(),
                "fullname": fullname.strip(),
                "quiz_id": quiz_id.strip()
            }

    if not st.session_state.get("exam_started"):
        st.info("Nhập thông tin và chọn bài để bắt đầu.")
        return

    # Render exam
    ctx = st.session_state["exam_ctx"]
    st.success(f"Đang làm bài: {ctx['quiz_id']} — {ctx['mssv']} - {ctx['fullname']}")

    # Filter questions of this quiz
    quiz_df = dfq[dfq["quiz_id"] == ctx["quiz_id"]].reset_index(drop=True)
    if quiz_df.empty:
        st.error("Bài này chưa có câu hỏi.")
        return

    # Stable shuffle of questions
    q_seed = stable_seed(ctx["mssv"], ctx["quiz_id"])
    order = list(range(len(quiz_df)))
    rng = np.random.default_rng(q_seed)
    rng.shuffle(order)
    quiz_df = quiz_df.iloc[order].reset_index(drop=True)

    st.markdown("---")
    st.markdown("### Câu hỏi")

    answers = {}
    mcq_correct = 0
    mcq_total = 0
    likert_sum = 0
    likert_count = 0

    for i, row in quiz_df.iterrows():
        st.markdown(f"**Câu {i+1}:** {row['question']}")
        if row["qtype"].upper() == "MCQ":
            mcq_total += 1
            pick = mcq_block(row, seed=stable_seed(ctx["mssv"], ctx["quiz_id"], row["question_id"]))
            answers[row["question_id"]] = {"type":"MCQ","picked": pick}
            if pick is not None and pick == str(row["answer"]).strip().upper():
                mcq_correct += 1

        elif row["qtype"].upper() == "LIKERT":
            val = likert_block(row)
            answers[row["question_id"]] = {"type":"LIKERT","value": int(val)}
            likert_sum += int(val)
            likert_count += 1
        else:
            st.warning("Loại câu hỏi không hợp lệ (phải là MCQ hoặc LIKERT).")

        st.markdown("---")

    if st.button("Nộp bài"):
        # Lưu kết quả
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp": ts,
            "quiz_id": ctx["quiz_id"],
            "mssv": ctx["mssv"],
            "fullname": ctx["fullname"],
            "score_mcq": str(mcq_correct),
            "total_mcq": str(mcq_total),
            "likert_sum": str(likert_sum),
            "likert_count": str(likert_count),
            "answers_json": pd.Series([answers]).to_json(orient="records")
        }
        append_submission(row)
        st.success(f"Đã nộp bài. Điểm MCQ: {mcq_correct}/{mcq_total}. Likert: tổng {likert_sum} trên {likert_count} mục.")
        # reset
        st.session_state["exam_started"] = False
        st.session_state["exam_ctx"] = None

# ---------- App ----------
st.set_page_config(page_title="App Trắc nghiệm", layout="wide")

st.title("App Trắc nghiệm (MCQ + Likert)")

tab_gv, tab_sv, tab_hdsd = st.tabs(["Đăng nhập GV", "Đăng nhập SV", "Hướng dẫn"])

with tab_gv:
    page_teacher()

with tab_sv:
    page_student()

with tab_hdsd:
    st.markdown(""")
### Hướng dẫn CSV
- Cột bắt buộc:
  - `quiz_id` | `qtype` (MCQ/LIKERT) | `question`
- MCQ thêm:
  - `option_a`..`option_d` và `answer` (A/B/C/D)
- Likert thêm:
  - `likert_left_label` và `likert_right_label` (tuỳ chọn, có mặc định)
- `question_id` có thể để trống, hệ thống sẽ tự sinh.

**Ví dụ 3 dòng:**



st.markdown("---")
st.markdown("© Bản quyền thuộc về TS. Đào Hồng Nam - Đại học Y Dược Thành phố Hồ Chí Minh")
