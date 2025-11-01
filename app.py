# -*- coding: utf-8 -*-
import time
import re
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# =========================[ UI HELPERS ]=========================
def render_banner():
    st.markdown(
        "<h2 style='margin:0'>🎓 Hệ thống</h2><div style='color:#777'>Demo tối giản</div>",
        unsafe_allow_html=True,
    )
    st.divider()

def teacher_panel():
    st.subheader("Bảng điều khiển Giảng viên")
    st.info("Trang mẫu (placeholder). Bạn có thể giữ nguyên hoặc thay nội dung GV của bạn vào đây.")

def student_panel():
    st.subheader("Trang Sinh viên (mẫu)")
    with st.form("sv_login", clear_on_submit=False):
        m = st.text_input("MSSV (mẫu – không bắt buộc cho trang Xem điểm)", max_chars=9)
        p = st.text_input("Mật khẩu (mẫu)", type="password")
        ok = st.form_submit_button("Đăng nhập (mẫu)")
    if ok and m.strip().isdigit() and len(m.strip()) == 9:
        st.session_state["sv_logged_in"] = True
        st.session_state["sv_mssv"] = m.strip()
        st.success("Đăng nhập SV (mẫu) thành công.")
    st.caption("Trang Xem điểm KHÔNG phụ thuộc vào đăng nhập mẫu này; bạn có thể bỏ qua.")

# =========================[ GOOGLE SHEETS ]=========================
_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

@st.cache_resource(show_spinner=False)
def _xd_get_ws():
    """Kết nối Google Sheet (cache connection)."""
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=_SCOPES
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["SHEET_ID"])
    ws = sh.worksheet(st.secrets.get("SHEET_TAB", "KQGK"))
    return ws

@st.cache_data(ttl=120, show_spinner=False)
def _xd_headers(_ws):
    return _ws.row_values(1)

def _xd_col_index_by_header(headers, target_names):
    """Trả về chỉ số cột (1-based) trùng 1 trong các tên."""
    lowers = [h.strip().lower() for h in headers]
    for name in target_names:
        nl = name.strip().lower()
        if nl in lowers:
            return lowers.index(nl) + 1
    return None

@st.cache_data(ttl=120, show_spinner=False)
def _xd_find_row_by_mssv(_ws, mssv: str, headers):
    """Tìm đúng hàng theo MSSV (9 chữ số) trong cột 'Mssv'."""
    mssv = (mssv or "").strip()
    if not (mssv.isdigit() and len(mssv) == 9):
        return None
    mssv_col = _xd_col_index_by_header(headers, ["Mssv", "MSSV", "Mã số", "Mã SV"])
    if not mssv_col:
        return None
    try:
        cell = _ws.find(mssv, in_column=mssv_col)
        return cell.row if cell else None
    except gspread.exceptions.CellNotFound:
        return None

def _xd_visible_fields(headers):
    """Các cột cho phép hiển thị theo đúng thứ tự; thiếu thì tự bỏ qua."""
    order = ["Mssv", "Họ và Tên", "Ngày sinh", "Tổ", "Điểm"]
    fields = [h for h in order if h in headers]
    return fields if fields else headers

def _xd_col_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

# ---- ép định dạng ngày dd/mm/YYYY cho mọi kiểu giá trị (serial/chuỗi) ----
_ddmmyyyy_re = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$")
def _force_ddmmyyyy(val):
    """Chuẩn hóa ngày về dd/mm/YYYY (không phụ thuộc locale)."""
    # Serial Excel/Sheets
    if isinstance(val, (int, float)):
        f = float(val)
        if 20000 < f < 60000:  # khoảng ngày hợp lý ~ 1955..2064
            base = datetime(1899, 12, 30)
            try:
                d = base + timedelta(days=f)
                return d.strftime("%d/%m/%Y")
            except Exception:
                return val
    # Chuỗi
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return s
        s_std = s.replace(".", "/").replace("-", "/")
        # dạng d/m/Y hoặc m/d/Y -> luôn coi vế đầu là ngày
        m = _ddmmyyyy_re.match(s_std)
        if m:
            d_, m_, y_ = m.groups()
            return f"{int(d_):02d}/{int(m_):02d}/{y_}"
        # dạng Y/m/d
        try:
            d = datetime.strptime(s_std, "%Y/%m/%d")
            return d.strftime("%d/%m/%Y")
        except Exception:
            pass
    return val

def _xd_read_row_strict(_ws, row_idx: int, ncols: int):
    """
    Đọc đúng 'ncols' cột của hàng 'row_idx' bằng A1 range,
    giữ ô trống ở giữa (không cắt đuôi) và chuẩn hóa ngày về dd/mm/YYYY.
    """
    end_col = _xd_col_letter(ncols)
    rng = f"A{row_idx}:{end_col}{row_idx}"
    rows = _ws.get(rng, value_render_option="UNFORMATTED_VALUE")
    if rows and len(rows) > 0:
        row = rows[0]
        if len(row) < ncols:
            row += [""] * (ncols - len(row))
        # ép ngày cho các cột có tên liên quan
        return row[:ncols]
    return [""] * ncols

# =========================[ PAGE: XEM ĐIỂM ]=========================
def render_xem_diem_page():
    st.title("Xem điểm")

    # 0) Nút làm mới cache (tiện debug khi đổi code/secrets)
    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("Làm mới cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # 1) Mật khẩu tab (để rỗng trong secrets nếu không muốn hỏi)
    if "xd_logged_in" not in st.session_state:
        st.session_state["xd_logged_in"] = False
    if not st.session_state["xd_logged_in"]:
        with st.form("xd_login_form", clear_on_submit=False):
            pwd = st.text_input("Mật khẩu trang Xem điểm", type="password")
            ok = st.form_submit_button("Đăng nhập")
        if ok:
            secret_pwd = str(st.secrets.get("XEM_DIEM_PASSWORD", "")).strip()
            if secret_pwd == "" or pwd.strip() == secret_pwd:
                st.session_state["xd_logged_in"] = True
                st.success("Đã vào trang Xem điểm.")
            else:
                st.error("Sai mật khẩu.")
        if not st.session_state["xd_logged_in"]:
            return

    # 2) MSSV: nếu đã khóa trong phiên thì chỉ hiển thị read-only
    locked = st.session_state.get("xd_locked_mssv")
    if locked:
        st.text_input("MSSV của bạn (đã khóa trong phiên)", value=locked, disabled=True)
        mssv = locked
        xem = st.button("Xem lại", type="primary", use_container_width=True)
        if not xem:
            return
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            mssv = st.text_input(
                "Nhập MSSV (9 chữ số)",
                value=st.session_state.get("xd_last_mssv", ""),
                max_chars=9,
                placeholder="VD: 511256000",
            )
        with col2:
            st.write("")
            xem = st.button("Xem điểm", type="primary", use_container_width=True)
            if not xem:
                return

    # 3) Validate MSSV
    mssv = (mssv or "").strip()
    if not (mssv.isdigit() and len(mssv) == 9):
        st.warning("MSSV phải gồm đúng 9 chữ số.")
        return

    # 4) Truy xuất 1 hàng & hiển thị
    with st.spinner("Đang truy xuất..."):
        try:
            ws = _xd_get_ws()
            headers = _xd_headers(ws)
            row_idx = _xd_find_row_by_mssv(ws, mssv, headers)
            if not row_idx:
                st.error("Không tìm thấy MSSV trong danh sách.")
                return

            values = _xd_read_row_strict(ws, row_idx, len(headers))
            # map header -> value
            rec = {headers[i]: (values[i] if i < len(values) else "") for i in range(len(headers))}

            # ép riêng cột ngày (nếu có)
            for key in headers:
                if key.strip().lower() in {"ngày sinh", "ngay sinh"}:
                    rec[key] = _force_ddmmyyyy(rec.get(key, ""))

            cols = _xd_visible_fields(headers)
            df = pd.DataFrame([{k: rec.get(k, "") for k in cols}])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Hàng dữ liệu: {row_idx} • Cập nhật: {time.strftime('%H:%M:%S')}")

            # Khóa MSSV sau lần xem đầu
            if not locked:
                st.session_state["xd_locked_mssv"] = mssv
            st.session_state["xd_last_mssv"] = mssv

        except gspread.exceptions.APIError as e:
            st.error("Lỗi Google API. Vui lòng thử lại sau.")
            st.exception(e)
        except Exception as e:
            st.error("Đã xảy ra lỗi không mong muốn.")
            st.exception(e)

# =========================[ ROUTER ]=========================
st.set_page_config(page_title="App", layout="wide")

page = st.sidebar.radio(
    "Chọn trang",
    ["Sinh viên", "Giảng viên", "Xem điểm", "Hướng dẫn"],
    index=0,
)

if page == "Sinh viên":
    render_banner()
    student_panel()

elif page == "Giảng viên":
    render_banner()
    teacher_panel()

elif page == "Xem điểm":
    render_banner()
    render_xem_diem_page()

else:
    render_banner()
    st.title("Hướng dẫn nhanh")
    st.write("- Vào **Xem điểm** để tra cứu theo MSSV (9 chữ số).")
    st.write("- Dùng nút **Làm mới cache** nếu vừa sửa mã hoặc secrets.")
