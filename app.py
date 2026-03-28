import os
import re
import json
import time
import datetime
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse
from contextlib import contextmanager

import urllib3
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 可选依赖
# ==========================================
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    from filelock import FileLock
except ImportError:
    FileLock = None


@contextmanager
def noop_lock(*args, **kwargs):
    yield


def get_lock(path: str):
    if FileLock is None:
        return noop_lock()
    return FileLock(path, timeout=10)


# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="DataHub AI 科研引擎", page_icon="🌌", layout="wide")

# ==========================================
# 全局配置
# ==========================================
DATA_FILE = "datasets_metadata.csv"
USAGE_FILE = "global_usage.txt"
BACKUP_DIR = "backups"
FREE_TRIES_LIMIT = 100

DATA_LOCK = DATA_FILE + ".lock"
USAGE_LOCK = USAGE_FILE + ".lock"

MODEL_NAME_MAP = {
    "ModelScope": "deepseek-ai/DeepSeek-V3",
    "Gemini": "gemini-2.5-flash",
    "DeepSeek": "deepseek-chat",
    "OpenAI": "gpt-4o-mini"
}

COLUMN_MAP = {
    "name": "数据集名称",
    "description": "简介",
    "macro_field": "主领域",
    "sub_field": "子领域",
    "use_case": "适用场景",
    "institution": "核心机构",
    "paper_citation": "关联论文/引用格式",
    "paper_link": "论文链接",
    "db_link": "数据库官方链接",
    "format": "数据格式",
    "detailed_license": "详细使用协议",
    "license_type": "协议类型",
    "detailed_size": "详细数据量级",
    "size_type": "量级(标准化)",
    "participant_count": "参与者/样本数量",
    "language": "语种",
    "release_year": "发布年份",
    "update_year": "最近更新年份",
}

ALL_COLUMNS = list(COLUMN_MAP.values()) + ["系统录入来源", "检索模型"]

VALID_LICENSE_TYPES = {"免费开源", "注册免费", "付费申请", "内部闭源"}
VALID_SIZE_TYPES = {"KB", "MB", "GB", "TB", "PB"}

VALID_MACRO_FIELDS = {
    "自然科学", "工程与技术", "生命科学与医学", 
    "农业与环境科学", "社会科学", "人文艺术", 
    "人工智能与计算机", "综合与交叉学科"
}

for p in [BACKUP_DIR]:
    if not os.path.exists(p):
        os.makedirs(p)


# ==========================================
# 调试日志
# ==========================================
def reset_debug_state():
    st.session_state["debug_logs"] = []
    st.session_state["last_raw_response"] = ""
    st.session_state["last_parsed_payload"] = {}
    st.session_state["last_update_logs"] = []
    st.session_state["last_prompt_preview"] = ""


def log_debug(stage: str, message: str, payload: Optional[Any] = None):
    if "debug_logs" not in st.session_state:
        st.session_state["debug_logs"] = []
    item = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "stage": stage,
        "message": message,
    }
    if payload is not None:
        item["payload"] = payload
    st.session_state["debug_logs"].append(item)


# ==========================================
# 基础函数
# ==========================================
def sanitize_error(msg: str) -> str:
    msg = str(msg)
    patterns = [
        r"sk-[A-Za-z0-9\-_]+",
        r"ms-[A-Za-z0-9\-_]+",
        r"AIza[0-9A-Za-z\-_]+",
    ]
    for p in patterns:
        msg = re.sub(p, "***", msg)
    msg = re.sub(r'(api[ -]?key["\']?\s*[:=]\s*["\']?)[^"\',\s]+', r"\1***", msg, flags=re.IGNORECASE)
    return msg


def get_api_key(provider: str) -> str:
    try:
        key = st.secrets.get(provider.upper() + "_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv(provider.upper() + "_KEY", "")


def is_valid_key_format(provider: str, key: str) -> bool:
    if not key:
        return False
    if provider == "DeepSeek":
        return key.startswith("sk-")
    if provider == "ModelScope":
        return key.startswith("ms-")
    if provider == "Gemini":
        return key.startswith("AIza")
    if provider == "OpenAI":
        return key.startswith("sk-")
    return True


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = str(name).strip().lower()
    name = re.sub(r"[\s\-_]+", " ", name)
    name = re.sub(r"[^\w\s\u4e00-\u9fff]", "", name)
    return name.strip()


def is_empty_val(val) -> bool:
    if pd.isna(val):
        return True
    if str(val).strip().lower() in {"", "none", "nan", "null", "未知"}:
        return True
    return False


def init_data_file():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=ALL_COLUMNS)
        save_data(df)


def load_data() -> pd.DataFrame:
    init_data_file()
    with get_lock(DATA_LOCK):
        try:
            df = pd.read_csv(DATA_FILE)
        except Exception:
            df = pd.DataFrame(columns=ALL_COLUMNS)
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df.reindex(columns=ALL_COLUMNS)


def save_data(df: pd.DataFrame):
    df = df.reindex(columns=ALL_COLUMNS)
    with get_lock(DATA_LOCK):
        tmp_file = DATA_FILE + ".tmp"
        df.to_csv(tmp_file, index=False, encoding="utf-8-sig")
        os.replace(tmp_file, DATA_FILE)


def get_global_usage() -> int:
    with get_lock(USAGE_LOCK):
        if not os.path.exists(USAGE_FILE):
            with open(USAGE_FILE, "w", encoding="utf-8") as f:
                f.write("0")
        with open(USAGE_FILE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            return int(raw) if raw.isdigit() else 0


def can_use_free_channel() -> bool:
    return get_global_usage() < FREE_TRIES_LIMIT


def increment_global_usage() -> int:
    with get_lock(USAGE_LOCK):
        current = 0
        if os.path.exists(USAGE_FILE):
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                raw = f.read().strip()
                current = int(raw) if raw.isdigit() else 0
        count = current + 1
        tmp_path = USAGE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(str(count))
        os.replace(tmp_path, USAGE_FILE)
        return count


def exhaust_global_usage():
    with get_lock(USAGE_LOCK):
        tmp_path = USAGE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(str(FREE_TRIES_LIMIT))
        os.replace(tmp_path, USAGE_FILE)


def create_requests_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@st.cache_data(ttl=3600)
def fetch_web_content(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        session = create_requests_session()
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        title = ""
        domain = urlparse(url).netloc
        if BeautifulSoup is not None:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "footer", "nav"]):
                tag.extract()

            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            main_block = None
            for selector in ["main", "article", "[role='main']"]:
                main_block = soup.select_one(selector)
                if main_block:
                    break
            if main_block:
                content = main_block.get_text(separator=" ", strip=True)
            else:
                content = soup.get_text(separator=" ", strip=True)
        else:
            content = re.sub(r"<[^>]+>", " ", response.text)

        content = re.sub(r"\s+", " ", content).strip()
        return f"网页标题: {title}\n来源域名: {domain}\n网页内容: {content[:12000]}"
    except Exception as e:
        return f"Error: {sanitize_error(str(e))}"


def classify_api_error(error_msg: str) -> str:
    msg = str(error_msg).lower()
    quota_markers = [
        "insufficient quota", "quota exceeded", "billing", "credit",
        "余额不足", "配额不足", "额度已用尽",
    ]
    if any(k in msg for k in quota_markers):
        return "quota_exhausted"

    rate_markers = [
        "429", "rate limit", "resource_exhausted", "too many requests",
    ]
    if any(k in msg for k in rate_markers):
        return "rate_limited"

    return "other"


def retry_with_backoff(func, *args, max_retries=3, initial_wait=2, **kwargs):
    wait = initial_wait
    last_error = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_msg = sanitize_error(str(e))
            kind = classify_api_error(error_msg)
            log_debug("retry", f"第 {attempt + 1} 次失败，类型={kind}", error_msg)
            if kind in {"rate_limited", "quota_exhausted"}:
                if attempt == max_retries - 1:
                    raise e
                match = re.search(r"retry in ([\d\.]+)s", error_msg.lower())
                if match:
                    wait = float(match.group(1)) + 1.0
                time.sleep(wait)
                wait *= 2
                continue
            raise e
    if last_error:
        raise last_error
    raise Exception("Max retries exceeded")


# ==========================================
# JSON 处理
# ==========================================
def looks_truncated_json(text: str) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False

    open_braces = stripped.count("{")
    close_braces = stripped.count("}")
    open_brackets = stripped.count("[")
    close_brackets = stripped.count("]")

    if open_braces > close_braces or open_brackets > close_brackets:
        return True
    return False


def extract_json_candidates(text: str) -> List[dict]:
    if not isinstance(text, str):
        return []

    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text).strip()

    decoder = json.JSONDecoder()
    candidates = []

    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    candidates.append(obj)
            except json.JSONDecodeError:
                continue

    return candidates


def extract_best_json_payload(text: str) -> dict:
    candidates = extract_json_candidates(text)
    for obj in candidates:
        if isinstance(obj, dict) and "datasets" in obj and isinstance(obj["datasets"], list):
            return obj

    raise json.JSONDecodeError("No valid payload with datasets found", text, 0)


def normalize_scalar_value(key: str, val):
    if isinstance(val, str):
        val = val.strip()
        if val.lower() in {"", "none", "null", "nan", "未知"}:
            return None

    if key in {"sub_field", "institution", "format", "language"}:
        if isinstance(val, str):
            val = re.sub(r'[，、/；;\|]+', ',', val)
            val = re.sub(r',\s*,', ',', val)
            val = ', '.join([v.strip() for v in val.split(',') if v.strip()])
            return val if val else None

    if key in {"release_year", "update_year"}:
        if val is None:
            return None
        m = re.search(r"(19|20)\d{2}", str(val))
        return m.group(0) if m else None

    if key == "license_type":
        return val if val in VALID_LICENSE_TYPES else None

    if key == "size_type":
        return val if val in VALID_SIZE_TYPES else None

    if key == "macro_field":
        mapping = {
            "医疗健康": "生命科学与医学",
            "基础科学": "自然科学",
            "金融经济": "社会科学",
            "工业制造": "工程与技术",
            "综合地理": "综合与交叉学科",
            "其他": "综合与交叉学科"
        }
        val = mapping.get(val, val)
        return val if val in VALID_MACRO_FIELDS else None

    return val


def normalize_record(item: dict) -> dict:
    normalized = {}
    for key in COLUMN_MAP.keys():
        normalized[key] = normalize_scalar_value(key, item.get(key))
    return normalized


def validate_and_normalize_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {"datasets": []}

    datasets = payload.get("datasets", [])
    if not isinstance(datasets, list):
        datasets = []

    clean_datasets = []
    for item in datasets:
        if isinstance(item, dict):
            clean_datasets.append(normalize_record(item))

    return {"datasets": clean_datasets}


# ==========================================
# Provider 检查与调用
# ==========================================
def ensure_provider_dependency(provider: str):
    if provider in {"DeepSeek", "OpenAI", "ModelScope"} and OpenAI is None:
        raise ImportError("缺少 openai 包。")
    if provider == "Gemini" and genai is None:
        raise ImportError("缺少 google-genai 包。")


def call_ai(provider: str, api_key: str, prompt: str, temperature: float = 0.2) -> str:
    ensure_provider_dependency(provider)
    log_debug("call_ai", f"调用 {provider}", {"model": MODEL_NAME_MAP[provider]})

    if provider == "Gemini":
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL_NAME_MAP[provider],
            contents=prompt
        )
        text = response.text or ""
        if not text.strip():
            raise RuntimeError("返回空内容")
        return text

    if provider == "ModelScope":
        client = OpenAI(api_key=api_key, base_url="https://api-inference.modelscope.cn/v1", timeout=60, max_retries=0)
        response = client.chat.completions.create(
            model=MODEL_NAME_MAP[provider],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2200,
        )
        return response.choices[0].message.content or ""

    if provider == "OpenAI":
        client = OpenAI(api_key=api_key, timeout=60, max_retries=0)
        response = client.chat.completions.create(
            model=MODEL_NAME_MAP[provider],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2200,
        )
        return response.choices[0].message.content or ""

    if provider == "DeepSeek":
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=60, max_retries=0)
        response = client.chat.completions.create(
            model=MODEL_NAME_MAP[provider],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2200,
        )
        return response.choices[0].message.content or ""

    raise ValueError(f"未知引擎: {provider}")


def repair_json_response(provider: str, api_key: str, raw_text: str) -> dict:
    repair_prompt = f"请把以下内容修复为严格JSON。仅输出JSON，包含datasets数组，缺失填null：\n\n{raw_text}"
    result_text = call_ai(provider, api_key, repair_prompt, temperature=0.0)
    return extract_best_json_payload(result_text)


# ==========================================
# prompt 构造
# ==========================================
def build_generation_prompt(input_text: str, exclude_list: List[str], is_url: bool) -> str:
    context_type = "网页内容" if is_url else "用户指令"
    exclude_str = ", ".join(exclude_list[-120:]) if exclude_list else "无"

    prompt = f"""
任务：解析以下【{context_type}】提取或挖掘新的数据集（最多 5 个）。
跳过已存在的：[{exclude_str}]
未找到返回：{{"datasets":[]}}

要求：
- 严格返回 JSON
- 所有描述性内容(简介, 场景等)务必翻译为【中文】
- 缺失填 null，不要编造文字

包含字段：
- name (中英文)
- description (中文简介)
- macro_field (选：自然科学, 工程与技术, 生命科学与医学, 农业与环境科学, 社会科学, 人文艺术, 人工智能与计算机, 综合与交叉学科)
- sub_field (中文)
- use_case (中文场景)
- institution
- paper_citation
- paper_link
- db_link
- format
- detailed_license (中文描述)
- license_type (选：免费开源, 注册免费, 付费申请, 内部闭源)
- detailed_size (中文描述)
- size_type (选：KB, MB, GB, TB, PB)
- participant_count
- language
- release_year
- update_year

内容：
{input_text}
"""
    return prompt.strip()


# ==========================================
# 元数据生成
# ==========================================
def generate_metadata(
    provider: str,
    api_key: str,
    input_text: str,
    exclude_list: List[str],
    is_url: bool = False,
    retries: int = 1,
) -> Dict[str, Any]:
    prompt = build_generation_prompt(input_text, exclude_list, is_url)
    st.session_state["last_prompt_preview"] = prompt[:4000]

    last_result_text = ""
    for attempt in range(retries + 1):
        try:
            result_text = call_ai(provider, api_key, prompt, temperature=0.10 if provider == "DeepSeek" else 0.20)
            last_result_text = result_text
            st.session_state["last_raw_response"] = result_text

            truncated = looks_truncated_json(result_text)
            if truncated:
                parsed = repair_json_response(provider, api_key, result_text)
            else:
                try:
                    parsed = extract_best_json_payload(result_text)
                except json.JSONDecodeError:
                    parsed = repair_json_response(provider, api_key, result_text)

            cleaned = validate_and_normalize_payload(parsed)
            st.session_state["last_parsed_payload"] = cleaned
            return cleaned

        except Exception as e:
            error_msg = sanitize_error(str(e))
            if attempt == retries:
                return {"error": error_msg, "raw": last_result_text[:2000]}
            time.sleep(1)

    return {"error": "失败", "raw": last_result_text[:2000]}


# ==========================================
# DataFrame 更新 
# ==========================================
def update_dataframe(new_rows: List[Dict], source: str, model_name: str) -> Tuple[int, int, int]:
    df = st.session_state.df.copy()

    name_to_idx = {}
    for idx, val in df["数据集名称"].dropna().items():
        name_to_idx[normalize_name(val)] = idx

    added_rows = 0
    updated_rows = 0
    updated_fields = 0
    update_logs = []

    for item in new_rows:
        ds_name = str(item.get("数据集名称", "")).strip()
        norm_name = normalize_name(ds_name)

        if not norm_name:
            continue

        if norm_name in name_to_idx:
            idx = name_to_idx[norm_name]
            row_updated = False
            for df_col in COLUMN_MAP.values():
                new_val = item.get(df_col)
                old_val = df.at[idx, df_col]
                if is_empty_val(old_val) and not is_empty_val(new_val):
                    df.at[idx, df_col] = new_val
                    updated_fields += 1
                    row_updated = True

            if row_updated:
                df.at[idx, "检索模型"] = model_name
                updated_rows += 1
        else:
            new_row = {col: item.get(col, None) for col in COLUMN_MAP.values()}
            new_row["系统录入来源"] = source
            new_row["检索模型"] = model_name
            for col in ALL_COLUMNS:
                if col not in new_row:
                    new_row[col] = None
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            name_to_idx[norm_name] = len(df) - 1
            added_rows += 1

    st.session_state.df = df
    save_data(df)
    return added_rows, updated_rows, updated_fields


# ==========================================
# UI 渲染
# ==========================================
st.markdown("<h1 style='text-align: center;'>🌌 DataHub AI 科研引擎</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>极简·智能的数据挖掘与管理中台</p>", unsafe_allow_html=True)
st.divider()

if "df" not in st.session_state:
    st.session_state.df = load_data()

with st.sidebar:
    st.header("⚙️ 引擎配置")

    current_usage = get_global_usage()
    free_tries_left = max(0, FREE_TRIES_LIMIT - current_usage)
    st.metric(label=f"平台额度 (总:{FREE_TRIES_LIMIT})", value=f"{free_tries_left} 次", delta=f"已用 {current_usage} 次", delta_color="inverse")
    st.caption("前端访问防御。底层限流由 AI 接管。")

    provider_choice = st.selectbox("🧠 AI 引擎", ["DeepSeek", "OpenAI", "ModelScope", "Gemini"])
    show_debug = st.checkbox("🔍 开启开发者日志")

    default_key = get_api_key(provider_choice)
    use_default = False
    user_api_key = ""

    if free_tries_left > 0 and default_key:
        if not is_valid_key_format(provider_choice, default_key):
            user_api_key = st.text_input(f"请输入您的 {provider_choice} Key:", type="password")
        else:
            use_default = st.checkbox(f"使用平台 {provider_choice} 免费通道", value=True)
            if not use_default:
                user_api_key = st.text_input(f"使用自有 {provider_choice} Key:", type="password")
    else:
        user_api_key = st.text_input(f"请输入自有 {provider_choice} Key:", type="password")

    st.markdown("---")
    with st.expander("🛠️ 管理员特权", expanded=False):
        admin_pwd = st.text_input("🔑 输入密码:", type="password")
        
        if admin_pwd == "datahub":
            if st.button("🔄 重置 100 次平台额度"):
                with open(USAGE_FILE, "w") as f: f.write("0")
                st.success("✅ 额度已归零，请刷新。")
            
            st.markdown("##### 🪄 数据库魔法棒")
            st.caption("AI 外科手术级数据修改。")
            magic_cmd = st.text_area("修改指令 (如: '将 EN 改为 英文'):", height=68)
            
            if st.button("✨ 施展魔法", type="primary"):
                active_key = default_key if (use_default and default_key) else user_api_key
                if not active_key:
                    st.error("缺失 API Key")
                else:
                    try:
                        df_magic = pd.read_csv(DATA_FILE)
                        if df_magic.empty:
                            st.warning("库为空。")
                        else:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            backup_path = os.path.join(BACKUP_DIR, f"backup_{timestamp}.csv")
                            df_magic.to_csv(backup_path, index=False)
                            st.info(f"💾 已备份至 {backup_path}")
                            
                            with st.spinner("魔法进行中..."):
                                magic_prompt = f"""
任务：根据指令修改 JSON 数据库。
指令："{magic_cmd}"
要求：仅返回被修改的行，且必须保留原"数据集名称"字段以便匹配。严格返回 JSON: {{"modified_rows": [...]}}
当前数据：{df_magic.to_json(orient="records", force_ascii=False)}
"""
                                result_text = call_ai(provider_choice, active_key, magic_prompt, temperature=0.1)
                                decoder = json.JSONDecoder()
                                cleaned_json = None
                                result_text = result_text.replace('```json', '').replace('```', '').strip()
                                for i, ch in enumerate(result_text):
                                    if ch == "{":
                                        try:
                                            obj, _ = decoder.raw_decode(result_text[i:])
                                            if isinstance(obj, dict) and "modified_rows" in obj:
                                                cleaned_json = obj
                                                break
                                        except: pass
                                
                                if cleaned_json and cleaned_json.get("modified_rows"):
                                    mods = cleaned_json["modified_rows"]
                                    mod_count = 0
                                    for mod in mods:
                                        name = mod.get("数据集名称")
                                        if not name: continue
                                        idx_mask = df_magic["数据集名称"].astype(str).str.lower() == str(name).lower()
                                        if idx_mask.any():
                                            idx = df_magic[idx_mask].index[0]
                                            for k, v in mod.items():
                                                if k in df_magic.columns and k != "数据集名称":
                                                    df_magic.at[idx, k] = v
                                                    mod_count += 1
                                    df_magic.to_csv(DATA_FILE, index=False)
                                    st.success(f"✨ 成功修改 {len(mods)} 条记录。")
                                    st.session_state.df = df_magic
                                    st.rerun()
                                else:
                                    st.warning("未检测到有效变动。")
                    except Exception as e:
                        st.error(f"操作失败: {e}")

# ==========================================
# 🌟 极简智能输入区
# ==========================================
st.subheader("🔍 新建挖掘任务")
# 紧凑排列的单选框，隐藏冗长标签
input_mode = st.radio("挖掘模式", ["🌐 网址提取", "💬 描述指令", "🔑 关键词模式"], horizontal=True, label_visibility="collapsed")

user_input = ""
col1, col2 = st.columns([3, 1])

with col1:
    if input_mode == "🌐 网址提取":
        user_input = st.text_input("🔗 目标网址 (含 http/https):")
    elif input_mode == "💬 描述指令":
        user_input = st.text_area("✍️ 详细指令 (如: '寻找医疗领域的影像数据集'):", height=68)
    else:
        history_sources = st.session_state.df["系统录入来源"].dropna().unique().tolist()
        history_keywords = [k for k in history_sources if not str(k).startswith("http") and k not in ["关键词搜索", "未知"]]
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            selected_history = st.selectbox("📚 选择历史搜索词:", ["-- 新探索 --"] + history_keywords)
        with col_k2:
            new_keyword = st.text_input("🔑 或输入全新关键词:")
        
        if new_keyword.strip():
            user_input = new_keyword.strip()
        elif selected_history != "-- 新探索 --":
            user_input = selected_history

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if input_mode != "💬 描述指令": st.markdown("<br>", unsafe_allow_html=True) # 对齐按钮
    submit_button = st.button("🚀 启动 AI 挖掘", width="stretch", type="primary")

if "debug_logs" not in st.session_state:
    reset_debug_state()

# ==========================================
# 提交逻辑
# ==========================================
if submit_button:
    reset_debug_state()

    if not user_input.strip():
        st.warning("⚠️ 请输入有效内容。")
    else:
        active_key = default_key if (use_default and default_key) else user_api_key
        if not active_key:
            st.error("请配置有效的 API Key。")
            st.stop()

        is_url = user_input.strip().startswith("http")
        process_text = user_input.strip()
        max_loops = 1 if is_url else 8

        if is_url:
            with st.spinner("🕸️ 正在抓取网页..."):
                web_res = fetch_web_content(user_input.strip())
                if web_res.startswith("Error:"):
                    st.error(f"网页抓取失败: {web_res}")
                    st.stop()
                process_text = f"原始URL:\n{user_input.strip()}\n\n提取内容:\n{web_res}"

        existing_names = st.session_state.df["数据集名称"].dropna().tolist()
        total_added = 0
        total_updated_rows = 0
        total_updated_fields = 0
        rounds_executed = 0

        with st.status("🕵️ 任务执行中...", expanded=True) as status_panel:
            batch = 1
            while batch <= max_loops:
                rounds_executed += 1
                status_panel.write(f"▶️ **[第 {batch} 轮]** 引擎 `{provider_choice}` 检索中...")

                if use_default and not can_use_free_channel():
                    status_panel.write("🚫 平台额度耗尽，中止。")
                    break
                if use_default: increment_global_usage()

                try:
                    metadata = retry_with_backoff(generate_metadata, provider_choice, active_key, process_text, existing_names, is_url)
                except Exception as e:
                    error_msg = sanitize_error(str(e))
                    kind = classify_api_error(error_msg)
                    if kind == "quota_exhausted":
                        status_panel.write("🚨 API 额度耗尽。")
                        if use_default: exhaust_global_usage()
                    else:
                        status_panel.write(f"❌ 阻碍: {error_msg[:30]}...")
                    break

                if "error" in metadata:
                    status_panel.write(f"❌ 解析失败: {metadata['error'][:40]}")
                    break

                datasets_list = metadata.get("datasets", [])
                if not datasets_list:
                    status_panel.write(f"💡 **[第 {batch} 轮]** 数据已见底。")
                else:
                    new_rows = [{COLUMN_MAP[key]: item.get(key, None) for key in COLUMN_MAP} for item in datasets_list]
                    added, updated_rows, updated_fields = update_dataframe(new_rows, user_input.strip(), MODEL_NAME_MAP.get(provider_choice, "未知"))
                    total_added += added
                    total_updated_rows += updated_rows
                    total_updated_fields += updated_fields
                    existing_names = st.session_state.df["数据集名称"].dropna().tolist()

                    if added > 0: status_panel.write(f"✅ **[第 {batch} 轮]** 成功新增 **{added}** 个！")
                    elif updated_rows > 0: status_panel.write(f"🔄 **[第 {batch} 轮]** 补充更新 **{updated_rows}** 个！")
                    else: status_panel.write(f"ℹ️ **[第 {batch} 轮]** 无需更新的重复项。")

                if not is_url and added == 0 and updated_rows == 0:
                    status_panel.write("💡 早停机制触发。")
                    break

                batch += 1
                if batch <= max_loops: time.sleep(1.2)

            status_panel.update(label=f"🎯 探测结束 (共 {rounds_executed} 轮)", state="complete", expanded=False)

        c1, c2, c3 = st.columns(3)
        c1.metric("✨ 新增记录", f"{total_added} 个")
        c2.metric("🔄 补全旧源", f"{total_updated_rows} 个")
        c3.metric("🧩 修复字段", f"{total_updated_fields} 个")

        if total_added > 0: st.balloons()

# ==========================================
# 调试面板
# ==========================================
if show_debug:
    st.divider()
    st.subheader("🧪 调试面板")
    with st.expander("查看 AI 原始返回", expanded=False):
        st.code(st.session_state.get("last_raw_response", "暂无数据"))

# ==========================================
# 数据展示区与海量图表
# ==========================================
st.divider()
st.subheader("📚 知识图谱数据库")

try:
    df_display = st.session_state.df
    if df_display.empty:
        st.info("暂无数据。")
    else:
        st.dataframe(
            df_display,
            width="stretch",
            hide_index=True,
            column_config={
                "数据集名称": st.column_config.TextColumn("名称", width="medium"),
                "主领域": st.column_config.TextColumn("领域", width="small"),
                "核心机构": st.column_config.TextColumn("机构", width="small"),
                "论文链接": st.column_config.LinkColumn("📄 论文", width="small"),
                "数据库官方链接": st.column_config.LinkColumn("🔗 数据源", width="small"),
            }
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_chart1, col_chart2 = st.columns([3, 2])

        with col_chart1:
            st.markdown("##### 🧩 领域热力树状图")
            tree_data = df_display.dropna(subset=["主领域", "子领域"]).copy()
            if not tree_data.empty:
                tree_data["子领域"] = tree_data["子领域"].astype(str).str.split(r'[,]+')
                tree_data = tree_data.explode("子领域")
                tree_data["子领域"] = tree_data["子领域"].str.strip()
                tree_data = tree_data[tree_data["子领域"] != ""]
                
                grouped_tree = tree_data.groupby(["主领域", "子领域"]).size().reset_index(name="数量")
                fig_treemap = px.treemap(grouped_tree, path=["主领域", "子领域"], values="数量", color="数量", color_continuous_scale="Blues")
                fig_treemap.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                st.plotly_chart(fig_treemap, width="stretch")

        with col_chart2:
            st.markdown("##### 📜 协议类型")
            license_data = df_display.dropna(subset=["协议类型"])["协议类型"].value_counts().reset_index()
            license_data.columns = ["协议类型", "数量"]
            if not license_data.empty:
                fig_donut = px.pie(license_data, names="协议类型", values="数量", hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
                fig_donut.update_traces(textposition="inside", textinfo="percent+label")
                fig_donut.update_layout(margin=dict(t=10, l=10, r=10, b=10), showlegend=False)
                st.plotly_chart(fig_donut, width="stretch")

        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            st.markdown("##### 📦 量级分布")
            size_data = df_display.dropna(subset=["量级(标准化)"])["量级(标准化)"].value_counts().reset_index()
            size_data.columns = ["量级", "数量"]
            if not size_data.empty:
                fig_size = px.bar(size_data, x="量级", y="数量", color="量级", color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_size, width="stretch")
                
        with col_chart4:
            st.markdown("##### 🌐 语种分布")
            lang_data = df_display.dropna(subset=["语种"])["语种"].str.split(r'[,]+').explode().str.strip()
            lang_data = lang_data[lang_data != ""]
            lang_data = lang_data.value_counts().reset_index()
            lang_data.columns = ["语种", "数量"]
            if not lang_data.empty:
                fig_lang = px.pie(lang_data, names="语种", values="数量", hole=0.3, color_discrete_sequence=px.colors.qualitative.Set3)
                fig_lang.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_lang, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🏛️ Top 10 核心数据机构")
        inst_data = df_display.dropna(subset=["核心机构"])["核心机构"].str.split(r'[,]+').explode().str.strip()
        inst_data = inst_data[inst_data != ""]
        inst_data = inst_data.value_counts().head(10).reset_index()
        inst_data.columns = ["机构", "收录数量"]
        if not inst_data.empty:
            fig_inst = px.bar(inst_data, x="收录数量", y="机构", orientation='h', color="收录数量", color_continuous_scale="Purples")
            fig_inst.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_inst, width="stretch")

except Exception as e:
    st.warning(f"读取失败: {sanitize_error(str(e))}")