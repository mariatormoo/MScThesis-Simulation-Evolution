# Streamlit UI for Module 3: Generative AI Decision Reports
from __future__ import annotations

import streamlit as st
import pandas as pd
#import openai
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional

# Optional OpenAI import lazily when needed.


# ----------------------------
# URL query-param helpers (match app shell)
# ----------------------------

def _get_qp(key: str, default: str) -> str:
    try:
        return st.query_params.get(key, default)  # type: ignore[attr-defined]
    except Exception:
        qp = st.experimental_get_query_params()
        return (qp.get(key, [default]) or [default])[0]


def _set_qp(**kwargs) -> None:
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        for k, v in kwargs.items():
            qp[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)


@dataclass
class ReportConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.4
    audience: str = "Board of Directors"
    tone: str = "Concise & executive"
    length: str = "Medium"
    sections: tuple[str, ...] = (
        "Executive Summary",
        "KPIs",
        "Risks",
        "Opportunities",
        "Next Actions",
    )


def _read_upload(file) -> str:
    if file is None:
        return ""
    try:
        content = file.read()
        try:
            return content.decode("utf-8")
        except Exception:
            return content.decode("latin-1", errors="ignore")
    except Exception:
        return ""
    

def _build_prompt(cfg: ReportConfig, key_metrics_text: str, context_text: str, kpi_df: Optional[pd.DataFrame]) -> tuple[str, list[dict]]:
    """Compose system+user messages. Accepts optional KPI table."""
    length_map = {"Short": "~300 words", "Medium": "~600 words", "Long": "~900 words"}

    # Sections as markdown bullet list
    sections_md = "".join(f"- {s}" for s in cfg.sections)

    # Structured KPI summary if provided
    kpi_block = ""
    if isinstance(kpi_df, pd.DataFrame) and not kpi_df.empty:
        # Keep it light: show up to 30 rows, 6 cols
        df_clip = kpi_df.iloc[:30, :6].copy()
        # Try to coerce numerics nicely
        for c in df_clip.columns:
            try:
                df_clip[c] = pd.to_numeric(df_clip[c])
            except Exception:
                pass
        kpi_lines = []
        # If has name/value columns, condense
        if set(map(str.lower, df_clip.columns)).issuperset({"metric", "value"}):
            name_col = [c for c in df_clip.columns if c.lower()=="metric"][0]
            val_col = [c for c in df_clip.columns if c.lower()=="value"][0]
            for _, row in df_clip.iterrows():
                kpi_lines.append(f"- {row[name_col]}: {row[val_col]}")
        else:
            # generic first-column as label
            label_col = df_clip.columns[0]
            for _, row in df_clip.iterrows():
                label = row[label_col]
                vals = ", ".join(f"{c}={row[c]}" for c in df_clip.columns[1:])
                kpi_lines.append(f"- {label}: {vals}")
        kpi_block = "".join(kpi_lines)

    system = (
        "You are a CFO copilot that writes clear, structured, board-ready decision reports. "
        "Use bullet points where helpful, include numbers and percentages if present, and keep the tone professional. "
        "Use Markdown headings for sections. If data is missing, do not fabricate; state assumptions explicitly."
    )

    user_parts = [
        f"Create a **{cfg.length}** ({length_map.get(cfg.length, '')}) **{cfg.tone}** report for the **{cfg.audience}**.",
        "**Sections to include (in this exact order):**" + sections_md,
    ]

    if kpi_block:
        user_parts.append("**Structured KPIs (from table):**"+ kpi_block)

    if key_metrics_text.strip():
        user_parts.append("**Financial input (verbatim / unstructured):**" + key_metrics_text.strip())

    if context_text.strip():
        user_parts.append("**Additional context (optional):**" + context_text.strip())

    user_parts.append(
        textwrap.dedent(
            """
            Rules:
            - Start with a crisp one-paragraph Executive Summary.
            - Use Markdown headings (#, ##) and lists; avoid tables unless necessary.
            - End with a short 'Decision & Owners' list under 'Next Actions'.
            """
        ).strip()
    )

    user = "".join(user_parts)
    return system, [{"role": "user", "content": user}]


# -----------------------------------------------------------------
# Suggestions mini-chat (local ideas or LLM if key available)
# -----------------------------------------------------------------

SUGGESTION_SEEDS = [
    "Add a scenario comparison table with KPI deltas vs baseline.",
    "Highlight 3 risks and 3 mitigations with owners and dates.",
    "Add a one-slide board appendix with assumptions.",
    "Show CAC payback vs last quarter and trend arrow.",
    "Split revenue by segment and call out mix-shift effects.",
]


def _suggestions_reply(prompt: str, api_key: str | None) -> str:
    prompt = (prompt or "").strip()
    base_hint = "Tip: Keep it tight. Use bullets and quantify impact where possible."

    if not api_key:
        # Local lightweight heuristic suggestions
        if not prompt:
            return "Here are ideas you can add: " + ", ".join(SUGGESTION_SEEDS[:3]) + "."
        if any(x in prompt.lower() for x in ["risk", "mitig"]):
            return "Consider ranking risks by likelihood x impact, and add a mitigation owner/date. " + base_hint
        if any(x in prompt.lower() for x in ["board", "exco", "exec"]):
            return "Open with a single-sentence ask, then 3 bullets: context, options, decision. " + base_hint
        return "Good direction. Add specific numbers, show trend vs last quarter, and finish with owners/dates. " + base_hint

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a helpful product/design coach for executive finance reports. Be concise and specific."},
                {"role": "user", "content": prompt or "Suggest 3 impactful additions for a CFO decision report."},
            ],
            max_tokens=200,
        )
        return r.choices[0].message.content or "(no suggestion)"
    except Exception:
        return "Couldn't contact the suggestion model. Try demo mode or remove the API key."


# -----------------------------------------------------------------
# Main module UI
# -----------------------------------------------------------------

def run_ai_reporting_module():
    st.header("🧾 Generative AI: Decision Report Generator")
    st.caption("Draft polished, executive-ready narratives from your numbers.")

    # Read initial values from URL
    init_model = _get_qp("model", "gpt-4o-mini")
    init_length = _get_qp("length", "Medium")
    init_tone = _get_qp("tone", "Concise & executive")
    init_aud = _get_qp("aud", "Board of Directors")

    with st.expander("Report settings", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0 if init_model=="gpt-4o-mini" else 1)
        with col2:
            temperature = st.slider("Creativity", 0.0, 1.0, 0.4, 0.05)
        with col3:
            length = st.select_slider("Length", options=["Short", "Medium", "Long"], value=init_length if init_length in ["Short","Medium","Long"] else "Medium")
        col4, col5 = st.columns(2)
        with col4:
            audience = st.text_input("Audience", value=init_aud)
        with col5:
            tone = st.selectbox("Tone", ["Concise & executive", "Neutral", "Persuasive", "Analytical"], index=max(0, ["Concise & executive","Neutral","Persuasive","Analytical"].index(init_tone) if init_tone in ["Concise & executive","Neutral","Persuasive","Analytical"] else 0))
        sections = st.multiselect(
            "Sections",
            ["Executive Summary", "KPIs", "Risks", "Opportunities", "Scenarios", "Next Actions"],
            default=["Executive Summary", "KPIs", "Risks", "Opportunities", "Next Actions"],
        )

        # Shareable link helper
        if st.button("🔗 Update URL with settings"):
            _set_qp(model=model, length=length, tone=tone, aud=audience)
            st.toast("URL updated – copy from your browser.")

    # API key and demo mode
    api_prefill = os.getenv("OPENAI_API_KEY", "")

    if not api_prefill:
        try:
            api_prefill = st.secrets["OPENAI_API_KEY"]
        except Exception:
            api_prefill = ""

    #api_prefill = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    api_key = st.text_input("OpenAI API key (or set in Secrets)", value=api_prefill, type="password", help="Store it in Secrets for convenience.")

    c1, c2 = st.columns([2, 1])
    with c1:
        key_metrics = st.text_area(
            "Paste key financial metrics / notes",
            height=180,
            placeholder="Revenue YoY +22%… Cash runway 13 months… CAC payback 9.2 months…",
        )
    with c2:
        context_file = st.file_uploader("Optional: attach context (txt/csv/md)", type=["txt", "csv", "md"])
        demo_mode = st.checkbox("Demo mode (no API call)", value=st.session_state.get("demo_mode", True))

    # Structured KPI editor: either uploaded CSV or editable demo table
    st.subheader("📋 KPI Table (optional)")
    uploaded_kpi = None
    if context_file is not None and context_file.name.lower().endswith(".csv"):
        try:
            uploaded_kpi = pd.read_csv(context_file)
        except Exception:
            uploaded_kpi = None

    if uploaded_kpi is None:
        demo_kpis = pd.DataFrame(
            {
                "Metric": ["Revenue YoY", "Gross Margin", "NRR", "CAC Payback (m)", "Runway (m)"],
                "Value": [22, 61, 114, 9.2, 13],
                "Unit": ["%", "%", "%", "months", "months"],
            }
        )
        kpi_df = st.data_editor(demo_kpis, num_rows="dynamic", use_container_width=True)
    else:
        st.info("Using uploaded CSV as KPI table (you can still edit below).")
        kpi_df = st.data_editor(uploaded_kpi, num_rows="dynamic", use_container_width=True)

    cfg = ReportConfig(
        model=model,
        temperature=temperature,
        audience=audience,
        tone=tone,
        length=length,
        sections=tuple(sections),
    )

    cta1, cta2 = st.columns([1,1])
    with cta1:
        generate = st.button("🚀 Generate report", type="primary")
    with cta2:
        st.write("")
        st.write("")
        copy_params = st.button("🔗 Copy current settings to URL")
        if copy_params:
            _set_qp(model=cfg.model, length=cfg.length, tone=cfg.tone, aud=cfg.audience)
            st.toast("URL updated – copy from your browser.")

    if generate:
        system_msg, user_msgs = _build_prompt(cfg, key_metrics.strip(), _read_upload(context_file) if context_file and not context_file.name.lower().endswith(".csv") else "", kpi_df)

        with st.status("Preparing your report…", expanded=False) as s:
            try:
                if demo_mode or not api_key:
                    s.update(label="Demo mode: generating sample output…")
                    demo_report = f"""
                    # Executive Summary
                    - Momentum remains strong with improving gross margins and disciplined opex. Cash runway > 12 months.

                    ## KPIs
                    - Revenue growth: +22% YoY; GM: 61% (+3pp); NRR: 114%; CAC payback: 9.2 months.

                    ## Risks
                    - Topline sensitivity to approvals; FX headwinds; hiring pace.

                    ## Opportunities
                    - Price uplift in Enterprise; self-serve funnel; working capital optimization.

                    ## Next Actions
                    - Approve FY plan scenario B; prioritize pipeline hygiene; finalize debt facility renewal.
                    """
                    result_md = demo_report
                else:
                    from openai import OpenAI  # lazy import

                    s.update(label="Calling GPT…", state="running")
                    client = OpenAI(api_key=api_key)
                    resp = client.chat.completions.create(
                        model=cfg.model,
                        temperature=cfg.temperature,
                        messages=[{"role": "system", "content": system_msg}, *user_msgs],
                        max_tokens=1200,
                    )
                    result_md = resp.choices[0].message.content or "(No content returned)"

                s.update(label="Formatting…", state="running")
                st.subheader("Generated Report")
                st.markdown(result_md)

                st.download_button(
                    "⬇️ Download Markdown",
                    data=result_md,
                    file_name="decision_report.md",
                    mime="text/markdown",
                )

                st.toast("Report ready")
                s.update(label="Done", state="complete")
            except Exception as e:
                st.error("Couldn't generate the report.")
                st.exception(e)

    # ------------------
    # Suggestions mini-chat
    # ------------------
    st.divider()
    st.subheader("💡 Suggestions Copilot")
    st.caption("Ask for ideas to refine your report. Works offline; improves with an API key.")

    if "suggest_msgs" not in st.session_state:
        st.session_state.suggest_msgs = []

    # Render history
    for role, content in st.session_state.suggest_msgs:
        with st.chat_message(role):
            st.markdown(content)

    user_msg = st.chat_input("Ask for improvements, e.g., 'How to present risks?' ")
    if user_msg is not None:
        st.session_state.suggest_msgs.append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = _suggestions_reply(user_msg, api_key if (api_key and not demo_mode) else None)
                st.markdown(reply)
                st.session_state.suggest_msgs.append(("assistant", reply))

