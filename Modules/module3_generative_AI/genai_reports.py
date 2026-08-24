# Streamlit UI for Module 3: Generative AI Decision Reports
from __future__ import annotations

# Import Libraries
import os
import json
#import openai
import textwrap
import pandas as pd
import streamlit as st
from dataclasses import dataclass, asdict
from typing import Optional

# API Helper
def _get_api_key():
    api_key = os.environ.get("OPENAI_API_KEY") 
    if api_key:
        return api_key
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None
    return None

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


# Report Default Configuration
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


# Extract content of Uploaded File 
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



def _get_openai_client(api_key: str):
    """
    Try to import the new (>=1.0) or legacy OpenAI SDK.
    Always return a tuple (mode, client) where mode is "new", "legacy" or None.
    """
    
    try:
        # new SDK Style
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        return ("new", client)
    
    except Exception:
        try:
            # legacy SDK Style
            import openai
            return ("legacy", openai)
        except Exception:
            #st.error("OpenAI SDK not found. Please install the OpenAI package.")
            return ("stub", None)
        

# -----------------------------------------------------------------
# Template-based Prompts
# -----------------------------------------------------------------

TEMPLATES = {
    "Forecast Summary": """
    You are an expert financial data analyst preparing a report for {audience}. 
    Analyze the provided financial and operational data, and provide a concise summary focusing on the forecasted trends, key insights, 
    and any actionable recommendations. Make it easy to understand for a non-technical audience.

    Write a concise CFO-ready forecast summary based on the data provided.

    Context:
     - Key metrics: {metrics}
     - Scenario Horizon: {horizon} months
     - Risk Appetite: {risk_appetite}
     - Target Tone: {tone}
     - Desired length: {length} words
     - Sections to include: {sections}

    Guidelines:
    - Deliver the report in {tone} style suitable for {audience}.
    - Structure according to the requested sections: {sections}.
    - Provide {length} words output (adjust depth accordingly).
    - Creativity level (temperature): {temperature}.

    Deliver 5-8 crisp bullet points with numbers where possible.
    """,

    "Risk Report": """
    You are a risk management expert tasked with preparing a professional risk report for {audience}. 
    Analyze the financial forecast data and identify potential risks, uncertainties, and mitigation strategies.
    Provide a clear and actionable risk report that highlights key risk factors and recommendations for managing them.      
   
    Produce a risk report for the board based on the data provided.

   Context:
     - Key metrics: {metrics}
     - Horizon: {horizon} months
     - Risk Appetite: {risk_appetite}
     - Tone: {tone}
     - Desired Length: {length} words
     - Sections to include: {sections}

    Guidelines:
    - Use a {tone} style tailored to {audience}.
    - Structure strictly by these sections: {sections}.
    - Keep the output in {length} words (scale detail accordingly).
    - Creativity level (temperature): {temperature}.
    - Include: top financial/market/operational risks, early indicators, mitigation strategies, and impact ranges with likelihoods.

    Use bullet points.
    """,

    "Board Presentation": """
    You are a senior expert business consultant preparing a board-level presentation for {audience}.
    Your task is to summarize the forecast and provide clear and concise one-pager. 

    Create a professional board presentation outline based on the following data.
    The outline should include key insights, strategic recommendations, and visual aids to effectively 
    communicate the forecast to the board of directors.  

    Context: 
     - Key metrics: {metrics}
     - Horizon: {horizon} months
     - Risk Appetite: {risk_appetite}
     - Tone: {tone}
     - Target Length: {length} words
     - Sections required: {sections}

     Guidelines:
     - Draft in a {tone} tone appropriate for {audience}.
     - Structure exactly with sections: {sections}.
     - Creativity level (temperature): {temperature}.
     - Deliver a {length} words report, optimized for board readability.
     - Use bullet points, clear headings, and suggest visuals where impactful.
    
    Draft a one-pager for the board.
    Include three section with headings: Outlook, Risks & Opportunities, Recommended Actions.
    Structure with clear sections and suggested visuals.

    Use bullet points. Use an engaging tone.
    """,

}

# Helper -> 1.4 tokens/word approximation
def length_words_to_tokens(words: int) -> int:
    return int(words * 1.4)


def _call_openai(prompt: str, api_key: Optional[str]) -> str:
    """
    Attempt to call OpenAI using either the new or legacy SDK.
    If no API key or client is available, return a helpful local stub.
    """

    # Control missing API key
    if not api_key:
        return(
            "Oops! Something happened.\n"
            "\nAI report stub. No API key found.\n"
            "\nIntroduce an OpenAI API Key to enable real generation.\n"
        )

    mode, client = _get_openai_client(api_key)

    try:
        # Call the appropriate OpenAI API based on the SDK version
        if mode == "new" and client:
            try:
                # New SDK Style
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert CFO assistant. Be concise, professional and actionable."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3, # Lower temperature for more focused responses
                    max_tokens=min(4000, int(length_words_to_tokens(length))),
                )
                return response.choices[0].message.content.strip()
            
            except Exception:
                return f"Error calling OpenAI API"
            
        elif mode == "legacy" and client:
            try:
                # Legacy SDK Style
                client.api_key = api_key
                response = client.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert CFO assistant. Be concise, professional and actionable."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3, # Lower temperature for more focused responses
                )
                return response.choices[0].text.strip()
            
            except Exception:
                return f"Error calling OpenAI API"
            
    except Exception:
        return f"Error initializing OpenAI client"



# -----------------------------------------------------------------
# Suggestions mini-chat (local ideas or LLM if key available)
# -----------------------------------------------------------------

"""SUGGESTION_SEEDS = [
    "Add a scenario comparison table with KPI deltas vs baseline.",
    "Highlight 3 risks and 3 mitigations with owners and dates.",
    "Add a one-slide board appendix with assumptions.",
    "Show CAC payback vs last quarter and trend arrow.",
    "Split revenue by segment and call out mix-shift effects.",
]"""

SUGGESTION_SEEDS = {
    'Risks':[
        "Rank risks by likelihood x impact and show top 3 in a heatmap.",
        "List risk owners with deadlines for mitigation plans.",
        "Highlight early-warning indicators linked to market signals.",
    ], 
    'KPIs': [
        "Add a comparison table showing KPI deltas vs last quarter.",
        "Visualize CAC payback trend with arrows and traffic-light colors.",
        "Segment revenue by region or product line to highlight mix-shift effects.",
    ], 
    'Visuals': [
        "Add a waterfall chart for Free Cash Flow drivers.",
        "Include a sensitivity tornado chart for interest rates and FX.",
        "Create a dashboard-style appendix with sparklines for 6 KPIs.",
    ], 
    'Board framing': [
        "Open with one-sentence decision ask, then 3 bullets: context, options, recommendation.",
        "Draft a 'Risk & Opportunities' section with balanced tone.",
        "Close with 'Next Steps' assigning accountability and dates."
    ],
    }

def _build_copilot_context(prompt: str, metrics: str, horizon: int, risk_appetite: str, 
                           tone: str, audience: str, sections: list[str]) -> str:
    """
    Comine user prompt with report context for smarter copilot replies.
    """

    context = f"""
    User prompt:{prompt}

    Report Context:
    - Audience: {audience}
    - Tone: {tone}
    - Time Horizon: {horizon} months
    - Risk Appetite: {risk_appetite}
    - Sections: {', '.join(sections)}
    - Metrics: {metrics}
    """

    return context.strip()


def _local_suggestion_engine(prompt: str) -> str:
    """
    Generate smart offline suggestions without API calls.
    """
    #prompt = (prompt or "").strip()
    prompt = (prompt or "").lower()

    if not prompt:
        # General suggestions (pick from all bullets)
        all_seeds = sum(SUGGESTION_SEEDS.values(), [])
        return "Here are some impactful ideas you can add: \n-" + "\n- ".join(all_seeds[:4])
    
    if any(word in prompt for word in ["risk", "mitig", "exposure", "uncertain"]):
       return "Risk-focused ideas:\n- " + "\n- ".join(SUGGESTION_SEEDS["Risks"])
    
    if any(word in prompt for word in ["kpi", "metric", "growth", "margin"]):
       return "KPI-focused ideas:\n- " + "\n- ".join(SUGGESTION_SEEDS["KPIs"])
    
    if any(word in prompt for word in ["visual", "chart", "slide", "image"]):
       return "Visualization-focused ideas:\n- " + "\n- ".join(SUGGESTION_SEEDS["Visuals"])
    
    if any(word in prompt for word in ["board", "cfo", "director", "uncertain", "executive"]):
       return "Board-framing ideas:\n- " + "\n- ".join(SUGGESTION_SEEDS["Board framing"])

    # Fallback
    all_seeds = sum(SUGGESTION_SEEDS.values(), [])
    return "General ideas:\n- " + "\n- ".join(all_seeds[:4])
    


def _suggestions_reply(prompt: str, api_key: str | None, metrics: str, horizon: int, risk_appetite: str, 
                       tone: str, audience: str, sections: list[str]) -> str:
    
    base_hint = "Tip: Keep it tight. Use bullets and quantify impact where possible."

    full_context = _build_copilot_context(prompt, metrics, horizon, risk_appetite, tone, audience, sections)

    if not api_key:
        # Local lightweight heuristic suggestions
        return _local_suggestion_engine(prompt)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.5,
            messages=[
                {"role": "system", "content": "You are a helpful product/design coach for executive finance reports. Be concise and specific."},
                {"role": "user", "content": full_context or prompt or "Suggest 3 impactful additions for a CFO decision report."},
            ],
            max_tokens=200,
        )
        return r.choices[0].message.content or "(no suggestion)"
    except Exception:
        return "Couldn't contact the suggestion model. Falling back to demo mode:\n\n" + _local_suggestion_engine(prompt)


# -----------------------------------------------------------------
# Main module UI
# -----------------------------------------------------------------

def run_genai_module():
    st.header("🧾 Generative AI: Decision Report Generator")
    st.caption("Generate CFO-style narrative reports with GPT based on your data.")

    api_key = _get_api_key()

    # Get initial values from URL query params
    init_model = _get_qp("model", "gpt-4o-mini")
    init_length = _get_qp("length", "Medium")
    init_tone = _get_qp("tone", "Concise & executive")
    init_aud = _get_qp("aud", "Board of Directors")
    
    # Read values (Inputs. section)
    st.subheader("⚙️ Report Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0 if init_model=="gpt-4o-mini" else 1)
    with col2:
        doc_type = st.selectbox("Document Type", list(TEMPLATES.keys()), index=0)
    with col3:
        temperature = st.slider("Creativity Level", 0.0, 1.0, 0.4, 0.05)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        risk_appetite = st.selectbox("Risk Appetite", ["Low", "Moderate", "High"], index=1)
    with col5:
        tone = st.selectbox("Tone", ["Concise & executive", "Neutral", "Persuasive", "Analytical"], index=max(0, ["Concise & executive","Neutral","Persuasive","Analytical"].index(init_tone) if init_tone in ["Concise & executive","Neutral","Persuasive","Analytical"] else 0))
    with col6:
        audience = st.text_input("Audience", value=init_aud)
    horizon = st.slider("Scenario Horizon (months)", min_value=1, max_value=36, value=6, step=1)
    metrics = st.text_area("Key Financial Metrics (comma-separated)", 
                           height=180,
                           placeholder="Revenue YoY 22%, Gross Margin 41%, NRR 114%, CAC Payback 9.2 months, Net Debt 5m"
                           )

    sections = st.multiselect(
            "Document Sections",
            ["Executive Summary", "KPIs", "Risks", "Opportunities", "Scenarios", "Next Actions"],
            default=["Executive Summary", "KPIs", "Risks", "Opportunities", "Next Actions"],
        )
    length = st.slider("Length (in words)", min_value=100, value=600, max_value=2500, step=50, help="Approximate target length in words.")
    
    # Shareable link helper
    if st.button("🔗 Update URL with settings"):
        _set_qp(model=model, length=length, tone=tone, aud=audience)
        st.toast("URL updated – copy from your browser.")

    # API key
    api_prefill = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API key (or set in Secrets)", value=api_prefill, type="password", help="Store it in Secrets for convenience.")

    
    # --------------------------
    # Context Upload + KPI Table
    # ---------------------------

    # Option to provide file with context
    st.divider()
    st.subheader("📂 More Context")
    context_file = st.file_uploader("Optional: upload additional context (CSV format)", type=["csv"])
    #demo_mode = st.checkbox("Demo mode (no additional context)", value=st.session_state.get("demo_mode", True))

    context_text = ""
    metrics_df = None

    # KPI Metrics Table
    # Structured KPI editor: either uploaded CSV or editable demo table
    st.subheader("📋 KPIs Table Visualization")

    # Uploaded CSV
    if context_file and context_file.name.endswith('.csv'):
        try:
            metrics_df = pd.read_csv(context_file)
        except Exception:
            metrics_df = None

    # No Context Uploaded
    if context_file is None:
        demo_kpis = pd.DataFrame(
            {
                "Metric": ["Revenue YoY", "Gross Margin", "NRR", "CAC Payback", "Net Debt"],
                "Value": [22, 41, 114, 9.2, 5],
                "Unit": ["%", "%", "%", "months", "milions"],
            }
        )

        metrics_df = st.data_editor(demo_kpis, num_rows="dynamic", use_container_width=True)

    else:
        st.info("Using uploaded CSV as KPI table (you can still edit below).")
        metrics_df = st.data_editor(metrics_df, num_rows='dynamic', use_container_width=True)



    # Generate Report

    if st.button("🚀 Generate Document", type="primary"):
        kpi_summary = ""
        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            kpi_summary = "\n".join(
                f"- {row.get('Metric','')}: {row.get('Value','')} {row.get('Unit', '')}"
                for _, row in metrics_df.iterrows()
            )

        prompt = TEMPLATES[doc_type].format(
            metrics=metrics, 
            horizon=horizon, 
            risk_appetite=risk_appetite,
            tone=tone,
            length=length,
            sections=sections,
            temperature=temperature,
            audience=audience,
        )
        
        with st.status("Generating report...", expanded=False) as s:
            s.update(label="Calling GPT...", state='running')
            try:
                report = _call_openai(prompt, api_key)
                if api_key:
                    s.update(label="Formatting...", state='running')
                    s.update(label="Successfully Generated! (click to access report)", state='complete')
                else:
                    s.update(label="An error occured! (click to check details)", state='complete')

                st.subheader("Result")
                st.write(report)
                st.download_button(
                    "⬇️ Download Report (TXT)", 
                    report, 
                    file_name=f"{doc_type.replace(' ', '_').lower()}_report.txt",
                    mime='text/markdown',
                )   

                st.toast("Report Ready")
            except Exception:
                st.error("An error occurred. Couldn't generate the report.")


    # ----------------------
    # Suggestions mini-chat
    # ----------------------
    st.divider()
    st.subheader("💡 Suggestions Copilot")
    st.caption("Ask for ideas to refine your report. Works offline; improves with an API key.")

    if "suggest_msgs" not in st.session_state:
        st.session_state.suggest_msgs = []

    # Render history
    for role, content in st.session_state.suggest_msgs:
        with st.chat_message(role):
            st.markdown(content)

    user_msg = st.chat_input("Ask for improvements, e.g., 'How to present risks?', 'Suggest 3 KPIs to add', 'How to frame it for the board members?'")
    if user_msg is not None:
        st.session_state.suggest_msgs.append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = _suggestions_reply(user_msg, 
                                           api_key if (api_key) else None,
                                           metrics,
                                           horizon,
                                           risk_appetite,
                                           tone,
                                           audience,
                                           sections,
                                        )
                st.markdown(reply)
                st.session_state.suggest_msgs.append(("assistant", reply))


        
