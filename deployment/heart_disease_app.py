import os
import streamlit as st

# Import only the two “tab” functions—no Streamlit calls at import time
from heart_disease_form import individual_tab
from heart_disease_batch import batch_tab

def main():
    # 1) This must be the very first Streamlit call in your app.
    st.set_page_config(
        page_title="🩺 💻 Heart Disease Predictor",
        page_icon="🩺",
        layout="wide",
    )

    # --- 2) Inject custom CSS for the page ---
    st.markdown(
        """
        <style>
        /* Page background */
        .stApp {
            background-color: #f7f9fc;
        }
        /* Main title */
        .main-title {
            font-size: 2.75rem;
            font-weight: 700;
            color: #2b2e33;
            text-align: center;
            margin-top: 1rem;
            margin-bottom: 0.25rem;
        }
        /* Subtitle / description */
        .subtitle {
            font-size: 1.25rem;
            color: #555555;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        /* Container box around tabs */
        .container-box {
            background-color: #ffffff;
            padding: 1rem 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #dde2e6;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        /* Style the tab labels (emojis + text) */
        .stTabs [role="tablist"] button {
            font-size: 1rem;
            font-weight: 600;
            color: #2b2e33;
        }
        .stTabs [role="tablist"] button[aria-selected="true"] {
            border-bottom: 3px solid #0074d9;
            color: #0074d9;
        }
        /* Hide any Streamlit-generated top margin for a cleaner look */
        .css-1avcm0n {
            margin-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- 4) Title & Subtitle (centered via CSS classes) ---
    st.markdown(
        '<div class="main-title">🩺 Heart Disease Predictor ❤️‍🩹</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Assess a single patient or run batch predictions in one click</div>',
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["🏥 Individual", "📦 Batch"])
    with tabs[0]:
        individual_tab()

    with tabs[1]:
        batch_tab()

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()