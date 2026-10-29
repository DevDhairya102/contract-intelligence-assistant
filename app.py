"""
Streamlit Web Application for Contract Intelligence Assistant.
Provides enterprise contract search, clause ranking, character-level span visualization,
contract filtering, and explicit abstention reporting.
"""
import streamlit as st
import time
from pathlib import Path
import json

from src.config import PipelineConfig
from src.pipeline import ContractIntelligenceEngine
from src.document_loader import DocumentLoader

st.set_page_config(
    page_title="Contract Intelligence Assistant",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished enterprise look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .abstain-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }
    .clause-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .span-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .score-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine(config_dict: str):
    """Loads and caches ContractIntelligenceEngine for Streamlit session."""
    cfg = PipelineConfig(**json.loads(config_dict))
    return ContractIntelligenceEngine(config=cfg)

def main():
    st.markdown('<div class="main-header">📜 Contract Intelligence Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated legal clause retrieval, exact character offset localization, and confident abstention.</div>', unsafe_allow_html=True)

    # Sidebar: Configurations and Filters
    st.sidebar.header("⚙️ Pipeline Configuration")
    
    preset = st.sidebar.selectbox(
        "Configuration Preset",
        [
            "Final Production Pipeline (Hybrid + Clause-Aware + Rerank + Abstention)",
            "Hybrid Fusion (BM25 + Dense)",
            "Dense Semantic Retrieval (all-MiniLM-L6-v2)",
            "Baseline BM25 (Lexical Only)"
        ]
    )

    if "Final Production" in preset:
        config = PipelineConfig(
            experiment_name="production_pipeline",
            chunk_strategy="clause_aware",
            chunk_size=1000,
            retriever_type="hybrid",
            bm25_weight=0.35,
            dense_weight=0.65,
            use_reranker=True,
            abstention_mode="score_threshold",
            abstention_threshold=0.22,
            span_refinement=True
        )
    elif "Hybrid Fusion" in preset:
        config = PipelineConfig(
            experiment_name="hybrid_pipeline",
            chunk_strategy="fixed",
            chunk_size=800,
            retriever_type="hybrid",
            bm25_weight=0.35,
            dense_weight=0.65,
            abstention_mode="score_threshold",
            abstention_threshold=0.32
        )
    elif "Dense" in preset:
        config = PipelineConfig(
            experiment_name="dense_pipeline",
            chunk_strategy="fixed",
            chunk_size=800,
            retriever_type="dense",
            abstention_mode="score_threshold",
            abstention_threshold=0.30
        )
    else:
        config = PipelineConfig(
            experiment_name="baseline_bm25",
            chunk_strategy="fixed",
            chunk_size=800,
            retriever_type="bm25",
            abstention_mode="none"
        )

    # Advanced parameter overrides in sidebar
    with st.sidebar.expander("Advanced Tuning"):
        top_k = st.slider("Top-K Passages", min_value=1, max_value=10, value=config.top_k)
        config.top_k = top_k
        threshold = st.slider("Abstention Threshold", min_value=0.0, max_value=0.8, value=float(config.abstention_threshold), step=0.02)
        config.abstention_threshold = threshold
        span_refine = st.checkbox("Span Refinement", value=config.span_refinement)
        config.span_refinement = span_refine

    # Document Loader for metadata
    doc_loader = DocumentLoader(data_dir=config.data_dir)
    all_contracts = ["All Indexed Contracts (Global Search)"] + doc_loader.list_contracts()
    
    st.sidebar.header("📁 Contract Scope")
    selected_contract = st.sidebar.selectbox("Select Target Contract", all_contracts)
    contract_filter = None if "All Indexed" in selected_contract else selected_contract

    # Sample query selector for quick testing
    sample_queries = [
        "Select a sample query...",
        # Answerable samples from benchmark
        "Under what conditions may the agreement be terminated for cause or breach?",
        "What is the governing law and jurisdiction governing this agreement?",
        "What are the audit rights and record retention obligations of the parties?",
        "What is the exclusivity period and non-compete covenant?",
        # Unanswerable queries
        "What are the rules regarding TikTok and Instagram short-form video monetization?",
        "Under what circumstances may the licensee use artificial intelligence or generative AI models?",
        "What are the penalties for non-compliance with the GDPR Article 33 72-hour breach notification?",
        "Does this agreement permit cryptocurrency payments such as Bitcoin or Ethereum?"
    ]
    
    selected_sample = st.selectbox("💡 Try a sample question or type your own below:", sample_queries)
    default_text = "" if selected_sample == sample_queries[0] else selected_sample

    query = st.text_input("Enter your natural-language contract question:", value=default_text, placeholder="e.g., Does this agreement permit sublicensing or assignment to third parties?")

    if st.button("🔍 Search Contracts", type="primary") or (query and query != default_text):
        if not query.strip():
            st.warning("Please enter a question to search.")
            return

        with st.spinner("Analyzing contract clauses and calculating relevance..."):
            engine = load_engine(json.dumps(config.to_dict()))
            resp = engine.search(query=query.strip(), top_k=config.top_k, contract_filter=contract_filter)

        # Performance summary metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Execution Latency", f"{resp.latency_ms:.1f} ms")
        m2.metric("Confidence Score", f"{resp.confidence:.4f}")
        m3.metric("Decision", "ABSTAIN" if resp.is_abstained else "ANSWERED")

        # Display Results
        if resp.is_abstained:
            st.markdown(f"""
            <div class="abstain-box">
                <h4 style="margin-top:0; color:#B45309;">⚠️ System Abstention</h4>
                <p><strong>Reason:</strong> {resp.abstention_reason}</p>
                <p style="margin-bottom:0; font-size:0.9rem; color:#6B7280;">
                    The system verified that no clause in the indexed contracts has sufficient relevance to reliably answer your question.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"Retrieved {len(resp.passages)} relevant clause(s) satisfying confidence threshold.")
            
            for p in resp.passages:
                with st.container():
                    st.markdown(f"""
                    <div class="clause-card">
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                            <span><strong>Rank #{p.rank}</strong> — <code>{p.file_path}</code></span>
                            <div>
                                <span class="span-badge">Chars [{p.start_char}:{p.end_char}]</span>
                                <span class="score-badge">Score: {p.score:.4f}</span>
                            </div>
                        </div>
                        <div style="font-size: 0.95rem; line-height: 1.5; color: #1F2937; white-space: pre-wrap; background: white; padding: 10px; border-radius: 4px; border: 1px solid #E5E7EB;">{p.text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Expandable Document Context Viewer
                    with st.expander(f"View Full Contract Context ({Path(p.file_path).name})"):
                        full_doc = doc_loader.get_document(p.file_path)
                        if full_doc:
                            ctx_start = max(0, p.start_char - 400)
                            ctx_end = min(len(full_doc), p.end_char + 400)
                            before = full_doc[ctx_start:p.start_char]
                            highlighted = full_doc[p.start_char:p.end_char]
                            after = full_doc[p.end_char:ctx_end]
                            
                            st.markdown(f"""
                            <div style="font-family: monospace; font-size: 0.85rem; line-height: 1.4; background:#F9FAFB; padding:12px; border-radius:4px; max-height:300px; overflow-y:auto;">
                                <span style="color:#6B7280;">... {before}</span>
                                <mark style="background-color:#FEF08A; padding:2px 4px; border-radius:2px; font-weight:bold;">{highlighted}</mark>
                                <span style="color:#6B7280;">{after} ...</span>
                            </div>
                            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
