import streamlit as st
import pandas as pd
import json
import glob
import os
import plotly.express as px

st.set_page_config(layout="wide", page_title="Agent-MT Analysis")

@st.cache_data
def load_data():
    # Search for outputs in current or parent directory
    search_paths = ["outputs", "../outputs"]
    files = []
    base_dir = ""
    
    for path in search_paths:
        found = glob.glob(os.path.join(path, "**/report.json"), recursive=True)
        if found:
            files = found
            base_dir = path
            break
            
    data = []
    
    for f in files:
        try:
            # Normalize path separators
            f_norm = os.path.normpath(f)
            parts = f_norm.split(os.sep)
            
            # Identify where the relative path starts (after 'outputs')
            if "outputs" in parts:
                idx = parts.index("outputs")
                # Expected: .../outputs/{dataset}/{lang_pair}/{method}/{model}/report.json
                
                if len(parts) < idx + 6:
                    continue
                    
                dataset = parts[idx+1]
                lang_pair = parts[idx+2]
                raw_method = parts[idx+3]
                model = parts[idx+4]
            else:
                continue
            
            # Filter for Qwen models only
            if "qwen" not in model.lower():
                continue

            # Handle Terminology variants
            is_term = raw_method.endswith(".term")
            method_name = raw_method.replace(".term", "")
            variant = "Terminology" if is_term else "Standard"
            
            with open(f, 'r') as json_file:
                content = json.load(json_file)
                
            summary = content.get("summary", {})
            
            # Extract metrics
            input_tokens = summary.get("total_tokens_input", 0)
            output_tokens = summary.get("total_tokens_output", 0)
            total_tokens = input_tokens + output_tokens
            
            chrf = summary.get("avg_chrf_score", 0)
            bleu = summary.get("avg_bleu_score", 0)
            term_rate = summary.get("avg_term_success_rate")
            if term_rate is None:
                term_rate = 0
            
            # New Statistics
            latency = summary.get("avg_latency_seconds", 0)
            total_samples = content.get("total_samples", 0)
            successful_samples = content.get("successful_samples", 0)
            success_rate = (successful_samples / total_samples * 100) if total_samples > 0 else 0
            
            data.append({
                "dataset": dataset,
                "lang_pair": lang_pair,
                "method": method_name,
                "variant": variant,
                "full_method": raw_method,
                "model": model,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "avg_chrf": chrf,
                "avg_bleu": bleu,
                "avg_term_success": term_rate,
                "avg_latency": latency,
                "success_rate": success_rate,
                "file_path": f
            })
            
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            continue
            
    return pd.DataFrame(data)

def main():
    st.title("Agent-MT: Cost vs Quality Trade-off")
    st.markdown("Comparing **Qwen** models across different agentic workflows.")

    df = load_data()
    
    if df.empty:
        st.error("No data found matching the criteria.")
        return

    # Sidebar Filters
    st.sidebar.header("Configuration")
    
    # Dataset Filter
    datasets = sorted(df['dataset'].unique())
    selected_dataset = st.sidebar.selectbox("Select Dataset", datasets)
    
    # Lang Pair Filter (dependent on dataset)
    filtered_df = df[df['dataset'] == selected_dataset]
    lang_pairs = sorted(filtered_df['lang_pair'].unique())
    selected_lang = st.sidebar.selectbox("Select Language Pair", lang_pairs)
    
    # Terminology Filter
    term_options = ["All", "Standard", "Terminology"]
    selected_term = st.sidebar.selectbox("Experiment Type", term_options)

    # Filter based on Terminology selection
    plot_df = filtered_df[filtered_df['lang_pair'] == selected_lang]
    if selected_term == "Standard":
        plot_df = plot_df[plot_df['variant'] == "Standard"]
    elif selected_term == "Terminology":
        plot_df = plot_df[plot_df['variant'] == "Terminology"]
    
    # Metric Selection
    metric = st.sidebar.radio("Quality Metric", ["BLEU", "chrF", "Term Success Rate", "Latency (s)", "Success Rate (%)"])
    
    metric_map = {
        "BLEU": "avg_bleu",
        "chrF": "avg_chrf",
        "Term Success Rate": "avg_term_success",
        "Latency (s)": "avg_latency",
        "Success Rate (%)": "success_rate"
    }
    y_axis = metric_map[metric]

    # Symbol Configuration
    symbol_option = st.sidebar.radio("Map Symbol to:", ["Model", "Experiment Type"])
    symbol_col = "model" if symbol_option == "Model" else "variant"
    
    # Main Plot
    if not plot_df.empty:
        fig = px.scatter(
            plot_df,
            x="total_tokens",
            y=y_axis,
            color="method",
            symbol=symbol_col,
            hover_data=["method", "variant", "model", "total_tokens", "avg_bleu", "avg_chrf", "avg_term_success", "avg_latency", "success_rate"],
            title=f"Trade-off: {metric} vs Total Tokens ({selected_dataset} / {selected_lang})",
            size_max=15,
            height=600
        )
        
        # Improve layout
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        fig.update_layout(
            xaxis_title="Total Tokens (Cost Proxy)",
            yaxis_title=f"Average {metric} Score",
            legend_title="Method",
            font=dict(size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data Table
        st.subheader("Detailed Data")
        st.dataframe(plot_df.sort_values(by=y_axis, ascending=False))
        
    else:
        st.warning("No data available for this selection.")

if __name__ == "__main__":
    main()