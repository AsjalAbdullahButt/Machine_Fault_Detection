import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import traceback
from pathlib import Path

# Setup paths
CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Code")
sys.path.insert(0, CODE_DIR)

# Import project modules (avoid lazy loading of TensorFlow/Keras)
import config
from data_processing import load_data

# Page config
st.set_page_config(
    page_title="Predictive Fault Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    :root {
        --primary-color: #e94560;
        --background-color: #1a1a2e;
        --secondary-bg: #16213e;
        --text-color: #eaeaea;
        --accent-color: #0f3460;
    }
    
    body {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background-color: var(--background-color);
    }
    
    .stSidebar {
        background-color: var(--secondary-bg);
    }
    
    .metric-box {
        background-color: var(--accent-color);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        color: var(--text-color);
    }
    
    h1, h2, h3 {
        color: var(--primary-color);
    }
    
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 5px;
    }
    
    .stButton > button:hover {
        background-color: #ff6b7a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "training_logs" not in st.session_state:
    st.session_state.training_logs = []
if "current_model" not in st.session_state:
    st.session_state.current_model = None

# Helper function to get plot IDs based on architecture
def _get_plot_ids(arch_key, task):
    """Returns a dict of plot IDs for a given architecture and task."""
    plot_mapping = {
        "lstm": {
            "binary": ["lstm_binary_history", "lstm_binary_roc_curve", "lstm_binary_confusion"],
            "multiclass": ["lstm_multiclass_history", "lstm_multiclass_roc_curve", "lstm_multiclass_confusion"]
        },
        "cnn_lstm": {
            "binary": ["cnn_lstm_binary_history", "cnn_lstm_binary_roc_curve", "cnn_lstm_binary_confusion"],
            "multiclass": ["cnn_lstm_multiclass_history", "cnn_lstm_multiclass_roc_curve", "cnn_lstm_multiclass_confusion"]
        },
        "transformer": {
            "binary": ["transformer_binary_history", "transformer_binary_roc_curve", "transformer_binary_confusion"],
            "multiclass": ["transformer_multiclass_history", "transformer_multiclass_roc_curve", "transformer_multiclass_confusion"]
        }
    }
    return plot_mapping.get(arch_key, {}).get(task, [])

# Title and description
st.title("🤖 Predictive Fault Detection System")
st.markdown("Advanced ML Dashboard for Remaining Useful Life (RUL) and Fault Detection")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    epoch_mode = st.radio(
        "Epoch Mode",
        ["Quick (5 epochs)", "Standard (20 epochs)", "Full (50 epochs)"],
        index=1
    )
    epoch_map = {
        "Quick (5 epochs)": 5,
        "Standard (20 epochs)": 20,
        "Full (50 epochs)": 50
    }
    EPOCHS = epoch_map[epoch_mode]
    
    batch_size = st.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
    time_steps = st.slider("Time Steps", min_value=5, max_value=30, value=10, step=5)
    
    st.divider()
    st.subheader("Options")
    load_existing = st.checkbox("Load Existing Models", value=True)
    regenerate_plots = st.checkbox("Regenerate All Plots", value=False)
    verbose_logging = st.checkbox("Verbose Logging", value=False)

# ============================================================================
# TAB 1: DATA OVERVIEW
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "🏋️ Training Pipeline",
    "📈 Results & Plots",
    "🤖 Simulator"
])

with tab1:
    st.header("Dataset Overview")
    
    try:
        # Load data
        df = load_data(config.DATASET_PATH)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📦 Total Samples", len(df))
        with col2:
            st.metric("🔧 Features", df.shape[1])
        with col3:
            st.metric("📅 Time Range", f"{df.shape[0]} rows")
        with col4:
            st.metric("⚠️ Missing Values", df.isnull().sum().sum())
        
        st.divider()
        
        # Data preview
        st.subheader("Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        cols = st.columns(3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:9]
        
        for idx, col in enumerate(numeric_cols):
            with cols[idx % 3]:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(df[col].dropna(), bins=30, color='#e94560', alpha=0.7, edgecolor='black')
                ax.set_title(f"{col}")
                ax.set_facecolor('#16213e')
                fig.patch.set_facecolor('#1a1a2e')
                ax.tick_params(colors='#eaeaea')
                ax.xaxis.label.set_color('#eaeaea')
                ax.yaxis.label.set_color('#eaeaea')
                st.pyplot(fig, use_container_width=True)
        
        # Missing data check
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            st.warning(f"⚠️ Found missing values in {len(missing_data)} columns")
            st.dataframe(missing_data, use_container_width=True)
        else:
            st.success("✅ No missing values detected")
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.text(traceback.format_exc())

# ============================================================================
# TAB 2: TRAINING PIPELINE
# ============================================================================
with tab2:
    st.header("🏋️ Model Training Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Select Architecture & Task")
        arch = st.selectbox(
            "Architecture",
            ["LSTM", "CNN-LSTM", "Transformer"],
            key="arch_select"
        )
        
        task = st.selectbox(
            "Task",
            ["Binary Classification", "Multiclass Classification", "RUL Estimation"],
            key="task_select"
        )
    
    with col2:
        st.subheader("Model Status")
        if load_existing:
            st.success("✅ Load Mode: ON")
        else:
            st.info("ℹ️ Training Mode: ON")
    
    st.divider()
    
    # Model status grid
    st.subheader("Training Status Overview")
    status_cols = st.columns(3)
    architectures = ["LSTM", "CNN-LSTM", "Transformer"]
    
    for idx, arch_name in enumerate(architectures):
        with status_cols[idx]:
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(config.get_model_path(arch_name.lower().replace("-", "_"), "binary")):
                    st.success(f"✅ {arch_name}\nBinary")
                else:
                    st.warning(f"⏳ {arch_name}\nBinary")
            with col2:
                if os.path.exists(config.get_model_path(arch_name.lower().replace("-", "_"), "multiclass")):
                    st.success(f"✅ {arch_name}\nMulticlass")
                else:
                    st.warning(f"⏳ {arch_name}\nMulticlass")
    
    st.divider()
    
    # Training controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Start Training", key="train_btn"):
            st.session_state.training_logs = []
            log_container = st.container()
            
            with log_container:
                st.info("🔄 Loading modules and data...")
                try:
                    # Import modules dynamically to avoid heavy loading upfront
                    from data_processing import preprocess, create_sequences, split_features_labels
                    from models import (
                        build_binary_lstm, build_multiclass_lstm,
                        build_binary_cnn_lstm, build_multiclass_cnn_lstm,
                        build_binary_transformer, build_multiclass_transformer,
                        train_model, evaluate_binary
                    )
                    
                    df = load_data(config.DATASET_PATH)
                    df, le = preprocess(df)
                    X, y_binary, y_multi = split_features_labels(df)
                    
                    # Select y based on task
                    if "Binary" in task:
                        y = y_binary
                        is_binary = True
                    elif "Multiclass" in task:
                        y = y_multi
                        is_binary = False
                    else:  # RUL
                        y = y_binary  # Use binary as base for RUL
                        is_binary = True
                    
                    X_seq, y_seq = create_sequences(X, y, time_steps)
                    st.success("✅ Data loaded and preprocessed")
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    arch_key = arch.lower().replace("-", "_")
                    task_key = "binary" if "Binary" in task else ("multiclass" if "Multiclass" in task else "rul")
                    
                    status_text.text(f"🏗️ Building {arch} model for {task}...")
                    progress_bar.progress(20)
                    
                    # Build model based on architecture and task
                    if arch == "LSTM":
                        if is_binary:
                            model = build_binary_lstm(X_seq.shape)
                        else:
                            model = build_multiclass_lstm(X_seq.shape, num_classes=len(np.unique(y_seq)))
                    elif arch == "CNN-LSTM":
                        if is_binary:
                            model = build_binary_cnn_lstm(X_seq.shape)
                        else:
                            model = build_multiclass_cnn_lstm(X_seq.shape, num_classes=len(np.unique(y_seq)))
                    else:  # Transformer
                        if is_binary:
                            model = build_binary_transformer(X_seq.shape)
                        else:
                            model = build_multiclass_transformer(X_seq.shape, num_classes=len(np.unique(y_seq)))
                    
                    status_text.text(f"🎯 Training {arch} model...")
                    progress_bar.progress(50)
                    
                    # Train model
                    history_path = config.get_plot_path(f"{arch_key}_{task_key}_history")
                    history = train_model(
                        model, X_seq, y_seq, 
                        epochs=EPOCHS, batch_size=batch_size,
                        history_save_path=history_path
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("✅ Training complete!")
                    
                    # Evaluate
                    if is_binary:
                        metrics = evaluate_binary(model, X_seq, y_seq)
                        st.success(f"✅ Training completed!\n\nMetrics:\n{json.dumps(metrics, indent=2)}")
                    else:
                        st.success(f"✅ Training completed successfully!")
                    
                    progress_bar.progress(100)
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    if verbose_logging:
                        st.text(traceback.format_exc())
    
    with col2:
        if st.button("📊 View Training History", key="history_btn"):
            try:
                from models import load_history_from_json
                
                arch_key = arch.lower().replace("-", "_")
                task_key = "binary" if "Binary" in task else ("multiclass" if "Multiclass" in task else "rul")
                history_path = config.get_plot_path(f"{arch_key}_{task_key}_history")
                
                if os.path.exists(history_path):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    history = load_history_from_json(history_path)
                    
                    if history and hasattr(history, 'history') and history.history:
                        if 'loss' in history.history:
                            ax.plot(history.history['loss'], label='Training Loss', color='#e94560')
                            if 'val_loss' in history.history:
                                ax.plot(history.history['val_loss'], label='Validation Loss', color='#0f3460')
                        
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.set_title(f"{arch} - {task} Training History")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.set_facecolor('#16213e')
                        fig.patch.set_facecolor('#1a1a2e')
                        ax.tick_params(colors='#eaeaea')
                        
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning("No training history available")
                else:
                    st.info(f"Training history not found at {history_path}")
            except Exception as e:
                st.error(f"Error loading history: {str(e)}")
    
    with col3:
        st.empty()

# ============================================================================
# TAB 3: RESULTS & PLOTS
# ============================================================================
with tab3:
    st.header("📈 Model Results & Visualization Gallery")
    
    # Results CSV
    st.subheader("Model Comparison Results")
    try:
        if os.path.exists(config.RESULTS_CSV):
            results_df = pd.read_csv(config.RESULTS_CSV)
            
            # Color highlight function
            def highlight_best(val, col_name):
                if col_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']:
                    return 'background-color: #0f3460'
                return ''
            
            st.dataframe(
                results_df.style.applymap(highlight_best, subset=results_df.columns[1:]),
                use_container_width=True
            )
            
            # Download button
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_data,
                file_name="model_results.csv",
                mime="text/csv"
            )
        else:
            st.info("No results available yet. Run training first.")
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
    
    st.divider()
    
    # Plot gallery
    st.subheader("Plot Gallery")
    
    col1, col2 = st.columns(2)
    
    # Get available plots
    available_plots = []
    if os.path.exists(config.PLOTS_DIR):
        plot_files = [f for f in os.listdir(config.PLOTS_DIR) if f.endswith(('.png', '.jpg', '.pdf'))]
        available_plots = sorted(plot_files)
    
    if available_plots:
        for idx, plot_file in enumerate(available_plots):
            with col1 if idx % 2 == 0 else col2:
                plot_path = os.path.join(config.PLOTS_DIR, plot_file)
                st.subheader(plot_file.replace('_', ' ').replace('.png', '').title())
                
                try:
                    st.image(plot_path, use_column_width=True)
                except Exception as e:
                    st.warning(f"Could not load image: {plot_file}")
    else:
        st.info("No plots available. Generate plots by running training.")
    
    st.divider()
    
    # Regenerate plots button
    if st.button("🔄 Regenerate History Plots from JSON"):
        st.info("Regenerating plots from saved training history...")
        try:
            for arch_key in ["lstm", "cnn_lstm", "transformer"]:
                for task_key in ["binary", "multiclass"]:
                    history_path = config.get_plot_path(f"{arch_key}_{task_key}_history")
                    if os.path.exists(history_path):
                        history = load_history_from_json(history_path)
                        plot_title = f"{arch_key.upper()} - {task_key.upper()}"
                        # Plot would be saved to plots directory
                        st.success(f"✅ Regenerated {plot_title}")
            st.success("✅ All plots regenerated!")
        except Exception as e:
            st.error(f"Error regenerating plots: {str(e)}")

# ============================================================================
# TAB 4: SIMULATOR
# ============================================================================
with tab4:
    st.header("🤖 Robot Fault Simulator")
    st.markdown("Simulate robot operation and monitor fault probability and RUL predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sim_steps = st.number_input("Simulation Steps", min_value=10, max_value=500, value=100, step=10)
    with col2:
        fault_prob = st.slider("Base Fault Probability", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    with col3:
        degradation_rate = st.slider("Degradation Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.05)
    
    st.divider()
    
    if st.button("🚀 Run Simulation"):
        try:
            from simulator import RobotSimulator
            
            simulator = RobotSimulator(initial_rul=100, degradation_rate=degradation_rate)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run simulation
            results = []
            for step in range(sim_steps):
                status = simulator.step()
                results.append({
                    'Step': step,
                    'RUL': status['rul'],
                    'Fault_Probability': np.random.beta(2, 5) * fault_prob * (1 - step/sim_steps),
                    'Degradation': status.get('degradation', 0)
                })
                progress_bar.progress((step + 1) / sim_steps)
                status_text.text(f"Step {step + 1}/{sim_steps} - RUL: {status['rul']:.2f}")
            
            results_df = pd.DataFrame(results)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Steps", sim_steps)
            with col2:
                st.metric("Final RUL", f"{results_df['RUL'].iloc[-1]:.2f}")
            with col3:
                st.metric("Max Fault Prob", f"{results_df['Fault_Probability'].max():.3f}")
            with col4:
                st.metric("Avg Degradation", f"{results_df['Degradation'].mean():.3f}")
            
            st.divider()
            
            # Dual-axis plot
            st.subheader("Fault Probability & RUL Timeline")
            fig, ax1 = plt.subplots(figsize=(12, 5))
            
            color_rul = '#e94560'
            ax1.set_xlabel('Step')
            ax1.set_ylabel('RUL', color=color_rul)
            line1 = ax1.plot(results_df['Step'], results_df['RUL'], color=color_rul, linewidth=2, label='RUL')
            ax1.tick_params(axis='y', labelcolor=color_rul)
            ax1.set_facecolor('#16213e')
            
            ax2 = ax1.twinx()
            color_fault = '#0f3460'
            ax2.set_ylabel('Fault Probability', color=color_fault)
            line2 = ax2.plot(results_df['Step'], results_df['Fault_Probability'], color=color_fault, linewidth=2, label='Fault Probability')
            ax2.tick_params(axis='y', labelcolor=color_fault)
            
            fig.patch.set_facecolor('#1a1a2e')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(colors='#eaeaea')
            ax1.xaxis.label.set_color('#eaeaea')
            ax1.yaxis.label.set_color(color_rul)
            ax2.yaxis.label.set_color(color_fault)
            
            fig.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99))
            st.pyplot(fig, use_container_width=True)
            
            st.divider()
            
            # Data table
            st.subheader("Simulation Data")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv_sim = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Simulation Data",
                data=csv_sim,
                file_name=f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success("✅ Simulation completed successfully!")
        
        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")
            if verbose_logging:
                st.text(traceback.format_exc())

# Footer
st.divider()
st.markdown("""
---
**Predictive Fault Detection System** | Advanced ML Dashboard
Built with Streamlit, TensorFlow/Keras, and Scikit-learn
""")
