import os
import streamlit as st
import requests
from io import BytesIO
import pandas as pd

# FastAPI server URL
API_URL = "http://localhost:8000"

def send_llm_prompt(prompt, filename, target_column):
    """
    Sends the user prompt to the FastAPI endpoint and returns the response.
    """
    url = f"{API_URL}/dataset/llm"
    json_data = {"prompt": prompt, "filename": filename, "target_column": target_column}
    response = requests.post(url, json=json_data)
    return response

# File upload function
def upload_file(file, target_column):
    url = f"{API_URL}/dataset/upload"
    files = {"file": file}
    data = {"target_column": target_column}  # Add target column to the request
    response = requests.post(url, files=files, data=data)  # Send target_column in the data
    return response

def clean_dataset(filename, target_class, use_gpt_eda_report):
    url = f"{API_URL}/dataset/clean"
    data = {
        "filename": filename,
        "target_class": target_class,
        "use_gpt": use_gpt_eda_report
    }
    response = requests.post(url, json=data)
    return response

# Train model function
def train_model(filename, target_column, use_gpt_model_report=False):
    url = f"{API_URL}/dataset/model"
    config = {
        "target_column": target_column,
        # The create_model_with_gpt flag in the config should match the backend's expectation
    }
    params = {"filename": filename, "use_gpt_model_report": use_gpt_model_report}
    response = requests.post(url, json=config, params=params)
    return response

# Show visualizations in a carousel
def show_visualization_carousel(viz_type, filename):
    url = f"{API_URL}/dataset/visualization/{filename}/{viz_type}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            if "links" in data and data["links"]:
                st.session_state['show_carousel'] = True
                st.session_state['carousel_links'] = data["links"]
                st.session_state['carousel_title'] = f"{viz_type.capitalize()} Visualizations"
            else:
                st.warning(f"No links found for {viz_type} visualizations.")
        except ValueError:
            st.error("Failed to decode JSON response.")
    else:
        st.error(f"Error fetching {viz_type} visualization links: {response.status_code} - {response.text}")

def show_model_visualization_carousel(viz_type, filename):
    url = f"{API_URL}/model/visualization/{filename}/{viz_type}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            if "links" in data and data["links"]:
                st.session_state['show_carousel'] = True
                st.session_state['carousel_links'] = data["links"]
                st.session_state['carousel_title'] = f"{viz_type.capitalize()} Visualizations"
            else:
                st.warning(f"No links found for {viz_type} visualizations.")
        except ValueError:
            st.error("Failed to decode JSON response.")
    else:
        st.error(f"Error fetching {viz_type} visualization links: {response.status_code} - {response.text}")

# Carousel HTML embedding
def display_carousel(links):
    if links:
        html_content = '<div style="display: flex; overflow-x: scroll;">'
        for link in links:
            html_content += f'<div style="flex-shrink: 0; margin-right: 10px;">' \
                            f'<iframe src="{API_URL+link}" width="600" height="400"></iframe></div>'
        html_content += '</div>'
        st.components.v1.html(html_content, height=500, width=700)

def main():
    # Initialize session state
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
    if 'metadata' not in st.session_state:
        st.session_state['metadata'] = None
    if 'cleaned_data' not in st.session_state:
        st.session_state['cleaned_data'] = None
    if 'show_carousel' not in st.session_state:
        st.session_state['show_carousel'] = False
    if 'carousel_links' not in st.session_state:
        st.session_state['carousel_links'] = []
    if 'carousel_title' not in st.session_state:
        st.session_state['carousel_title'] = ""
    if 'model_data' not in st.session_state:
        st.session_state['model_data'] = None

    st.title("Pocket Data Scientist Front-End")

    # Upload dataset settings
    st.write("Settings")
    generate_gpt_eda_report = st.toggle("Generate EDA Report with GPT", value=False)
    generate_gpt_model_report = st.toggle("Generate Model Report with GPT", value=False)

    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")
    target_class = st.text_input("Target column for prediction:", "")

    # Show preview of file if uploaded and target not provided
    if uploaded_file and not target_class:
        with st.spinner("Loading file preview..."):
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())

    if uploaded_file is not None and target_class:
        # Upload file and display metadata
        if uploaded_file != st.session_state['uploaded_file']:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['cleaned_data'] = None
            st.session_state['model_data'] = None
            with st.spinner("Uploading file..."):
                response = upload_file(uploaded_file, target_class)
            if response.status_code == 200:
                st.session_state['metadata'] = response.json()
                st.write("✅ File uploaded successfully!")
            else:
                st.error(f"Error uploading file: {response.status_code} - {response.text}")
                st.session_state['metadata'] = None
        elif st.session_state['metadata']:
            st.write("Dataset Metadata:")
            st.json({
                key: (
                    {sub_key: sub_value for sub_key, sub_value in value.items() if sub_key != "missing_values"}
                    if isinstance(value, dict) else value
                )
                for key, value in st.session_state['metadata'].items() if key != "preview"
            })

        if st.session_state['metadata']:
            filename = st.session_state['metadata']["filename"]

            # === ORIGINAL DATASET PREVIEW AFTER UPLOAD ===
            st.header("📄 Uploaded Dataset Preview")
            df = pd.read_csv(f"datasets/{filename}")
            st.dataframe(df.head())

            # === CLEANING SECTION ===
            st.header("🧹 Clean Dataset")
            if st.button("Clean Dataset"):
                with st.spinner("Cleaning dataset..."):
                    clean_response = clean_dataset(filename, target_class, generate_gpt_eda_report)
                if clean_response.status_code == 200:
                    st.session_state['cleaned_data'] = clean_response.json()

                    # Accessing the correct response attributes after cleaning
                    results = clean_response.json().get('results', {})
                    column_types = results.get('column_types', {})
                    missing_values = results.get('missing_values', {})
                    outlier_counts = results.get('outlier_counts', {})

                    if column_types:
                        st.subheader("Column Types")
                        st.json(column_types)

                    if missing_values:          
                        st.subheader("Values Replaced/Dropped")
                        st.json(missing_values)

                    if outlier_counts: 
                        st.subheader("Outliers Replaced/Dropped")
                        st.json(outlier_counts)
                else:
                    st.error(f"Error cleaning dataset: {clean_response.status_code} - {clean_response.text}")

            # === PREVIEW OF CLEANED/TRAINING DATASET ===
            if not st.session_state['cleaned_data'] == None:
                cleaned_dataset_path = st.session_state['cleaned_data'].get("cleaned_dataset_path", "")
                if cleaned_dataset_path:
                    df = pd.read_csv(cleaned_dataset_path)
                    csv_data = df.to_csv(index=False)
                    
                    st.header("🧪 Cleaned Dataset")
                    st.dataframe(df)

                    st.download_button(
                    label="Download Cleaned Dataset",
                    data=csv_data,
                    file_name="cleaned_dataset.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No cleaned dataset path available.")

            # === EDA Report & Visuals ===
            if st.session_state.get('cleaned_data'):
                st.header("📊 EDA Report & Visualizations")
                report_path = st.session_state['cleaned_data'].get("report_path")
                if report_path and os.path.exists(report_path):
                    try:
                        with open(report_path, "rb") as file:
                            st.download_button(
                                label="Download EDA Report",
                                data=file.read(),
                                file_name="eda_report.pdf",
                                mime="application/pdf",
                                key="eda_report_download_button"
                            )
                    except Exception as e:
                        st.error(f"Error reading the report file: {e}")
                else:
                    st.warning("EDA report not found. Please generate the report first.")

                visualization_type = st.selectbox("Choose Visualization Type", ["correlation", "distribution", "boxplot", "correlation heatmap", "all"])
                if st.button(f"Show {visualization_type} visualizations"):
                    with st.spinner("Loading visualizations..."):
                        show_visualization_carousel(visualization_type, filename)

            # === MODEL TRAINING ===
    if st.session_state.get('cleaned_data'):
        st.header("🧠 Train Model")

        # === Show training data preview before training ===
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                train_response = train_model(filename, target_class, use_gpt_model_report=generate_gpt_model_report)
            if train_response.status_code == 200:
                st.session_state['model_data'] = train_response.json()
                st.success(st.session_state['model_data']["message"])

                st.write("Best Model:", st.session_state['model_data'].get("best_model_name", "N/A"))
                st.write("Best Score:", st.session_state['model_data'].get("best_score", "N/A"))
                st.write(f"Columns Removed: {st.session_state['model_data'].get('columns_removed', '')}")
                st.write("Tuning Results:")
                st.json(st.session_state['model_data'].get("tuning_results", {}))
                

                report_path = st.session_state['model_data'].get("report_path")
                if report_path and os.path.exists(report_path):
                    try:
                        with open(report_path, "rb") as file:
                            st.download_button(
                                label="Download Model Report",
                                data=file.read(),
                                file_name="model_report.pdf",
                                mime="application/pdf",
                                key="model_report_download_button"
                            )
                    except Exception as e:
                        st.error(f"Error reading the report file: {e}")
                else:
                    st.warning("Model report not found.")
            else:
                st.error(f"Error training model: {train_response.status_code} - {train_response.text}")

        # --- New Section: Download Trained Model ---
        # Only show if model_data exists in session state
        if "model_data" in st.session_state and st.session_state["model_data"]:
            st.header("Download Trained Model")
            try:
                model_file_url = f"{API_URL}/model/download/{st.session_state['metadata']['filename']}"
                model_file_response = requests.get(model_file_url)
                if model_file_response.status_code == 200:
                    model_file_name = os.path.basename(st.session_state['model_data']["model_path"])
                    st.download_button(
                        label="Download Trained Model",
                        data=model_file_response.content,
                        file_name=model_file_name,
                        mime="application/octet-stream",
                        key="model_file_download_button"
                    )
                else:
                    st.warning("Trained model file is not available for download.")
            except Exception as e:
                st.error(f"Error downloading model file: {e}")

        # === Model Visualizations ===
        if st.session_state.get('model_data'):
            st.header("🖼️ Model Visualizations")
            model_viz_type = st.selectbox("Choose Model Visualization Type", [
                "actual vs predicted", "residuals", "feature importance", "roc curve", "confusion matrix", "correlation heatmap"
            ])
            if st.button(f"Show {model_viz_type}"):
                with st.spinner("Loading model visualizations..."):
                    show_model_visualization_carousel(model_viz_type, filename)
    
    # --- New Section: DatasetLLM Chat ---
    if st.session_state['metadata']:
        if filename:
            st.header("💬 DatasetLLM Chat")
            llm_prompt = st.text_area("Enter your prompt for the LLM", height=100)
            if st.button("Send Prompt"):
                if llm_prompt:
                    with st.spinner("Waiting for LLM response..."):
                        llm_response = send_llm_prompt(llm_prompt, filename, target_class)
                    if llm_response.status_code == 200:
                        data = llm_response.json()
                        st.write("#### LLM Response")
                        st.write(data["response"])
                    else:
                        st.error(f"Error: {llm_response.status_code} - {llm_response.text}")
                else:
                    st.warning("Please enter a prompt before sending.")

    # === Visualization Carousel ===
    if st.session_state['show_carousel']:
        st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
        st.header("🖼️ Visualization Carousel")
        with st.container():
            col1, col2, _ = st.columns([1, 10, 1])
            with col1:
                if st.button("X"):
                    st.session_state['show_carousel'] = False
            with col2:
                st.subheader(st.session_state['carousel_title'])

            display_carousel(st.session_state['carousel_links'])
    elif uploaded_file is not None and not target_class:
        st.warning("Please provide the target column name.")

if __name__ == "__main__":
    main()
