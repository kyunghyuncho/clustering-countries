# clustering-countries

This project is an interactive Streamlit web application that clusters the world's countries based on demographic and economic statistics using K-Means and Gaussian Mixture Models (GMM) with a strictly enforced uniform prior. The data is fetched live from the World Bank API.

## Setup & Running

This project uses `uv` for package management.

1.  Make sure you have `uv` installed.
2.  Install dependencies and activate the virtual environment:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    uv run streamlit run app.py
    ```