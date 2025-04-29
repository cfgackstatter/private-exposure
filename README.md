# Private Exposure

A web application and toolkit for exploring and optimizing exposure to private companies (e.g., SpaceX) via public mutual funds.

## Features

- **Find funds with exposure to any private company** (e.g., "SpaceX") using SEC filings
- **Aggregate and visualize exposures** across funds
- **Update and store historical fund compositions** as Parquet files
- **Modern Dash webapp** with Bootstrap styling
- **(Coming soon)**: Compute optimal long/short portfolios to maximize pure private company exposure

## Setup

1. **Clone the repo**
    ```
    git clone https://github.com/cfgackstatter/private-exposure.git
    cd private-exposure
    ```

2. **Create and activate a virtual environment**
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

4. **Run the webapp**
    ```
    python app.py
    ```

5. **(Optional) Update fund compositions from the CLI**
    ```
    python main.py
    ```

## Usage

- Use the webapp to search for funds with exposure to your target company.
- Update fund data as needed.
- Explore aggregated exposures and (soon) optimal portfolio suggestions.

## License

MIT License