import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def parquet_path(ticker: str) -> Path:
    """
    Return the Path object for the parquet file of a given ticker.

    Args:
        ticker: Fund ticker symbol.

    Returns:
        Path to the parquet file.
    """
    return DATA_DIR / f"{ticker.upper()}.parquet"


def clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe columns to ensure compatibility with pyarrow parquet saving.
    Specifically, convert 'N/A' strings and other non-numeric placeholders in numeric columns to NaN.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df_clean = df.copy()
    # List of columns that should be numeric (add more as needed)
    numeric_cols = ['balance', 'valUSD', 'pctVal', 'units']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    return df_clean


def ensure_fund_name_populated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that the 'fund_name' column is populated in the DataFrame.
    If 'fund_name' is missing or null, copy values from 'series_name' if available.

    Args:
        df: DataFrame containing fund composition data.

    Returns:
        DataFrame with 'fund_name' populated.
    """
    df = df.copy()
    if 'fund_name' not in df.columns and 'series_name' in df.columns:
        df['fund_name'] = df['series_name']
    elif 'fund_name' in df.columns and 'series_name' in df.columns:
        # Fill missing fund_name values with series_name
        df['fund_name'] = df['fund_name'].fillna(df['series_name'])
    elif 'fund_name' not in df.columns:
        # If fund_name is missing and series_name is also missing, create fund_name as None
        df['fund_name'] = None
    return df


def update_fund_compositions(ticker: str) -> Optional[str]:
    """
    Download and update fund compositions for a given ticker.
    Stores new compositions as parquet files in the data directory.

    Args:
        ticker: Fund ticker symbol.

    Returns:
        Status message or None.
    """
    logging.info(f"Updating compositions for ticker: {ticker}")
    fund_info = get_fund_identifiers(ticker)
    if fund_info:
        class_id = fund_info['class_id']
        filings_df = get_nport_filings(class_id)
    else:
        logging.info(f"Fund identifiers not found for {ticker}. Trying fallback SEC API search.")
        filings_df = get_nport_filings_from_sec_api(ticker)
        # Fallback fund_info: fill with None except ticker
        fund_info = {
            'ticker': ticker.upper(),
            'fund_name': None,
            'series_name': None,
            'class_name': None,
            'series_id': None,
            'class_id': None,
            'cik': None
        }

    if filings_df is None or filings_df.empty:
        logging.warning("No filings found.")
        return "No filings found."
    
    pq_path = parquet_path(ticker)
    existing_dates = set()
    if pq_path.exists():
        existing_df = pd.read_parquet(pq_path)
        existing_dates = set(existing_df['reporting_date'].unique())
    new_compositions = []

    for _, row in filings_df.iterrows():
        url = row['filing_url']
        fund_info_row = fund_info.copy() if fund_info else {}
        # Fill from row if present
        for col in ['fund_name', 'ticker', 'cik', 'filing_date', 'reporting_date', 'series_name', 'class_name']:
            if col in row:
                fund_info_row[col] = row[col]
            elif col == 'filing_date' and 'Filing Date' in row:
                fund_info_row['filing_date'] = row['Filing Date']
            elif col == 'reporting_date' and 'Reporting for' in row:
                fund_info_row['reporting_date'] = row['Reporting for']

        comp_df = extract_composition_from_filing(url, fund_info_row)
        if comp_df is not None and not comp_df.empty:
            rep_date = comp_df['reporting_date'].iloc[0]
            if rep_date not in existing_dates:
                new_compositions.append(comp_df)

    if new_compositions:
        all_new = pd.concat(new_compositions)
        # Ensure fund_name is populated
        all_new = ensure_fund_name_populated(all_new)
        all_new = clean_dataframe_for_parquet(all_new)
        if pq_path.exists():
            all_new = pd.concat([existing_df, all_new])
            all_new = clean_dataframe_for_parquet(all_new)
        all_new.to_parquet(pq_path, index=False)
        logging.info(f"Downloaded and stored {len(new_compositions)} new compositions.")
        return f"Downloaded and stored {len(new_compositions)} new compositions."
    else:
        logging.info("No new compositions found.")
        return "No new compositions found."


def search_compositions(search_string: str) -> pd.DataFrame:
    """
    Search for holdings containing a specific string in the most recent filings of all funds.

    Args:
        search_string: Substring to search for in holding names.

    Returns:
        DataFrame of matching holdings.
    """
    logging.info(f"Searching for holdings containing: {search_string}")
    results = []
    for pq_file in DATA_DIR.glob("*.parquet"):
        df = pd.read_parquet(pq_file)
        most_recent = df['reporting_date'].max()
        df_latest = df[df['reporting_date'] == most_recent]
        mask = df_latest['name'].str.contains(search_string, case=False, na=False)
        if mask.any():
            results.append(df_latest[mask])
    if results:
        logging.info(f"Found {sum(len(r) for r in results)} matching holdings.")
        return pd.concat(results)
    else:
        logging.info("No results found.")
        return pd.DataFrame()
    
    
def get_fund_identifiers(ticker: str) -> Optional[Dict[str, str]]:
    """
    Retrieve CIK, Series ID, and Class/Contract ID for a mutual fund by ticker symbol.

    Args:
        ticker: The ticker symbol of the mutual fund (e.g., "BFGFX").

    Returns:
        Dictionary containing fund identifiers or None if not found.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url = f"https://www.sec.gov/cgi-bin/series?ticker={ticker.lower()}&sc=companyseries"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        target_table = None
        for table in tables:
            text = table.get_text(separator='|', strip=True)
            if 'CIK|Series|Ticker|Class/Contract' in text:
                target_table = table
                break
        if not target_table:
            logging.warning(f"No fund information found for ticker {ticker}")
            return None
        text = target_table.get_text(separator='|', strip=True)
        parts = text.split('|')
        cik_index = parts.index('CIK') if 'CIK' in parts else -1
        if cik_index == -1 or len(parts) < cik_index + 12:
            logging.warning(f"Insufficient information found for ticker {ticker}")
            return None
        fund_info = {
            'cik': parts[cik_index + 6],
            'series_id': parts[cik_index + 8],
            'class_id': parts[cik_index + 10],
            'fund_name': parts[cik_index + 7],
            'series_name': parts[cik_index + 9],
            'class_name': parts[cik_index + 11],
            'ticker': ticker.upper()
        }
        logging.info(f"Identifiers for {ticker}: {fund_info}")
        return fund_info
    except Exception as e:
        logging.error(f"Error retrieving fund identifiers for {ticker}: {e}")
        return None
    

def get_nport_filings(class_cik: str) -> Optional[pd.DataFrame]:
    """
    Extract NPORT-P filings table for a fund class from SEC EDGAR.

    Args:
        class_cik: The CIK of the fund class (e.g., "C000065146").

    Returns:
        DataFrame of NPORT-P filings with fund information or None if not found.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={class_cik}&action=getcompany&scd=filings&type=NPORT-P"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        fund_info = {}
        ticker_match = re.search(r'Ticker Symbol:\s*\*\*([A-Z]+)\*\*', str(soup))
        if ticker_match:
            fund_info['ticker'] = ticker_match.group(1)
        class_match = re.search(r'Class/Contract:\s*\*\*([^*]+)\*\*', str(soup))
        if class_match:
            fund_info['share_class'] = class_match.group(1).strip()
        fund_name_match = re.search(r'for ([^)]+)\)', str(soup))
        if fund_name_match:
            fund_info['parent_fund'] = fund_name_match.group(1)
            series_match = re.search(r'Series:\s*\*\*([^*]+)\*\*', str(soup))
            if series_match:
                fund_info['fund_name'] = series_match.group(1).strip()
            else:
                fund_info['fund_name'] = None
        table = soup.find('table', {'class': 'tableFile2'})
        if not table:
            logging.warning(f"No filings table found for {class_cik}")
            return None
        rows = []
        headers_list = [th.text.strip() for th in table.find_all('th')]
        for tr in table.find_all('tr')[1:]:
            cells = tr.find_all('td')
            if len(cells) >= len(headers_list):
                row_data = {}
                row_data.update(fund_info)
                row_data['class_cik'] = class_cik
                for i, cell in enumerate(cells):
                    row_data[headers_list[i]] = cell.text.strip()
                    if headers_list[i] == 'Description':
                        acc_match = re.search(r'Acc-no: (\d{10}-\d{2}-\d{6})', cell.text)
                        if acc_match:
                            row_data['accession_number'] = acc_match.group(1)
                if 'accession_number' in row_data:
                    acc_num = row_data['accession_number']
                    acc_num_no_dashes = acc_num.replace('-', '')
                    cik_match = re.search(r'CIK=(\d+)', str(soup))
                    parent_cik = cik_match.group(1) if cik_match else "1217673"
                    row_data['filing_url'] = f"https://www.sec.gov/Archives/edgar/data/{parent_cik}/{acc_num_no_dashes}/{acc_num}-index.htm"
                rows.append(row_data)
        df = pd.DataFrame(rows)
        if df.empty:
            logging.warning(f"No NPORT-P filings found for {class_cik}")
            return None
        cols = ['fund_name', 'ticker', 'class_cik', 'share_class', 'parent_fund',
                'Filings', 'Filing Date', 'accession_number', 'filing_url', 'Description']
        existing_cols = [col for col in cols if col in df.columns]
        return df[existing_cols]
    except Exception as e:
        logging.error(f"Error retrieving NPORT-P filings for {class_cik}: {e}")
        return None
      

def extract_composition_from_filing(
        filing_url: str,
        fund_info: Optional[Dict[str, str]] = None
) -> Optional[pd.DataFrame]:
    """
    Extract fund composition from an NPORT-P filing URL (either HTML index or direct XML).

    Args:
        filing_url: URL to the SEC filing.
        fund_info: Optional dictionary with fund identifiers to add to the output.

    Returns:
        DataFrame containing holdings with identifiers, dates, and holdings data.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        if filing_url.endswith('.xml'):
            # Direct XML: parse as XML
            response = requests.get(filing_url, headers=headers)
            response.raise_for_status()
            xml_soup = BeautifulSoup(response.content, 'lxml-xml')
            reporting_date = None
            filing_date = None
            period_tag = xml_soup.find(['periodOfReport', 'period_end', 'period_ending'])
            if period_tag:
                reporting_date = period_tag.text.strip()
            filed_tag = xml_soup.find(['filedAt', 'file_date'])
            if filed_tag:
                filing_date = filed_tag.text.strip()
            if not filing_date and fund_info and 'filing_date' in fund_info:
                filing_date = fund_info['filing_date']
            if not reporting_date and fund_info and 'reporting_date' in fund_info:
                reporting_date = fund_info['reporting_date']
            holdings = []
            for inv in xml_soup.find_all('invstOrSec'):
                holding_data = {}
                for field in ['name', 'lei', 'title', 'cusip', 'balance', 'units',
                              'curCd', 'valUSD', 'pctVal', 'payoffProfile', 'assetCat',
                              'issuerCat', 'invCountry', 'fairValLevel']:
                    elem = inv.find(field)
                    holding_data[field] = elem.text if elem else None
                identifiers = inv.find('identifiers')
                if identifiers:
                    for id_type in ['isin', 'cusip', 'ticker']:
                        id_elem = identifiers.find(id_type)
                        if id_elem:
                            holding_data[id_type] = id_elem.get('value')
                debt_sec = inv.find('debtSec')
                if debt_sec:
                    for field in ['maturityDt', 'couponKind', 'annualizedRt']:
                        elem = debt_sec.find(field)
                        holding_data[field] = elem.text if elem else None
                for field in ['valUSD', 'pctVal', 'balance']:
                    if field in holding_data and holding_data[field]:
                        try:
                            holding_data[field] = float(holding_data[field])
                        except (ValueError, TypeError):
                            pass
                holdings.append(holding_data)
            if not holdings:
                logging.warning(f"No holdings found in {filing_url}")
                return None
            df = pd.DataFrame(holdings)
            df['filing_date'] = filing_date
            df['reporting_date'] = reporting_date
            # Add fund identifiers if provided
            if fund_info:
                for key, value in fund_info.items():
                    df[key] = value
                for col in ['series_name', 'class_name']:
                    if col not in df.columns and col in fund_info:
                        df[col] = fund_info[col]
            id_cols = ['filing_date', 'reporting_date', 'ticker', 'fund_name', 'series_id',
                       'series_name', 'class_id', 'class_name', 'cik']
            existing_id_cols = [col for col in id_cols if col in df.columns]
            other_cols = [col for col in df.columns if col not in existing_id_cols]
            return df[existing_id_cols + other_cols]
        else:
            # HTML index: original logic
            response = requests.get(filing_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            filing_date = reporting_date = None
            for info_head in soup.find_all('div', {'class': 'infoHead'}):
                if 'Filing Date' in info_head.text:
                    info_div = info_head.find_next_sibling('div', {'class': 'info'})
                    if info_div:
                        filing_date = info_div.text.strip()
                if 'Period of Report' in info_head.text:
                    info_div = info_head.find_next_sibling('div', {'class': 'info'})
                    if info_div:
                        reporting_date = info_div.text.strip()
            doc_link = None
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        cell_text = cells[2].text.lower() if len(cells) > 2 else ""
                        if 'primary_doc.xml' in cell_text and cells[2].find('a'):
                            doc_link = 'https://www.sec.gov' + cells[2].find('a')['href']
                            break
            if not doc_link:
                logging.warning(f"No primary_doc.xml found in {filing_url}")
                return None
            logging.info(f"Downloading XML from: {doc_link}")
            doc_response = requests.get(doc_link, headers=headers)
            doc_response.raise_for_status()
            xml_soup = BeautifulSoup(doc_response.content, 'lxml-xml')
            holdings = []
            for inv in xml_soup.find_all('invstOrSec'):
                holding_data = {}
                for field in ['name', 'lei', 'title', 'cusip', 'balance', 'units',
                              'curCd', 'valUSD', 'pctVal', 'payoffProfile', 'assetCat',
                              'issuerCat', 'invCountry', 'fairValLevel']:
                    elem = inv.find(field)
                    holding_data[field] = elem.text if elem else None
                identifiers = inv.find('identifiers')
                if identifiers:
                    for id_type in ['isin', 'cusip', 'ticker']:
                        id_elem = identifiers.find(id_type)
                        if id_elem:
                            holding_data[id_type] = id_elem.get('value')
                debt_sec = inv.find('debtSec')
                if debt_sec:
                    for field in ['maturityDt', 'couponKind', 'annualizedRt']:
                        elem = debt_sec.find(field)
                        holding_data[field] = elem.text if elem else None
                for field in ['valUSD', 'pctVal', 'balance']:
                    if field in holding_data and holding_data[field]:
                        try:
                            holding_data[field] = float(holding_data[field])
                        except (ValueError, TypeError):
                            pass
                holdings.append(holding_data)
            if not holdings:
                logging.warning(f"No holdings found in {doc_link}")
                return None
            df = pd.DataFrame(holdings)
            df['filing_date'] = filing_date
            df['reporting_date'] = reporting_date
            if fund_info:
                for key, value in fund_info.items():
                    df[key] = value
                for col in ['series_name', 'class_name']:
                    if col not in df.columns and col in fund_info:
                        df[col] = fund_info[col]
            id_cols = ['filing_date', 'reporting_date', 'ticker', 'fund_name', 'series_id',
                       'series_name', 'class_id', 'class_name', 'cik']
            existing_id_cols = [col for col in id_cols if col in df.columns]
            other_cols = [col for col in df.columns if col not in existing_id_cols]
            return df[existing_id_cols + other_cols]
    except Exception as e:
        logging.error(f"Error extracting composition from {filing_url}: {e}")
        return None
    

def get_nport_filings_from_sec_api(ticker: str) -> pd.DataFrame:
    """
    Fallback: Get NPORT-P filings for a ticker from the SEC's search API.

    Args:
        ticker: Fund ticker symbol.

    Returns:
        DataFrame with columns: 'filing_url', 'Filing Date', 'Reporting for', 'fund_name', 'ticker'
    """
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyFundApp/1.0; email@example.com)'}
    api_url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "entityName": ticker.lower(),
        "category": "custom",
        "forms": "NPORT-P",
        "from": 0,
        "size": 100
    }
    try:
        resp = requests.get(api_url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        filings = []
        for hit in data.get('hits', {}).get('hits', []):
            src = hit.get('_source', {})
            adsh = src.get('adsh')
            cik = src.get('ciks', [None])[0]
            file_date = src.get('file_date')
            period_ending = src.get('period_ending')
            display_names = src.get('display_names', [])
            fund_name = display_names[0] if display_names else None
            # Construct the XML URL
            if adsh and cik:
                cik_nolead = str(int(cik))  # remove leading zeros
                adsh_nodash = adsh.replace("-", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_nolead}/{adsh_nodash}/primary_doc.xml"
                )
            else:
                filing_url = None
            filings.append({
                'fund_name': fund_name,
                'ticker': ticker.upper(),
                'cik': cik,
                'filing_url': filing_url,
                'Filing Date': file_date,
                'Reporting for': period_ending
            })
        df = pd.DataFrame(filings)
        print(df)
        # Remove entries without a filing_url
        df = df[df['filing_url'].notnull()]
        if df.empty:
            logging.warning(f"No NPORT-P filings found via SEC API for {ticker}")
            return pd.DataFrame()
        return df
    except Exception as e:
        logging.error(f"Error scraping SEC API for {ticker}: {e}")
        return pd.DataFrame()
    

def aggregate_fund_positions(
        df: pd.DataFrame,
        group_cols=None,
        name_col='name',
        weight_col='pctVal'
) -> pd.DataFrame:
    """
    Aggregate multiple positions in the same fund for the same target exposure.

    - Combines all rows for a fund (grouped by ticker, fund_name, reporting_date, etc.)
    - For each group, concatenates the position names and their weights, sorted by size.
    - Adds a total percentage weight column, rounded to 2 decimals.

    Args:
        df: DataFrame with columns including 'ticker', 'fund_name', 'reporting_date', name_col, weight_col.
        group_cols: Columns to group by (default: ['ticker', 'fund_name', 'reporting_date']).
        name_col: Name of the column with the security/position name.
        weight_col: Name of the column with the percentage weight.

    Returns:
        Aggregated and sorted DataFrame.
    """
    if group_cols is None:
        group_cols = ['ticker', 'fund_name', 'reporting_date']

    def combine_names_weights(subdf):
        # Sort by weight descending
        subdf = subdf.sort_values(weight_col, ascending=False)
        return "\n".join(
            f"{row[name_col]} ({row[weight_col]:.2f}%)"
            for _, row in subdf.iterrows()
        )

    agg_df = (
        df.groupby(group_cols)
        .apply(lambda subdf: pd.Series({
            'Positions': combine_names_weights(subdf),
            'Total Weight (%)': round(subdf[weight_col].sum(), 2)
        }))
        .reset_index()
    )

    # Sort descending by total weight
    agg_df = agg_df.sort_values('Total Weight (%)', ascending=False).reset_index(drop=True)
    return agg_df