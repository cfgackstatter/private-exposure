from fundtools import (
    get_fund_identifiers,
    get_nport_filings,
    extract_composition_from_filing,
    get_nport_filings_from_sec_api
)

if __name__ == "__main__":
    # Example: Download and save the latest composition for a given ticker
    ticker = "DXYZ"
    print(f"Getting fund identifiers for {ticker}...")
    fund_info = get_fund_identifiers(ticker)

    if fund_info:
        print(f"Fund: {fund_info['fund_name']} ({fund_info['ticker']})")
        print(f"CIK: {fund_info['cik']}")
        print(f"Series ID: {fund_info['series_id']}")
        print(f"Class ID: {fund_info['class_id']}")
        class_cik = fund_info['class_id']
        print(f"\nGetting NPORT-P filings for {class_cik}...")
        filings_df = get_nport_filings(class_cik)
        fund_info_for_each = [fund_info] * len(filings_df)  # use same info for all
    else:
        print(f"Could not retrieve fund identifiers for {ticker}, using fallback SEC API search.")
        filings_df = get_nport_filings_from_sec_api(ticker)
        # For fallback, create a fund_info dict for each row from the DataFrame
        fund_info_for_each = []
        for _, row in filings_df.iterrows():
            fund_info_row = {
                'ticker': row.get('ticker'),
                'fund_name': row.get('fund_name'),
                'cik': row.get('cik'),
                'filing_date': row.get('Filing Date'),
                'reporting_date': row.get('Reporting for'),
                'series_name': None,
                'class_name': None,
                'series_id': None,
                'class_id': None
            }
            fund_info_for_each.append(fund_info_row)

    if filings_df is None or filings_df.empty:
        print(f"No NPORT-P filings found for {ticker}")
        exit(1)

    print(f"Found {len(filings_df)} NPORT-P filings")
    print(filings_df[['Filing Date', 'filing_url']].head())

    # Get the most recent filing
    filing_url = filings_df.iloc[0]['filing_url']
    fund_info_to_use = fund_info_for_each[0]
    print(f"\nExtracting composition from {filing_url}...")
    composition_df = extract_composition_from_filing(filing_url, fund_info_to_use)

    if composition_df is None or composition_df.empty:
        print(f"Could not extract composition from {filing_url}")
        exit(1)

    print(f"Successfully extracted {len(composition_df)} holdings")
    print(composition_df.head())

    # Save to CSV
    output_file = f"{ticker}_{composition_df['reporting_date'].iloc[0]}.csv"
    composition_df.to_csv(output_file, index=False)
    print(f"Saved composition to {output_file}")