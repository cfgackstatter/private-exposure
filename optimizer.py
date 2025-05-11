import logging
import re
import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cp
from cache import get_cached_stock_info_fast, get_fund_expense_ratio_fast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def optimize_target_exposure(target_company, investment_amount, cash_interest_rate=0.05,
                            margin_rate=0.05, borrow_cost=0.02,
                            transaction_cost=0.001, fixed_transaction_cost=1.0,
                            max_leverage=5.0, holding_period_years=1.0,
                            update_cache=True, min_position_pct=0.01):
    """
    Optimize a portfolio to maximize exposure to a target private company
    while minimizing all other exposures through hedging when possible.
    
    Parameters:
    -----------
    target_company : str
        Name of the target private company (e.g., "space expl")
    investment_amount : float
        Total investment amount in dollars
    cash_interest_rate : float
        Annual interest rate for cash (default: 5%)
    margin_rate : float
        Annual interest rate for margin (default: 5%)
    borrow_cost : float
        Annual cost to borrow securities for shorting (default: 2%)
    transaction_cost : float
        Transaction cost as a fraction of trade value (default: 0.1%)
    fixed_transaction_cost : float
        Fixed cost per transaction in dollars (default: $1.00)
    update_cache : bool
        Whether to update cache with new data (default: True)
        
    Returns:
    --------
    dict
        Portfolio allocation with fund positions, shorts, and metrics
    """
    # 1. Find funds with target exposure
    from fundtools import search_compositions, aggregate_fund_positions
    from cache import get_cached_stock_info, update_stock_info_cache, get_fund_expense_ratio
    
    df = search_compositions(target_company)
    if df.empty:
        return {"error": f"No funds found with exposure to '{target_company}'"}
    
    # Aggregate positions by fund
    agg_df = aggregate_fund_positions(df)
    
    # 2. Extract all holdings from these funds
    target_weights = {}  # Fund ticker -> target weight
    other_holdings = {}  # Symbol -> {fund ticker -> weight}
    non_hedgeable = {}  # Fund ticker -> {holding name -> weight}

    for _, row in agg_df.iterrows():
        fund_ticker = row['ticker']
        target_weights[fund_ticker] = row['Total Weight (%)'] / 100
        non_hedgeable[fund_ticker] = {}

        # Get all holdings for this fund
        try:
            fund_df = pd.read_parquet(f"data/{fund_ticker}.parquet")
            latest_date = fund_df['reporting_date'].max()
            latest_holdings = fund_df[fund_df['reporting_date'] == latest_date]
            
            # Identify target vs non-target holdings
            target_mask = latest_holdings['name'].str.contains(target_company, case=False, na=False)
            
            # Process all non-target holdings
            non_target = latest_holdings[~target_mask]
            for _, holding in non_target.iterrows():
                weight = holding['pctVal'] / 100
                holding_name = holding['name'] if pd.notna(holding['name']) else "Unknown"
                
                # Check if this holding has a valid identifier (ISIN or CUSIP) we can use for hedging
                identifier = None

                # First check for ISIN (preferred)
                if 'isin' in holding and pd.notna(holding['isin']):
                    potential_isin = str(holding['isin']).strip().upper()
                    # Validate it's not a placeholder like "N/A" or all zeros
                    if (potential_isin and 
                        potential_isin not in ["N/A", "NA", "NONE", "NULL", ""] and
                        not all(c == '0' for c in potential_isin)):
                        # ISIN validation: 12 characters, alphanumeric
                        if len(potential_isin) == 12 and potential_isin.isalnum():
                            identifier = potential_isin
                
                # Then check for CUSIP if no valid ISIN
                if not identifier and 'cusip' in holding and pd.notna(holding['cusip']):
                    potential_cusip = str(holding['cusip']).strip().upper()
                    # Validate it's not a placeholder or all zeros
                    if (potential_cusip and 
                        potential_cusip not in ["N/A", "NA", "NONE", "NULL", ""] and
                        not all(c == '0' for c in potential_cusip) and
                        potential_cusip != "000000000"):
                        # CUSIP validation: 9 characters, alphanumeric
                        if len(potential_cusip) == 9 and potential_cusip.isalnum():
                            identifier = potential_cusip
                
                # If no valid ISIN/CUSIP, try to extract ticker from name
                if not identifier and pd.notna(holding['name']):
                    name = str(holding['name']).upper()
                    if any(pattern in name for pattern in [':', ' - ', '(', '/']):
                        # Try to extract ticker from name if it follows common patterns
                        match = re.search(r'[:(]\s*([A-Z]{1,5})[),]', name)
                        if match:
                            potential_ticker = match.group(1).strip()
                            if (potential_ticker and 
                                potential_ticker not in ["N/A", "NA", "NONE", "NULL", ""] and
                                not all(c == '0' for c in potential_ticker)):
                                identifier = potential_ticker
                
                if identifier:
                    # This holding has a valid identifier and can potentially be hedged
                    if identifier not in other_holdings:
                        other_holdings[identifier] = {}
                    other_holdings[identifier][fund_ticker] = weight
                else:
                    # This holding can't be hedged - store by name
                    non_hedgeable[fund_ticker][holding_name] = weight
        except (FileNotFoundError, KeyError) as e:
            logging.error(f"Error processing fund {fund_ticker}: {e}")
            return {"error": f"Error processing fund {fund_ticker}: {e}"}
    
    # 3. Get current prices for holdings that can be hedged
    prices = {}
    hedgeable_symbols = []
    
    for symbol in other_holdings:
        try:
            # First check if we have fresh cached info
            cached_info = get_cached_stock_info_fast(symbol)
            
            if cached_info:
                # Use cached price information
                price = None
                for attr in ['regularMarketPrice', 'currentPrice', 'previousClose', 'open', 'ask']:
                    if attr in cached_info and cached_info[attr] is not None:
                        price = cached_info[attr]
                        break
                        
                if price and price > 0:
                    prices[symbol] = price
                    hedgeable_symbols.append(symbol)
                    logging.info(f"Using cached price for {symbol}: ${price}")
                else:
                    # If no valid price found, add to non-hedgeable
                    for fund, weight in other_holdings[symbol].items():
                        non_hedgeable[fund][f"No price: {symbol}"] = weight
                    logging.warning(f"No valid price in cached data for {symbol}, treating as non-hedgeable")
            elif update_cache:
                # No cache or stale cache, fetch from yfinance
                import yfinance as yf
                logging.info(f"Fetching fresh price data for {symbol}")
                stock = yf.Ticker(symbol)
                stock_info = stock.info
                
                # Store in cache for future use
                update_stock_info_cache(symbol, stock_info)
                
                # Try different price attributes as they may vary
                price = None
                for attr in ['regularMarketPrice', 'currentPrice', 'previousClose', 'open', 'ask']:
                    if attr in stock_info and stock_info[attr] is not None:
                        price = stock_info[attr]
                        break
                
                if price and price > 0:
                    prices[symbol] = price
                    hedgeable_symbols.append(symbol)
                    logging.info(f"Fetched price for {symbol}: ${price}")
                else:
                    # If no valid price found, add to non-hedgeable
                    for fund, weight in other_holdings[symbol].items():
                        non_hedgeable[fund][f"No price: {symbol}"] = weight
                    logging.warning(f"Could not get price for {symbol}, treating as non-hedgeable")
            else:
                # No cache update requested, treat as non-hedgeable
                for fund, weight in other_holdings[symbol].items():
                    non_hedgeable[fund][f"No price lookup: {symbol}"] = weight
                logging.info(f"Skipping price lookup for {symbol} as update_cache=False")
        except Exception as e:
            # If error fetching price, add to non-hedgeable
            for fund, weight in other_holdings[symbol].items():
                non_hedgeable[fund][f"Error: {symbol}"] = weight
            logging.warning(f"Error fetching price for {symbol}, treating as non-hedgeable: {e}")
    
    # Remove symbols that couldn't be hedged from other_holdings
    other_holdings = {k: v for k, v in other_holdings.items() if k in hedgeable_symbols}
    
    # 4. Set up optimization problem using CVXPY with SCIP
    funds = list(target_weights.keys())
    symbols = hedgeable_symbols
    
    n_funds = len(funds)
    n_symbols = len(symbols)
    
    # Create exposure matrix: how much of each hedgeable holding is in each fund
    exposure_matrix = np.zeros((n_funds, n_symbols))
    for j, symbol in enumerate(symbols):
        for i, fund in enumerate(funds):
            exposure_matrix[i, j] = other_holdings[symbol].get(fund, 0)
    
    # Get expense ratios for each fund
    expense_ratios = {}
    for fund in funds:
        expense_ratios[fund] = get_fund_expense_ratio_fast(fund)
        logging.info(f"Using expense ratio of {expense_ratios[fund]*100:.2f}% for {fund}")
    
    # Create variables
    import cvxpy as cp
    fund_vars = cp.Variable(n_funds, nonneg=True)
    short_vars = cp.Variable(n_symbols, nonpos=True)
    cash_var = cp.Variable()
    
    # Binary variables for fixed transaction costs
    fund_binary = cp.Variable(n_funds, boolean=True)
    short_binary = cp.Variable(n_symbols, boolean=True)
    
    # Constraints
    constraints = []
    
    # Budget constraint
    constraints.append(cp.sum(fund_vars) + cp.sum(short_vars) + cash_var == investment_amount)
    
    # Add a leverage limit constraint
    constraints.append(cp.norm(fund_vars, 1) + cp.norm(short_vars, 1) <= max_leverage * investment_amount)
    
    # Binary variable constraints for fixed transaction costs
    min_position = min_position_pct * investment_amount  # Minimum position size
    for i in range(n_funds):
        # If fund_vars[i] > min_position, then fund_binary[i] = 1
        constraints.append(fund_vars[i] <= max_leverage * investment_amount * fund_binary[i])
        constraints.append(min_position * fund_binary[i] <= fund_vars[i])
    
    for j in range(n_symbols):
        # If short_vars[j] < -min_position, then short_binary[j] = 1
        constraints.append(short_vars[j] >= - max_leverage * investment_amount * short_binary[j])  # Upper bound when binary=1
        constraints.append(short_vars[j] <= -min_position * short_binary[j])  # Lower bound when binary=1
    
    # Objective components
    # Target exposure (maximize as percentage)
    target_exp_pct = sum(fund_vars[i] * target_weights[funds[i]] for i in range(n_funds)) / investment_amount

    # Calculate net exposure for each hedgeable stock
    stock_net_exposures = []
    for j in range(n_symbols):
        # Long exposure from all funds for this stock
        long_exposure = sum(fund_vars[i] * exposure_matrix[i, j] for i in range(n_funds))
        # Short position for this stock
        short_exposure = short_vars[j]
        # Net exposure for this stock
        net_exposure = long_exposure + short_exposure  # short_exposure is already negative
        # Absolute value of net exposure
        abs_net_exposure = cp.abs(net_exposure)
        stock_net_exposures.append(abs_net_exposure)
    
    # Sum the absolute net exposures for all hedgeable stocks
    hedgeable_abs_net_exposure = sum(stock_net_exposures)

    # Define non_hedge_weights based on non_hedgeable dictionary
    non_hedge_weights = [sum(weight for _, weight in non_hedgeable[funds[i]].items()) for i in range(n_funds)]
    
    # Add non-hedgeable exposure (which can't be netted)
    non_hedge_exposure = sum(fund_vars[i] * non_hedge_weights[i] for i in range(n_funds))
    abs_non_hedge_exposure = cp.abs(non_hedge_exposure)

    # Add cash to the non-targeted exposure calculation
    cash_abs_exposure = cp.abs(cash_var)

    # Total absolute net exposure as percentage (including cash)
    total_abs_net_exposure_pct = (hedgeable_abs_net_exposure + abs_non_hedge_exposure + cash_abs_exposure) / investment_amount

    # Costs (all annualized dollar amounts)
    # Expense ratio cost (annual)
    expense_ratio_cost = sum(fund_vars[i] * expense_ratios[funds[i]] for i in range(n_funds))

    # Borrow cost (annual)
    borrow_cost_expr = cp.sum(cp.abs(short_vars)) * borrow_cost

    # Margin cost (annual)
    margin_cost_expr = cp.maximum(0, -cash_var) * margin_rate

    # Transaction costs (one-time, but annualized by dividing by holding period)
    proportional_trans_cost = transaction_cost * (cp.sum(cp.abs(fund_vars)) + cp.sum(cp.abs(short_vars))) / holding_period_years

    # Fixed transaction costs (one-time, annualized)
    fixed_trans_cost = fixed_transaction_cost * (cp.sum(fund_binary) + cp.sum(short_binary)) / holding_period_years

    # Cash interest earned (annual)
    # Binary variable to indicate if cash is positive
    cash_is_positive = cp.Variable(boolean=True)
    positive_cash = cp.Variable(nonneg=True) # New variable for positive cash
    constraints.append(positive_cash <= cash_var) # positive_cash is at most cash_var
    M = investment_amount * max_leverage # Big-M value
    constraints.append(positive_cash <= M * cash_is_positive) # If cash_is_positive=0, positive_cash=0
    constraints.append(cash_var - positive_cash <= M * (1 - cash_is_positive)) # If cash_is_positive=1, positive_cash=cash_var

    # Then use positive_cash for interest calculation
    cash_interest_earned = positive_cash * cash_interest_rate

    # Total costs as percentage of investment
    total_cost_pct = (expense_ratio_cost + borrow_cost_expr + margin_cost_expr + proportional_trans_cost + fixed_trans_cost) / investment_amount
    cash_interest_earned_pct = cash_interest_earned / investment_amount

    # Objective: maximize target exposure percentage, minimize absolute non-target exposure percentage and costs
    objective = cp.Maximize(target_exp_pct - total_abs_net_exposure_pct - total_cost_pct + cash_interest_earned_pct)
    
    # Create and solve problem
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCIP, verbose=True)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return {"error": f"Optimization failed: {prob.status}"}
        
        # Extract solution
        fund_allocations = fund_vars.value
        short_allocations = short_vars.value
        cash_allocation = cash_var.value
        
    except Exception as e:
        return {"error": f"Optimization failed: {str(e)}"}
    
    # Calculate number of shares for each position
    fund_shares = {}
    for i, fund in enumerate(funds):
        if fund_allocations[i] > 0 * investment_amount:  # Minimum threshold
            try:
                import yfinance as yf
                fund_ticker_obj = yf.Ticker(fund)
                fund_price = fund_ticker_obj.info.get('navPrice',
                                                     fund_ticker_obj.info.get('regularMarketPrice',
                                                                            fund_ticker_obj.info.get('previousClose', None)))
                if not fund_price or fund_price <= 0:
                    fund_price = None  # Default if no valid price
            except Exception as e:
                logging.warning(f"Error getting price for fund {fund}: {e}")
                fund_price = None  # Default if error
                
            fund_shares[fund] = fund_allocations[i] / fund_price
    
    # Short shares calculation
    short_shares = {}
    for j, symbol in enumerate(symbols):
        if abs(short_allocations[j]) > 0 * investment_amount:  # Minimum threshold
            price = prices.get(symbol, None)
            shares = short_allocations[j] / price
            short_shares[symbol] = shares
    
    # Calculate metrics
    target_exposure = sum(fund_allocations[i] * target_weights[funds[i]] for i in range(n_funds))
    target_exposure_pct = target_exposure / investment_amount * 100
    
    # Calculate non-target exposure in two steps
    # Step 1: Calculate gross non-target exposure from funds (both hedgeable and non-hedgeable)
    gross_non_target_exposure = sum(
        fund_allocations[i] * (1 - target_weights[funds[i]])
        for i in range(n_funds)
    )

    # Step 2: Subtract the short positions (which hedge part of the non-target exposure)
    hedged_exposure = sum(abs(short_allocations[j]) for j in range(n_symbols))
    hedged_exposure_pct = hedged_exposure / investment_amount * 100
    unhedged_exposure = gross_non_target_exposure - hedged_exposure
    unhedged_exposure_pct = unhedged_exposure / investment_amount * 100
    
    # Calculate costs
    expense_ratio_val = sum(fund_allocations[i] * expense_ratios[funds[i]] for i in range(n_funds))
    borrow_cost_val = sum(abs(short_allocations[j]) * borrow_cost for j in range(n_symbols))
    margin_cost_val = max(0, -cash_allocation) * margin_rate
    
    # Count number of transactions
    num_fund_transactions = sum(1 for i in range(n_funds) if fund_allocations[i] > 0.0 * investment_amount)
    num_short_transactions = sum(1 for j in range(n_symbols) if abs(short_allocations[j]) > 0.0 * investment_amount)
    
    # Calculate transaction costs
    proportional_trans_cost_val = transaction_cost * (sum(abs(fund_allocations)) + sum(abs(short_allocations)))
    fixed_trans_cost_val = fixed_transaction_cost * (num_fund_transactions + num_short_transactions)
    total_trans_cost_val = proportional_trans_cost_val + fixed_trans_cost_val
    
    # Annualized costs
    total_annual_cost_val = expense_ratio_val + borrow_cost_val + margin_cost_val + total_trans_cost_val / holding_period_years

    # Calculate the absolute net exposure for metrics
    stock_net_exposures_val = []
    for j in range(n_symbols):
        long_exposure_val = sum(fund_allocations[i] * exposure_matrix[i, j] for i in range(n_funds))
        short_exposure_val = short_allocations[j]
        net_exposure_val = long_exposure_val + short_exposure_val
        abs_net_exposure_val = abs(net_exposure_val)
        stock_net_exposures_val.append(abs_net_exposure_val)

    hedgeable_abs_net_exposure_val = sum(stock_net_exposures_val)
    non_hedge_exposure_val = sum(fund_allocations[i] * non_hedge_weights[i] for i in range(n_funds))
    abs_non_hedge_exposure_val = abs(non_hedge_exposure_val)
    cash_abs_exposure_val = abs(cash_allocation)
    total_abs_net_exposure_val = hedgeable_abs_net_exposure_val + abs_non_hedge_exposure_val + cash_abs_exposure_val
    total_abs_net_exposure_pct_val = total_abs_net_exposure_val / investment_amount * 100
    
    return {
        "fund_allocations": {funds[i]: fund_allocations[i] for i in range(n_funds) if fund_allocations[i] > 0.0 * investment_amount},
        "fund_shares": fund_shares,
        "short_allocations": {symbols[j]: short_allocations[j] for j in range(n_symbols) if abs(short_allocations[j]) > 0.0 * investment_amount},
        "short_shares": short_shares,
        "short_names": {
            symbols[j]: get_cached_stock_info_fast(symbols[j]).get('shortName', 
                get_cached_stock_info_fast(symbols[j]).get('longName', symbols[j])) 
            if get_cached_stock_info_fast(symbols[j]) else symbols[j]
            for j in range(n_symbols)
        },
        "cash": cash_allocation,
        "target_exposure": target_exposure,
        "unhedged_exposure": unhedged_exposure,
        "hedged_exposure": hedged_exposure,
        "abs_net_exposure": total_abs_net_exposure_val,
        "costs": {
            "expense_ratio": expense_ratio_val,
            "borrow_cost": borrow_cost_val,
            "margin_cost": margin_cost_val,
            "transaction_cost": {
                "proportional": proportional_trans_cost_val,
                "fixed": fixed_trans_cost_val,
                "total": total_trans_cost_val
            },
            "total_annual_cost": total_annual_cost_val
        },
        "metrics": {
            "target_pct": target_exposure_pct,
            "unhedged_pct": unhedged_exposure_pct,  # Renamed from non_hedgeable_pct
            "hedged_pct": hedged_exposure_pct,
            "abs_net_exposure_pct": total_abs_net_exposure_pct_val,
            "cost_pct": total_annual_cost_val / investment_amount * 100,
            "num_transactions": num_fund_transactions + num_short_transactions
        }
    }


def main():
    """
    Test case for the optimizer.
    """
    logging.info("Running test case for portfolio optimizer")
    
    # Test parameters
    target_company = "space expl"
    investment_amount = 10000
    
    # Run optimization
    result = optimize_target_exposure(
        target_company=target_company,
        investment_amount=investment_amount,
        cash_interest_rate=0.05,
        margin_rate=0.06,
        borrow_cost=0.02,
        transaction_cost=0.001,
        fixed_transaction_cost=10.0,
        max_leverage=5.0,
        holding_period_years=1.0,
        min_position_pct=0.0
    )
    
    # Print results
    if "error" in result:
        logging.error(f"Optimization failed: {result['error']}")
    else:
        logging.info(f"Optimization successful!")
        logging.info(f"Target exposure: ${result['target_exposure']:.2f} ({result['metrics']['target_pct']:.2f}%)")
        logging.info(f"Non-hedgeable exposure: ${result['unhedged_exposure']:.2f} ({result['metrics']['unhedged_pct']:.2f}%)")
        logging.info(f"Hedged exposure: ${result['hedged_exposure']:.2f} ({result['metrics']['hedged_pct']:.2f}%)")
        logging.info(f"Cash position: ${result['cash']:.2f}")
        logging.info(f"Total annual cost: ${result['costs']['total_annual_cost']:.2f} ({result['metrics']['cost_pct']:.2f}%)")
        logging.info(f"Number of transactions: {result['metrics']['num_transactions']}")
        
        logging.info("\nFund allocations:")
        for fund, amount in result['fund_allocations'].items():
            logging.info(f"  {fund}: ${amount:.2f} ({amount/investment_amount*100:.2f}%)")
        
        logging.info("\nShort allocations:")
        for symbol, amount in result['short_allocations'].items():
            logging.info(f"  {symbol}: ${amount:.2f} ({amount/investment_amount*100:.2f}%)")

if __name__ == "__main__":
    main()