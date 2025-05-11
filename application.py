import logging
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input, State, dash_table
from fundtools import update_fund_compositions, search_compositions, aggregate_fund_positions
from optimizer import optimize_target_exposure
from cache import load_all_stock_cache, load_all_fund_cache
from flask_caching import Cache
import time
from threading import Thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


app = dash.Dash(__name__,
                title="Private Company Exposure Optimizer",
                assets_folder='static',
                external_stylesheets=[dbc.themes.BOOTSTRAP]
)
application = app.server

# Configure cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache',
    'CACHE_DEFAULT_TIMEOUT': 3600  # 1 hour timeout
})

# Preload caches at app startup
def load_caches():
    from cache import load_all_stock_cache, load_all_fund_cache
    logging.info("Preloading caches...")
    stock_cache = load_all_stock_cache()
    fund_cache = load_all_fund_cache()
    logging.info(f"Caches loaded: {len(stock_cache)} stocks, {len(fund_cache)} funds")

# Background thread to periodically refresh the cache
def cache_refresh_worker():
    while True:
        try:
            load_caches()
            logging.info("Cache refreshed")
        except Exception as e:
            logging.error(f"Error refreshing cache: {e}")
        time.sleep(3600)  # Refresh every hour

# Start the cache refresh thread
cache_thread = Thread(target=cache_refresh_worker, daemon=True)
cache_thread.start()

# Initial cache load
load_caches()

app.layout = dbc.Container([
    html.Div([
        html.Img(
            src="static/logo.svg",  # Place your SVG in the assets folder
            style={"height": "60px", "margin-right": "15px"}
        ),
        html.H2("Private Company Exposure Optimizer", className="mb-0")
    ], className="d-flex align-items-center mt-4 mb-3"),
    html.P("Find and optimize your exposure to private companies via public funds."),
    
    # Card 1: Update Fund Data
    dbc.Card([
        dbc.CardHeader("Update Fund Data", className="fw-bold"),
        dbc.CardBody([
            html.P("Add or update a mutual fund's composition data:"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Fund Ticker:", html_for="ticker-input"),
                    dbc.InputGroup([
                        dbc.Input(id="ticker-input", placeholder="e.g., BFGFX", type="text"),
                        dbc.Button("Update", id="update-btn", color="primary", className="ms-2")
                    ], className="mb-2"),
                    html.Div(id="update-msg", style={"color": "green"})
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Card 2: Find Exposure
    dbc.Card([
        dbc.CardHeader("Find Private Company Exposure", className="fw-bold"),
        dbc.CardBody([
            html.P("Search for funds with exposure to a specific private company:"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Company Name:", html_for="target-input"),
                    dbc.InputGroup([
                        dbc.Input(id="target-input", placeholder="e.g., Space Expl", type="text"),
                        dbc.Button("Search", id="target-search-btn", color="primary", className="ms-2")
                    ], className="mb-3")
                ])
            ]),
            html.Div(id="search-results")
        ])
    ], className="mb-4"),
    
    # Card 3: Optimize Portfolio
    dbc.Card([
        dbc.CardHeader("Optimize Portfolio", className="fw-bold"),
        dbc.CardBody([
            html.P("Create an optimal long/short portfolio to maximize exposure to a private company:"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Target Company:", html_for="optimize-target-input"),
                    dbc.Input(id="optimize-target-input", placeholder="e.g., Space Expl", type="text", className="mb-2"),
                    
                    dbc.Label("Investment Amount:", html_for="investment-amount"),
                    dbc.InputGroup([
                        dbc.InputGroupText("$"),
                        dbc.Input(id="investment-amount", placeholder="100", type="number", value=10000),
                    ], className="mb-3"),
                    
                    # Collapsible advanced parameters
                    html.Div([
                        dbc.Button(
                            "Advanced Parameters",
                            id="collapse-button",
                            className="mb-3",
                            color="secondary",
                            outline=True,
                        ),
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Cash Interest Rate (%):", html_for="cash-interest-rate"),
                                        dbc.Input(id="cash-interest-rate", type="number", value=5, min=0, max=20, step=0.1, className="mb-2"),

                                        dbc.Label("Margin Interest Rate (%):", html_for="margin-rate"),
                                        dbc.Input(id="margin-rate", type="number", value=6, min=0, max=20, step=0.1, className="mb-2"),
                                        
                                        dbc.Label("Cost to Borrow (%):", html_for="borrow-cost"),
                                        dbc.Input(id="borrow-cost", type="number", value=2, min=0, max=10, step=0.1, className="mb-2"),
                                        
                                        dbc.Label("Transaction Cost (%):", html_for="transaction-cost"),
                                        dbc.Input(id="transaction-cost", type="number", value=0.1, min=0, max=5, step=0.01, className="mb-2"),
                                    ], md=6),
                                    dbc.Col([
                                        dbc.Label("Fixed Transaction Cost ($):", html_for="fixed-transaction-cost"),
                                        dbc.Input(id="fixed-transaction-cost", type="number", value=10.0, min=0, max=50, step=0.5, className="mb-2"),
                                        
                                        dbc.Label("Maximum Leverage:", html_for="max-leverage"),
                                        dbc.Input(id="max-leverage", type="number", value=5.0, min=1, max=10, step=0.5, className="mb-2"),
                                        
                                        dbc.Label("Holding Period (years):", html_for="holding-period"),
                                        dbc.Input(id="holding-period", type="number", value=1.0, min=0.25, max=10, step=0.25, className="mb-2"),

                                        dbc.Label("Minimum Position Size (%):", html_for="min-position-pct"),
                                        dbc.Input(id="min-position-pct", type="number", value=0, min=0.1, max=10, step=0.1, className="mb-2")
                                    ], md=6)
                                ]),
                            ])),
                            id="collapse",
                            is_open=False,
                        ),
                    ]),
                    
                    dbc.Button("Optimize Portfolio", id="optimize-btn", color="primary", className="mb-3"),
                    html.Div(id="optimize-results")
                ])
            ])
        ])
    ])
], fluid=True, className="mb-5")

# --- Callbacks ---

@app.callback(
    Output('update-btn', 'n_clicks'),
    Input('ticker-input', 'n_submit'),
    State('update-btn', 'n_clicks'),
    prevent_initial_call=True
)
def update_on_enter(n_submit, n_clicks):
    """Trigger update button when Enter is pressed in ticker input."""
    return (n_clicks or 0) + 1

@app.callback(
    Output('target-search-btn', 'n_clicks'),
    Input('target-input', 'n_submit'),
    State('target-search-btn', 'n_clicks'),
    prevent_initial_call=True
)
def search_on_enter(n_submit, n_clicks):
    """Trigger search button when Enter is pressed in target input."""
    return (n_clicks or 0) + 1

@app.callback(
    Output('optimize-btn', 'n_clicks'),
    Input('optimize-target-input', 'n_submit'),
    State('optimize-btn', 'n_clicks'),
    prevent_initial_call=True
)
def optimize_on_enter(n_submit, n_clicks):
    """Trigger optimize button when Enter is pressed in optimize input."""
    return (n_clicks or 0) + 1

@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('update-msg', 'children'),
    Input('update-btn', 'n_clicks'),
    State('ticker-input', 'value'),
    prevent_initial_call=True
)
def update_fund(n_clicks, ticker):
    """Callback to update fund compositions for a given ticker."""
    if not ticker:
        return "Please enter a ticker."
    msg = update_fund_compositions(ticker.strip())
    return msg

@app.callback(
    Output('search-results', 'children'),
    Input('target-search-btn', 'n_clicks'),
    State('target-input', 'value'),
    prevent_initial_call=True
)
def show_exposure_funds(n_clicks, target):
    """Show funds with exposure to the target private company."""
    if not target:
        return html.Div("Please enter a private company name.", style={"color": "red"})
    
    df = search_compositions(target.strip())
    if df.empty:
        return html.Div("No funds found with exposure to this company.", style={"color": "orange"})
    
    # Show only most relevant columns
    agg_df = aggregate_fund_positions(df)
    show_cols = ['ticker', 'fund_name', 'reporting_date', 'Positions', 'Total Weight (%)']
    
    table = dash_table.DataTable(
        data=agg_df[show_cols].to_dict('records'),
        columns=[
            {'id': 'ticker', 'name': 'Fund Ticker'},
            {'id': 'fund_name', 'name': 'Fund Name'},
            {'id': 'reporting_date', 'name': 'Reporting Date'},
            {'id': 'Positions', 'name': 'Position(s) & Weights'},
            {'id': 'Total Weight (%)', 'name': 'Total Exposure Weight (%)'},
        ],
        page_size=20,
        style_table={
            'height': 'auto',
            'width': '100%',
            'minWidth': '100%',
            'border': '1px solid #dee2e6',
            'backgroundColor': '#fff',
            'borderRadius': '6px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.05)'
        },
        style_cell={
            'whiteSpace': 'normal',
            'height': 'auto',
            'textAlign': 'left',
            'fontFamily': 'Segoe UI, Arial, sans-serif',
            'fontSize': '15px',
            'maxWidth': '350px',
            'minWidth': '80px',
            'padding': '8px'
        },
        style_cell_conditional=[
            {
                'if': {'column_id': 'Positions'},
                'maxWidth': '500px',
                'minWidth': '500px',
                'width': '35%',
                'whiteSpace': 'pre-line',
            },
        ],
        style_header={
            'backgroundColor': '#f1f3f5',
            'fontWeight': 'bold',
            'fontFamily': 'Segoe UI, Arial, sans-serif',
            'fontSize': '15px'
        },
    )
    
    return html.Div([
        html.H4(f"Funds with '{target}' exposure"),
        table
    ])

# Callback for the optimize button
@app.callback(
    Output("optimize-results", "children"),
    Input("optimize-btn", "n_clicks"),
    [State("optimize-target-input", "value"),
     State("investment-amount", "value"),
     State("cash-interest-rate", "value"),
     State("margin-rate", "value"),
     State("borrow-cost", "value"),
     State("transaction-cost", "value"),
     State("fixed-transaction-cost", "value"),
     State("max-leverage", "value"),
     State("holding-period", "value"),
     State("min-position-pct", "value")],
    prevent_initial_call=True
)
def run_optimization(n_clicks, target, investment, cash_interest_rate, margin_rate, borrow_cost,
                    transaction_cost, fixed_transaction_cost, max_leverage,
                    holding_period, min_position_pct):
    if not target:
        return html.Div("Please enter a target company name.", style={"color": "red"})
    
    if not investment or investment <= 0:
        return html.Div("Please enter a valid investment amount.", style={"color": "red"})
    
    # Convert percentage inputs to decimals
    cash_interest_rate = cash_interest_rate / 100 if cash_interest_rate else 0.05
    margin_rate = margin_rate / 100 if margin_rate else 0.05
    borrow_cost = borrow_cost / 100 if borrow_cost else 0.02
    transaction_cost = transaction_cost / 100 if transaction_cost else 0.001

    # Set default values for new parameters if they're None
    fixed_transaction_cost = fixed_transaction_cost if fixed_transaction_cost is not None else 1.0
    max_leverage = max_leverage if max_leverage is not None else 4.0
    holding_period = holding_period if holding_period is not None else 1.0

    # Convert min_position_pct from percentage to decimal
    min_position_pct = min_position_pct / 100 if min_position_pct is not None else 0.01
    
    # Run the optimization
    result = optimize_target_exposure(
        target, 
        investment,
        cash_interest_rate=cash_interest_rate,
        margin_rate=margin_rate,
        borrow_cost=borrow_cost,
        transaction_cost=transaction_cost,
        fixed_transaction_cost=fixed_transaction_cost,
        max_leverage=max_leverage,
        holding_period_years=holding_period,
        min_position_pct=min_position_pct
    )
    
    if "error" in result:
        return html.Div(result["error"], style={"color": "red"})
    
    # Format the results
    fund_rows = []
    for fund, amount in result["fund_allocations"].items():
        shares = result["fund_shares"].get(fund, 0)
        fund_rows.append(
            html.Tr([
                html.Td(fund),
                html.Td(f"${amount:.2f}"),
                html.Td(f"{shares:.2f}")
            ])
        )
    
    short_rows = []
    sorted_shorts = sorted(result["short_allocations"].items(), key=lambda x: abs(x[1]), reverse=True)
    for symbol, amount in sorted_shorts:
        shares = result["short_shares"].get(symbol, None)
        name = result.get("short_names", {}).get(symbol, "Unknown")
        display_name = f"{name} ({symbol})"
        short_rows.append(
            html.Tr([
                html.Td(display_name),
                html.Td(f"${amount:.2f}"),
                html.Td(f"{shares:.2f}")
            ])
        )
    
    return html.Div([
        html.H4(f"Optimal Portfolio for {target}"),
        html.P(f"Investment amount: ${investment}"),
        
        html.Div([
            html.H5("Portfolio Metrics:", className="mt-3"),
            html.Table([
                html.Tr([html.Td("Target Exposure:"), html.Td(f"${result['target_exposure']:.2f} ({result['target_exposure']/investment*100:.2f}%)")]),
                html.Tr([html.Td("Absolute Net Non-Target Exposure:"), html.Td(f"${result['abs_net_exposure']:.2f} ({result['metrics']['abs_net_exposure_pct']:.2f}%)")]),
                html.Tr([html.Td("Cash Position:"), html.Td(f"${result['cash']:.2f}")]),
            ], className="table table-sm")
        ]),
        
        html.Div([
            html.H5("Recommended Long Positions:", className="mt-3"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Fund", style={"width": '50%'}), 
                    html.Th("Amount", style={"width": '25%'}), 
                    html.Th("Shares", style={"width": '25%'})
                ])),
                html.Tbody(fund_rows)
            ], className="table table-striped") if fund_rows else html.P("No long positions recommended.")
        ]),
        
        html.Div([
            html.H5("Recommended Short Positions:", className="mt-3"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Symbol", style={"width": '50%'}), 
                    html.Th("Amount", style={"width": '25%'}), 
                    html.Th("Shares", style={"width": '25%'})
                ])),
                html.Tbody(short_rows)
            ], className="table table-striped") if short_rows else html.P("No short positions recommended.")
        ]),
        
        html.Div([
            dbc.Alert(
                [
                    html.H5("Important Note:", className="alert-heading"),
                    html.P("This is a theoretical portfolio optimization that assumes you can short the exact stocks in the proportions needed. In practice, implementation may be limited by broker availability, margin requirements, and borrowing constraints.")
                ],
                color="info",
                className="mt-3"
            )
        ])
    ])

if __name__ == "__main__":
    app.run(debug=True)