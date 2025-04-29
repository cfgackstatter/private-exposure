import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Output, Input, State, dash_table
import pandas as pd
from fundtools import update_fund_compositions, search_compositions, aggregate_fund_positions

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Private Company Exposure Optimizer"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Private Company Exposure Optimizer"),
            html.P("Find and optimize your exposure to private companies (e.g. Space Exploration) via public funds."),
            dbc.Card([
                dbc.CardHeader("Step 1: Enter Target Private Company"),
                dbc.CardBody([
                    dcc.Input(
                        id='target-input', type='text',
                        placeholder='e.g. Space Exploration',
                        style={'marginRight': '10px'}
                    ),
                    html.Button('Search Funds', id='target-search-btn', className='btn btn-primary')
                ]),
            ], className="mb-3"),
            dbc.Card([
                dbc.CardHeader("Step 2: Update Fund Data (optional)"),
                dbc.CardBody([
                    dcc.Input(
                        id='ticker-input', type='text',
                        placeholder='Enter fund ticker (e.g. BFGFX)',
                        style={'marginRight': '10px'}
                    ),
                    html.Button('Update', id='update-btn', className='btn btn-secondary'),
                    html.Div(id='update-msg', style={'marginTop': '10px', 'color': 'green'})
                ])
            ])
        ], width=3, style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '8px'}),
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Exposure Funds", tab_id="tab-funds"),
                dbc.Tab(label="Optimal Portfolio (coming soon)", tab_id="tab-opt"),
            ], id="tabs", active_tab="tab-funds"),
            html.Div(id='tab-content', style={'marginTop': '20px'})
        ], width=9)
    ])
], fluid=True)

# --- Callbacks ---

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
    Output('tab-content', 'children'),
    Input('target-search-btn', 'n_clicks'),
    State('target-input', 'value'),
    State('tabs', 'active_tab'),
    prevent_initial_call=True
)
def show_exposure_funds(n_clicks, target, active_tab):
    """Show funds with exposure to the target private company."""
    if not target:
        return html.Div("Please enter a private company name.", style={"color": "red"})
    df = search_compositions(target.strip())
    if df.empty:
        return html.Div("No funds found with exposure to this company.", style={"color": "orange"})
    # Merge fund_name and series_name for display
    df = df.copy()
    df['Fund Name'] = df['series_name'].combine_first(df['fund_name'])
    # Show only most relevant columns
    agg_df = aggregate_fund_positions(df)
    show_cols = ['ticker', 'Fund Name', 'reporting_date', 'Positions', 'Total Weight (%)']

    table = dash_table.DataTable(
        data=agg_df[show_cols].to_dict('records'),
        columns=[
            {'id': 'ticker', 'name': 'Fund Ticker'},
            {'id': 'Fund Name', 'name': 'Fund Name'},
            {'id': 'reporting_date', 'name': 'Reporting Date'},
            {'id': 'Positions', 'name': 'Position(s) & Weights'},
            {'id': 'Total Weight (%)', 'name': 'Total Target Weight (%)'},
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
    if active_tab == 'tab-funds':
        return html.Div([
            html.H4(f"Funds with '{target}' Exposure"),
            table
        ])
    elif active_tab == 'tab-opt':
        # Placeholder for future optimal portfolio UI
        return html.Div([
            html.H4("Optimal Long/Short Portfolio (Coming Soon)"),
            html.P("This tab will show the optimal portfolio to maximize pure exposure to your target company, net of costs and hedges.")
        ])

if __name__ == "__main__":
    app.run(debug=True)