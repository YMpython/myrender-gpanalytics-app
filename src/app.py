
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_table
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from scipy import stats

# Initialize the Dash app with Bootstrap for better design
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Load and clean your dataset
df = pd.read_csv("/Users/yutaromaeda/MBA関連/Summer project/Analysis/Patient satisfaction survey(2018-2023)/Cleaned_GPPS_2024_Practice_data.csv")

# Ensure no NaN values for calculations
df_nonan = df.dropna(subset=[
    "overallexp.pcteval", "gpcontactoverall.pcteval", 
    "lastgpapptwait.pcteval", 'localgpservicesphone.pcteval', 
    'localgpserviceswebsite.pcteval', 'localgpservicesreception.pcteval', 
    'gpcontactnextstep.pcteval', 'lastgpapptchoice.pcteval',
    'localgpservicesprefhp.pcteval', 'localgpservicesprefhpsee.pcteval'
])

# Create the new variable for Continuity of Care
df_nonan['prefhp_interaction'] = df_nonan['localgpservicesprefhp.pcteval'] * df_nonan['localgpservicesprefhpsee.pcteval']

# Independent variables for regression analysis
X_columns = [
    "lastgpapptwait.pcteval", 'localgpservicesphone.pcteval', 
    'localgpserviceswebsite.pcteval', 'localgpservicesreception.pcteval', 
    'gpcontactnextstep.pcteval', 'lastgpapptchoice.pcteval'
]

# Predictor mapping for dropdown selection and scatter plot
predictor_mapping = {
    "Waiting time": 'gpcontactnextstep.pcteval',
    "Phone": 'localgpservicesphone.pcteval',
    "Online access": 'lastgpapptchoice.pcteval',
    "Reception": 'localgpservicesreception.pcteval',
    "Information quality": 'lastgpapptwait.pcteval',
    "Choice": 'localgpserviceswebsite.pcteval'
}

# Helper functions
def create_card(title, content):
    return dbc.Card(
        dbc.CardBody([
            html.H5(title, className="card-title", style={"color": "#007bff", "fontWeight": "bold"}),
            content
        ]),
        className="mb-4",
        style={"boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)", "borderRadius": "10px"}
    )

def calculate_r_squared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2

def count_practices(data):
    return len(data)

# App layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("GP Practice Patient Satisfaction Dashboard", className="text-center my-4",
                style={"fontWeight": "bold", "fontSize": "2.5rem", "color": "#343A40"}),
        
        dbc.Row([
            dbc.Col(create_card(
                "Select GP Practice", 
                dcc.Dropdown(
                    id='practice-dropdown', 
                    options=[{'label': practice, 'value': practice} for practice in df_nonan['ad_practicename'].unique()],
                    placeholder='Select a GP Practice',
                    style={"width": "100%"}
                )
            ), width=12)
        ]),

        dbc.Row([
            dbc.Col(html.H2(id='ics-name', className="text-center my-4", style={"color": "#343A40", "fontWeight": "bold"}))
        ]),
        
        dbc.Row([
            dbc.Col(create_card(
                "Overall Experience Satisfaction Distribution", 
                dcc.Graph(id='satisfaction-histogram')
            ), width=6),
            dbc.Col(create_card(
                "Overall Satisfaction vs GP Contact Satisfaction", 
                dcc.Graph(id='satisfaction-scatterplot')
            ), width=6),
        ]),

        dbc.Row([
            dbc.Col(create_card(
                "Select Data Level", 
                dcc.Dropdown(
                    id='data-level-dropdown', 
                    options=[
                        {'label': 'National', 'value': 'national'},
                        {'label': 'ICS Level', 'value': 'ics'}
                    ],
                    placeholder='Select Data Level',
                    style={"width": "100%"}
                )
            ), width=12),
        ]),

        dbc.Row([
            dbc.Col(create_card(
                "1. Where to start? Improving the front door experience", 
                html.Div(id='r-squared-explanation', className="text-justify")
            ), width=12),
        ]),

        dbc.Row([
            dbc.Col(create_card(
                "2. What to prioritize? What affects front door satisfaction?",
                html.Div([
                    html.Div(
                        "A prediction model for front door satisfaction, based on responses to all relevant questions in the survey questionnaire identifies six features as most powerful predictors.",
                        style={"color": "black", "fontWeight": "normal", "marginBottom": "15px"}
                    ),
                    dash_table.DataTable(
                        id='table',
                        columns=[
                            {"name": "Predictor of overall satisfaction with front door experience", "id": "Predictor"},
                            {"name": "Predictive power", "id": "Predictive power", "type": "numeric", "format": {"specifier": ".4f"}},
                            {"name": "Ranking in ICS", "id": "Ranking"}
                        ],
                        style_table={'width': '100%', 'margin': 'auto'},
                        style_cell={'textAlign': 'center', 'padding': '10px'},
                        style_cell_conditional=[
                            {'if': {'column_id': 'Predictor'}, 'textAlign': 'left'}
                        ],
                        style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}
                        ]
                    ),
                    html.Div(
                        id='r-squared-text',
                        style={"marginTop": "20px", "fontSize": "1.2rem", "color": "black", "textAlign": "left"}
                    )
                ])
            ), width=12),
        ]), 

        dbc.Row([
            dbc.Col(create_card(
                "Select Predictor", 
                dcc.Dropdown(
                    id='predictor-dropdown', 
                    options=[{'label': key, 'value': value} for key, value in predictor_mapping.items()],
                    placeholder='Select a Predictor',
                    style={"width": "100%"}
                )
            ), width=6),
            dbc.Col(create_card(
                "Scatter Plot for Selected Predictor Satisfaction", 
                dcc.Graph(id='ics-predictor-scatterplot')
            ), width=6),
            dbc.Col(create_card(
                "3. Moving beyond the front door", 
                html.Div(id='effect-text', className="text-center my-3", style={"fontSize": "1.25rem", "fontWeight": "bold"})
            ), width=12)
        ]),

        dbc.Row([
            dbc.Col(create_card(
                "4. Continuity of Care as a service design principle", 
                html.Div([
                    dcc.Graph(id='continuity-care-scatterplot'),
                    html.Div(id='continuity-care-text', className="text-justify", style={"marginTop": "20px", "fontSize": "1.2rem"})
                ])
            ), width=12)
        ])
    ],
    style={"padding": "20px"}
)

# Callback functions
@app.callback(
    Output('satisfaction-histogram', 'figure'),
    [Input('data-level-dropdown', 'value'), Input('practice-dropdown', 'value')]
)
def update_histogram(data_level, selected_practice):
    if data_level == 'ics' and selected_practice:
        selected_ics = df_nonan.loc[df_nonan['ad_practicename'] == selected_practice, 'ad_icsname'].values[0]
        histogram_data = df_nonan[df_nonan['ad_icsname'] == selected_ics]['overallexp.pcteval']
        title = f'Histogram of Overall Experience Satisfaction for {selected_ics}'
    else:
        histogram_data = df_nonan['overallexp.pcteval']
        title = 'Histogram of Overall Experience Satisfaction (National)'

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=histogram_data, nbinsx=20))
    fig.update_layout(
        title=title,
        xaxis_title='Overall Experience Satisfaction (%)',
        yaxis_title='Frequency',
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor='#f9f9f9'
    )
    return fig

@app.callback(
    [Output('satisfaction-scatterplot', 'figure'),
     Output('r-squared-explanation', 'children')],
    [Input('practice-dropdown', 'value'),
     Input('data-level-dropdown', 'value')]
)
def update_scatterplot_and_r_squared(selected_practice, data_level):
    fig = go.Figure()

    if data_level == 'ics' and selected_practice:
        selected_ics = df_nonan.loc[df_nonan['ad_practicename'] == selected_practice, 'ad_icsname'].values[0]
        scatter_data = df_nonan[df_nonan['ad_icsname'] == selected_ics]
        level_text = "ICS"
        practice_count = count_practices(scatter_data)
    else:
        scatter_data = df_nonan
        level_text = "Nation"
        practice_count = count_practices(df_nonan)

    fig.add_trace(go.Scatter(
        x=scatter_data['gpcontactoverall.pcteval'],
        y=scatter_data['overallexp.pcteval'],
        mode='markers',
        marker=dict(
            color='rgba(0, 0, 255, 0)',
            line=dict(width=2, color='lightblue'),
            size=10
        ),
        hoverinfo='text',
        text=scatter_data['ad_practicename'],
        showlegend=False
    ))

    ranking = scatter_data['gpcontactoverall.pcteval'].rank(ascending=False, method='min').astype(int)
    selected_gp_rank = ranking[scatter_data['ad_practicename'] == selected_practice].values[0] if selected_practice else None

    if selected_practice:
        highlight = df_nonan[df_nonan['ad_practicename'] == selected_practice]
        fig.add_trace(go.Scatter(
            x=highlight['gpcontactoverall.pcteval'],
            y=highlight['overallexp.pcteval'],
            mode='markers',
            marker=dict(color='red', size=12, line=dict(width=2, color='DarkSlateGrey')),
            name=selected_practice,
            showlegend=True
        ))

    fig.update_layout(
        title="Overall Patient Satisfaction vs. GP Contact Satisfaction",
        xaxis_title='GP Contact Satisfaction (%)',
        yaxis_title='Overall Patient Satisfaction (%)',
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor='#f9f9f9'
    )

    r_squared = calculate_r_squared(scatter_data['gpcontactoverall.pcteval'], scatter_data['overallexp.pcteval'])
    r_squared_percent = int(r_squared * 100)

    explanation = (
        f"A section of the survey questionnaire refers to the respondent's last contact with the practice. "
        f"One of the questions in this section evaluates the overall front door experience: "
        f"\"Overall, how would you describe your experience of contacting your GP practice on this occasion?\", "
        f"with answers ranging from \"very good\" to \"very poor\". The front door satisfaction score is the "
        f"proportion of patients who assessed their experience as \"very good\" or \"fairly good\".\n\n"
        f"A multivariate analysis showed that satisfaction with the front door experience is the most powerful "
        f"predictor of overall patient satisfaction, far surpassing the predictive power of any other question "
        f"in the survey. The above graph illustrates this strong relationship for the {practice_count} practices "
        f"in the {level_text}. The front door experience explains {r_squared_percent}% of the variation in "
        f"overall patient satisfaction in the {level_text}."
    )

    if selected_practice and selected_gp_rank:
        explanation += f"\n\nOn front door satisfaction, {selected_practice} is ranked {selected_gp_rank}th out of {practice_count} practices."

    return fig, explanation

@app.callback(
    [Output('table', 'data'), Output('ics-name', 'children')],
    [Input('practice-dropdown', 'value')]
)
def update_table_and_ics(selected_practice):
    if selected_practice:
        selected_ics = df_nonan.loc[df_nonan['ad_practicename'] == selected_practice, 'ad_icsname'].values[0]
        ics_data = df_nonan[df_nonan['ad_icsname'] == selected_ics]

        ics_data_nonan = ics_data.dropna(subset=X_columns + ['gpcontactoverall.pcteval'])

        if not ics_data_nonan.empty:
            X = ics_data_nonan[X_columns]
            y = ics_data_nonan['gpcontactoverall.pcteval']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            coefficients = pd.DataFrame({
                "Predictor": [
                    "Waiting time: How do you feel about how long you waited for your appointment?", 
                    "Phone: Generally, how easy or difficult is it to contact your GP practice on the phone?", 
                    "Online access: Generally, how easy or difficult is it to contact your GP practice using their website?", 
                    "Reception: Overall, how helpful do you find the reception and administrative team at your GP practice?", 
                    "Information quality: Did you know what the next step in dealing with your request would be?", 
                    "Choice: Were you offered a choice of time or day for your appointment?"
                ],
                "Predictive power": model.params[1:]
            })

            selected_gp_data = ics_data_nonan[ics_data_nonan['ad_practicename'] == selected_practice]
            rankings = {}
            for predictor in X_columns:
                rankings[predictor] = ics_data_nonan[predictor].rank(ascending=False, method='min')[selected_gp_data.index[0]]

            coefficients['Ranking'] = coefficients['Predictor'].map(lambda x: f"{int(rankings[X_columns[coefficients.index[coefficients['Predictor'] == x][0]]])} / {len(ics_data_nonan)}")

            ics_message = f"ICS: {selected_ics}"

            return coefficients.to_dict('records'), ics_message

    return [], "Select a GP to view the ICS-level data."

@app.callback(
    [Output('ics-predictor-scatterplot', 'figure'), 
     Output('effect-text', 'children'),
     Output('r-squared-text', 'children')],
    [Input('practice-dropdown', 'value'), 
     Input('predictor-dropdown', 'value')]
)
def update_predictor_scatterplot(selected_practice, selected_predictor):
    if selected_practice and selected_predictor:
        selected_ics = df_nonan.loc[df_nonan['ad_practicename'] == selected_practice, 'ad_icsname'].values[0]
        ics_data = df_nonan[df_nonan['ad_icsname'] == selected_ics]

        ics_data_nonan = ics_data.dropna(subset=X_columns + ['gpcontactoverall.pcteval'])

        X = ics_data_nonan[X_columns]
        y = ics_data_nonan['gpcontactoverall.pcteval']
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        r_squared_six_predictors = model.rsquared * 100

        df_mediation = ics_data[['overallexp.pcteval', selected_predictor, 'gpcontactoverall.pcteval']].copy()
        df_mediation.columns = ['OverallSatisfaction', 'Predictor', 'FrontDoorExp']

        model_a = ols("FrontDoorExp ~ Predictor", data=df_mediation).fit()
        model_c = ols("OverallSatisfaction ~ FrontDoorExp + Predictor", data=df_mediation).fit()

        indirect_effect = model_a.params['Predictor'] * model_c.params['FrontDoorExp']
        direct_effect = model_c.params['Predictor']
        total_effect = indirect_effect + direct_effect

        percentage_indirect = int((indirect_effect / total_effect) * 100)
        percentage_direct = int((direct_effect / total_effect) * 100)

        effect_type = "positive" if direct_effect > 0.1 else "limited"

        readable_predictor = [key for key, value in predictor_mapping.items() if value == selected_predictor][0]

        effect_text = html.Div(
            children=(
                f"Components of the front door service, such as the reception team, the phone system or the online appointments process impact overall satisfaction through their positive effect on a patient's front door satisfaction. "
                f"However, they can also have a direct effect on overall satisfaction, beyond a patient's satisfaction with the front door service itself.\n\n"
                f"The {readable_predictor} has a {effect_type} effect directly on overall satisfaction, even after removing its influence on front door experience. "
                f"In fact, the data suggests that {percentage_indirect}% of the receptionist team's impact on overall patient satisfaction is attributable to its effect on front door experience, "
                f"with the remaining {percentage_direct}% being a direct effect on overall patient experience, beyond their satisfaction with the experience of contacting the practice. "
                f"The bigger the direct effect is, the bigger it contributes to the overall patient satisfaction."
            ),
            style={
                "textAlign": "left",  
                "fontWeight": "normal",  
                "whiteSpace": "pre-line"  
            }
        )

        r_squared_text = (
            f"The responses to these six questions explain over {r_squared_six_predictors:.2f}% of the variation in overall front door satisfaction between practices in the ICS. "
            f"The following plot illustrates the strong relationship between the selected predictor and overall front-door satisfaction."
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ics_data[selected_predictor],
            y=ics_data['gpcontactoverall.pcteval'],
            mode='markers',
            marker=dict(size=10, color='lightblue', line=dict(width=1, color='blue')),
            text=ics_data['ad_practicename'],
            hoverinfo='text',
            showlegend=False
        ))

        highlight = ics_data[ics_data['ad_practicename'] == selected_practice]
        if not highlight.empty:
            fig.add_trace(go.Scatter(
                x=highlight[selected_predictor],
                y=highlight['gpcontactoverall.pcteval'],
                mode='markers',
                marker=dict(color='red', size=12, line=dict(width=2, color='DarkSlateGrey')),
                name=selected_practice,
                showlegend=True
            ))

        fig.update_layout(
            title=f"Scatter Plot for {readable_predictor} satisfaction",
            xaxis_title=f"Proportion for {readable_predictor}",
            yaxis_title='Overall Frontdoor Satisfaction (%)',
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='#f9f9f9'
        )

        return fig, effect_text, r_squared_text

    return go.Figure(), "Please select a GP practice and predictor.", ""

@app.callback(
    Output('continuity-care-scatterplot', 'figure'),
    Output('continuity-care-text', 'children'),
    [Input('practice-dropdown', 'value')]
)
def update_continuity_care_scatterplot(selected_practice):
    if selected_practice:
        selected_ics = df_nonan.loc[df_nonan['ad_practicename'] == selected_practice, 'ad_icsname'].values[0]
        ics_data = df_nonan[df_nonan['ad_icsname'] == selected_ics]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ics_data['prefhp_interaction'],
            y=ics_data['overallexp.pcteval'],
            mode='markers',
            marker=dict(size=10, color='lightblue', line=dict(width=1, color='blue')),
            text=ics_data['ad_practicename'],
            hoverinfo='text',
            showlegend=False
        ))

        highlight = ics_data[ics_data['ad_practicename'] == selected_practice]
        if not highlight.empty:
            fig.add_trace(go.Scatter(
                x=highlight['prefhp_interaction'],
                y=highlight['overallexp.pcteval'],
                mode='markers',
                marker=dict(color='red', size=12, line=dict(width=2, color='DarkSlateGrey')),
                name=selected_practice,
                showlegend=True
            ))

        fig.update_layout(
            title=f"Continuity of Care and Overall patient satisfaction for {selected_practice}",
            xaxis_title='Continuity of Care (Interaction)',
            yaxis_title='Overall Patient Satisfaction (%)',
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='#f9f9f9'
        )

        coc_score = int(highlight['prefhp_interaction'].values[0] * 100)

        continuity_care_text = (
            f"The survey provides estimates of the proportion of patients who (a) said that they have a preferred healthcare professional in the practice "
            f"and (b) said that they see this preferred professional most of the time. The relationship between continuity of care and overall patient satisfaction "
            f"shows an interesting pattern, akin to a tipping point.\n\n"
            f"There does not seem to be a strong correlation below 20-30%, but all ICS practices that achieve 20-30% continuity of care or above achieve high overall "
            f"patient satisfaction. {selected_practice}'s COC score is at {coc_score}%."
        )

        return fig, continuity_care_text

    return go.Figure(), "Please select a GP practice to view Continuity of Care data."

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)