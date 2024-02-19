import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Dashboard Life Expectancy", page_icon=":bar_chart:", layout="wide")
with open('style.css') as source_style:
    st.markdown(f'<style>{source_style.read()}</style>', unsafe_allow_html=True)

shape_path = "E:\IT\Code\BI\cb_2016_us_nation_5m\cb_2016_us_nation_5m.shp"


@st.cache_data
def load_data():
    df = pd.read_csv("Life_Expectancy_Data_Fix.csv")
    return df

df= load_data()
st.markdown("<h1 class='labels-for-title'>Dashboard Life Expectancy</h1>", unsafe_allow_html=True)

st.sidebar.header("Filter Data Here: ")
continent = st.sidebar.multiselect(
    "Select the Continent: ",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)
year = st.sidebar.multiselect(
    "Select the Years: ",
    options=df['Year'].unique(),
    default=df['Year'].unique()
)
status = st.sidebar.multiselect(
    "Select the Status: ",
    options=df['Status'].unique(),
    default=df['Status'].unique()
)

df_selection = df.query(
    "Region == @continent & Status == @status & Year == @year"
)


Checkbox = st.sidebar.checkbox("Display Dataset")
def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 28,
            },
            title={
                "text": label,
                "font": {"size": 16},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=random.sample(range(0, 101), 30),
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        paper_bgcolor='#5e697a',
        # paper_bgcolor="lightgrey",
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        height=100
    )

    st.plotly_chart(fig, use_container_width=True)

def top_left_plot():
    grouped_data = df_selection.groupby('Region')[['Adult Mortality', 'infant deaths', 'under-five deaths ']].sum().reset_index()

    melted_data = pd.melt(grouped_data, id_vars='Region', var_name='Mortality Type', value_name='Total Deaths')

    fig = px.bar(melted_data, x='Region', y='Total Deaths', color='Mortality Type',
                labels={'Total Deaths': 'Total Deaths'},
                category_orders={'Mortality Type': ['Adult Mortality', 'infant deaths', 'under-five deaths ']},
                color_discrete_map={'Adult Mortality': '#e69c09', 'infant deaths': '#e6ed24', 'under-five deaths ': '#3af026'}
                )
    fig.update_xaxes(tickmode='linear', dtick=1)
    fig.update_layout(
        paper_bgcolor='#5e697a',
        plot_bgcolor='rgba(0,0,0,0)',
        #yaxis=dict(showgrid=False),
        #xaxis=dict(showgrid=False), 
        title = "☠  Number of Death"
    )
    st.plotly_chart(fig, use_container_width=True, width=100,height=50)
def life_expectancy_by_status():
    avg_life_expectancy = df_selection.groupby('Status')['Life expectancy '].mean().reset_index()
    fig = go.Figure(data=[go.Pie(labels=avg_life_expectancy['Status'], 
                                 values=avg_life_expectancy['Life expectancy '].round(2), 
                                 hole=.3,title="⚪", 
                                 textinfo='percent+value')])

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.1,
        xanchor="center",
        x=0.5),  
        #paper_bgcolor='#5e697a',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=False),
        width = 400,
        height = 330, title = "Average Life Expectancy by Status"
    )

    st.plotly_chart(fig, use_container_width=True)
def gdp_by_year():
    avg_gdp = df_selection.groupby('Year')['GDP'].mean().reset_index()

    fig = px.area(avg_gdp, x='Year', y='GDP')
    fig.update_layout(
        title='Average GDP Over Years',
        xaxis_title='Year',
        yaxis_title='Average GDP',
        #markers = True,
        #text = avg_gdp,
        showlegend=True
    )
    fig.update_xaxes(tickmode='linear', dtick=1, tickfont_color='white')
    fig.update_yaxes(tickfont_color='white')
    fig.update_layout(
        #paper_bgcolor='#5e697a',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=False)
    )
    # Add markers with labels
    fig.add_trace(go.Scatter(
        x=avg_gdp['Year'],
        y=avg_gdp['GDP'],
        mode='markers+text',
        text=avg_gdp['GDP'].astype(int),
        textposition='top center',
        marker=dict(size=8, color='#0a698c', symbol='circle'),
        hoverinfo='text',
        name="Average GDP",
    ))
    st.plotly_chart(fig, use_container_width=True)
def data_by_year(selected_columns):
    avg_data = df_selection.groupby('Year')[selected_columns].mean().reset_index()

    fig = px.area(avg_data, x='Year', y=selected_columns)
    fig.update_layout(
        title="Average " + selected_columns + " Over Years",
        xaxis_title='Year',
        yaxis_title='Average ' + selected_columns,
        #markers = True,
        #text = avg_gdp,
        showlegend=True
    )
    fig.update_xaxes(tickmode='linear', dtick=1, tickfont_color='white')
    fig.update_yaxes(tickfont_color='white')
    fig.update_layout(
        #paper_bgcolor='#5e697a',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=False)
    )
    # Add markers with labels
    fig.add_trace(go.Scatter(
        x=avg_data['Year'],
        y=avg_data[selected_columns],
        mode='markers+text',
        text=avg_data[selected_columns].astype(int),
        textposition='top center',
        marker=dict(size=8, color='#0a698c', symbol='circle'),
        hoverinfo='text',
        name="Average " + selected_columns,
    ))
    st.plotly_chart(fig, use_container_width=True)    
def forecast_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)

    future_years = np.arange(X['Year'].max() + 1, X['Year'].max() + 7).reshape(-1, 1)
    predicted_values = model.predict(future_years)

    return future_years.flatten(), predicted_values
def forecast_data_plot(selected_columns):
    forecast_data = df_selection.groupby('Year')[selected_columns].mean().reset_index()

    X = forecast_data[['Year']]
    y = forecast_data[selected_columns]
    future_years, predicted_values = forecast_linear_regression(X, y)

    fig = px.area(forecast_data, x='Year', y=selected_columns)
    fig.update_layout(
        title=f'Average {selected_columns} Over Years with Forecasting',
        xaxis_title='Year',
        yaxis_title=f'Average {selected_columns}',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=False)
    )
    fig.update_xaxes(tickmode='linear', dtick=1)


    # Add markers with labels
    fig.add_trace(go.Scatter(
        x=forecast_data['Year'],
        y=forecast_data[selected_columns],
        mode='markers+text',
        text=forecast_data[selected_columns].astype(int),
        textposition='top center',
        marker=dict(size=8, color='#0a698c', symbol='circle'),
        hoverinfo='text',
        name=f"Average {selected_columns}",
    ))

    # Add forecast line
    fig.add_trace(go.Scatter(
        x=future_years,
        y=predicted_values,
        mode='lines',
        line=dict(color='#a63430', dash='solid'),
        name=f"Forecast {selected_columns}",
    ))

    st.plotly_chart(fig, use_container_width=True)
def life_expectancy_top_by_region():
    avg_top_data = df_selection.groupby(by=["Region"]).mean()[["Life expectancy "]].sort_values(by="Life expectancy ")
    fig = px.bar (
        avg_top_data, x = "Life expectancy ",
        y = avg_top_data.index,
        orientation="h",
        text_auto = True,
        color_discrete_sequence=["#0a698c"] * len(avg_top_data)
    )
    fig.update_layout(
        title = "Best Region by Life Expectancy",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='#5e697a',
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)
def pie_for_heatlhCare():
    selected_columns = ['Hepatitis B', 'Measles ', 'Polio', 'Diphtheria ', ' HIV/AIDS']

    average_values = df_selection[selected_columns].mean()
    average_values.index = average_values.index.astype(str)

    fig = px.pie(
        values=average_values,
        names=average_values.index,
        title='Average Distribution of Health Conditions', 
        #color_discrete_map={'Hepatitis B': '#0a698c',
                            #'Diphtheria ': '#a63430',
                            #' HIV/AIDS': '#b5a238',
                            #'Measles ': '#2a9649',
                            #'Polio': '#b57712',
                            #}
        color_discrete_sequence=['#0a698c', '#a63430', '#b5a238', '#2a9649', '#b57712']
    )
    #fig.update_traces(textposition = 'inside')
    fig.update_layout(
        width = 400,
        height = 330
    )

    st.plotly_chart(fig, use_container_width=True)

def line_for_gdp():
    avg_life_expectancy = df_selection.groupby(['Year', 'Region'])['Life expectancy '].mean().reset_index()
    fig_life_expectancy = px.line(avg_life_expectancy, 
                                  x='Year', 
                                  y='Life expectancy ', 
                                  color='Region',
                                  markers = True,
                                  title='Average Life Expectancy Over Years by Region',
                                  color_discrete_map={'Asia': '#0a698c',
                                                      'Afrika': '#a63430',
                                                      'Oceania': '#b5a238',
                                                      'Europe': '#2a9649',
                                                      'North America': '#b57712',
                                                      'South America': '#18a387',
                                                      }
                                    )
    fig_life_expectancy.update_xaxes(tickmode='linear', dtick=1)
    fig_life_expectancy.update_layout(
        paper_bgcolor='#5e697a',
        plot_bgcolor='rgba(0,0,0,0)',
        #yaxis=dict(showgrid=False),
        #xaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_life_expectancy, use_container_width=True, width=600, height=400)

top_left_columns, top_mid_columns,top_right_columns = st.columns((1,1,2))
bottom_left_columns, bottom_right_columns = st.columns(2)

with top_left_columns:
    top_left_plot()
with top_mid_columns:
    life_expectancy_top_by_region()
with top_right_columns:
    top2_left_columns, top2_mid1_columns,top2_mid2_columns, top2_right_columns = st.columns(4)
    bottom2_left_columns, bottom2_right_columns = st.columns(2)

    avg_life_expectancy = df_selection['Life expectancy '].mean()
    total_population = df_selection['Population'].sum()
    total_expenditure = df_selection['Total expenditure'].sum()
    avg_schooling = df_selection['Schooling'].mean()
    with top2_left_columns:
        plot_metric(
            "Average Life Expectancy",
            avg_life_expectancy,
            prefix="",
            suffix="",
            show_graph=True,
            color_graph="#0a698c",
        )
    with top2_mid1_columns:
        plot_metric(
            "Total Population",
            total_population,
            prefix="",
            suffix="",
            show_graph=True,
            color_graph="#a63430",
        )
    with top2_mid2_columns:
        plot_metric(
            "Total Expenditure",
            total_expenditure,
            prefix="",
            suffix="",
            show_graph=True,
            color_graph="#b5a238",
        )
    with top2_right_columns:
        plot_metric(
            "🏫Average Schooling",
            avg_schooling,
            prefix="",
            suffix=" year",
            show_graph=True,
            color_graph="#2a9649",
        )
    
    with bottom2_left_columns:
        life_expectancy_by_status()
    with bottom2_right_columns:
        pie_for_heatlhCare()

with bottom_left_columns:
    line_for_gdp()
with bottom_right_columns:
    gdp_by_year()

def table():
    with st.expander("Database Table"):
        showData = st.multiselect("Filter Data: ",
                                  df_selection.columns,
                                  default=["Country","Year", "Region","Status","Life expectancy ", "Adult Mortality", 
                                           "infant deaths", "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ", " BMI ",                                
                                            "under-five deaths ", "Polio", "Total expenditure", "Diphtheria "," HIV/AIDS","GDP", "Population", 
                                            " thinness  1-19 years", " thinness 5-9 years", "Income composition of resources", "Schooling"
                                           ]
                                  )
        st.dataframe(df_selection[showData], use_container_width=True)
table()

numeric_columns = df_selection.select_dtypes(include=np.number).columns.tolist()
numeric_columns_except_year = [col for col in numeric_columns if col != 'Year']
select_data = st.selectbox("Choose Columns: ", numeric_columns_except_year, index = 13, placeholder="Select Columns to ForeCasting")
fore_columns_left, fore_columns_right = st.columns(2)
with fore_columns_left:
    data_by_year(select_data)
with fore_columns_right:
    forecast_data_plot(select_data)       

if Checkbox:
    desc_columns, tabel_columns = st.columns(2)
    with desc_columns:
        st.markdown("<h3>Data Describe</h3>", unsafe_allow_html=True)
        st.write(df_selection.describe().T)
    with tabel_columns:
        st.markdown("<h3>Data Corelation</h3>", unsafe_allow_html=True)
        st.write(df_selection.corr())



# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
    
