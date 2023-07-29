import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Life Expectancy Prediction
Let's predict **life expectancy**!
""")

with st.expander("Background"):
    st.write("""Life Expectancy is the average number of years a person is expected to live for based on demographics. 
    This model is created to help insurance companies predict the life expectancy in age for their clients and then suggest the appropriate policies to them.""")
    with st.expander("Features for prediction"):
        st.write("""1. alcohol: Alcohol consumption recorded in litres of pure alcohol per capita (15+ years old)
        2. hepatitis_b: Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
        3. measles: Number of reported Measles cases per 1000 population
        4. polio: Polio (Pol3) immunization coverage among 1-year-olds (%)
        5. total_expenditure: General government expenditure on health as a percentage of total government expenditure (%)
        6. diphtheria: Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
        7. hiv/aids: Death rate per 1000 live births HIV/AIDS (0-4 years)
        8. gdp: Gross Domestic Product per capita (in USD)
        9. schooling: Average number of years spent on schooling
        10. thinness_5-19_years: Prevalence of thinness among children and adolescents for Age 5 to 19 (%)
        """)
