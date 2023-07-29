import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler #standardisation
from sklearn.ensemble import RandomForestRegressor

st.markdown(back, unsafe_allow_html=True)

original = pd.read_csv(r'https://raw.githubusercontent.com/wernhuit/life-expectancy/main/Life%20Expectancy%20Data.csv?token=GHSAT0AAAAAACFU7G72VMSR3WEMVMATFZOIZGFAJMA')
preview = original.head(20)

st.image('https://www.incimages.com/uploaded_files/image/1920x1080/getty_178026691_20001333181884387489_201886.jpg', caption='Credits: https://www.inc.com/nicolas-cole/30-things-about-life-everyone-should-learn-before-turning-30.html')

st.write("""
# Life Expectancy Prediction
Let's predict _life expectancy_!
Slide the sliders on the left sidebar to predict :)
""")

with st.expander("**Background**"):
    st.write("""_Life Expectancy_ is the average number of years a person is expected to live for based on demographics. 
    This model is created to help insurance companies (target user) predict the life expectancy in age for their clients and then suggest the appropriate policies to them.""")
with st.expander("**Features for prediction**"):
    st.write("""
    1. alcohol: Alcohol consumption recorded in litres of pure alcohol per capita (15+ years old)
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
with st.expander('**The Dataset**'):
    st.write('This dataset can be found on Kaggle, https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who, and is from the World Health Organisation (WHO).')
    st.write('Below is the preview of the ORIGINAL dataset: ', preview)
with st.expander('**The Model**'):
    st.write('We will be using _RandomForestRegressor_ for our model! It is a bagging ensemble learning algorithm that combines multiple decision trees to create a more accurate model.')

st.sidebar.header('Input Desired Parameters Here!')

def user_input_features():
    alcohol = st.sidebar.slider('Alcohol Consumption in litres', 0.01, 13.94, 0.01)
    hepatitis_b = st.sidebar.slider('HepB Immunization Coverage in %', 2, 99, 65)
    measles = st.sidebar.slider('Measles Cases', 0, 863, 7)
    polio = st.sidebar.slider('Pol3 Immunization Coverage in %', 3, 99, 6)
    total_expenditure = st.sidebar.slider('Total Expenditure on Health in %', 1.21, 11.78, 8.16)
    diphtheria = st.sidebar.slider('DTP3 Immunization Coverage in %', 2, 99, 65)
    hiv_aids = st.sidebar.slider('HIV/AIDS Death Rate in %', 0.1, 31.9, 0.1)
    gdp = st.sidebar.slider('GDP in USD', 21.362, 66775.394, 584.259)
    schooling = st.sidebar.slider('Average Schooling Years', 2.8, 20.7, 10.1)
    thinness_5_19_years = st.sidebar.slider('Prevalence of Thinness in %', 0.1, 28.15, 17.25)
    data = {'alcohol': alcohol,
            'hepatitis_b': hepatitis_b,
            'measles': measles ** (1/3), #cube root transformation
            'polio': polio,
            'total_expenditure': total_expenditure,
            'diphtheria': diphtheria,
            'hiv/aids': hiv_aids ** (1/3),
            'gdp': gdp,
            'schooling': schooling,
            'thinness_5-19_years': thinness_5_19_years}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Parameters Inputed: ')
st.write(df)

life = pd.read_csv(r'https://raw.githubusercontent.com/wernhuit/life-expectancy/main/final_clean_transformed_dataset.csv?token=GHSAT0AAAAAACFU7G72DTANKLRPQGQYWECCZGFAKKA')

y = life['life_expectancy']
X = life.iloc[:,2:]

#standardise all columns before modelling (this will ensure columns with different ranges of values can be comparable with each other)
scaler = StandardScaler() #create scaler object
X = scaler.fit_transform(X)
df = scaler.transform(df) #use same scaler to standardise parameters too

RFR = RandomForestRegressor( #these were the best parameters from gridsearch
    max_depth=10,
    min_samples_leaf=4,
    min_samples_split=5,
    n_estimators=1000,
    random_state=7)

RFR.fit(X, y)

prediction = RFR.predict(df)

st.subheader('Predicted Life Expectancy')
st.subheader(f':violet[{round(prediction[0],1)}]')
