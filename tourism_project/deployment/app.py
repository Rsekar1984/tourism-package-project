import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Exact column order used during training — NEVER change this list
FEATURES = [
    'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation', 'Gender',
    'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

@st.cache_resource(show_spinner='Loading model from Hugging Face...')
def load_model():
    path = hf_hub_download(
        repo_id='rknv1984/tourism-project-model',
        filename='best-tourism-model-v1.joblib',
        repo_type='model'
    )
    return joblib.load(path)

st.set_page_config(page_title='Tourism Package Prediction', page_icon='🏖️', layout='wide')
st.title('🏖️ Tourism Package Purchase Prediction')
st.markdown('Predict whether a customer will purchase the **Wellness Tourism Package**.')
st.divider()

model = load_model()
st.success('✅ Model loaded successfully!')

st.subheader('Enter Customer Details')
col1, col2, col3 = st.columns(3)

with col1:
    Age                    = st.number_input('Age', 18, 100, 30)
    CityTier               = st.selectbox('City Tier', [1, 2, 3])
    NumberOfPersonVisiting = st.number_input('Persons Visiting', 1, 10, 2)
    PreferredPropertyStar  = st.selectbox('Preferred Property Star', [1, 2, 3, 4, 5])
    NumberOfTrips          = st.number_input('Trips per Year', 0, 20, 2)
    Passport               = st.selectbox('Has Passport?', [0, 1], format_func=lambda x: 'Yes' if x else 'No')

with col2:
    OwnCar                   = st.selectbox('Owns a Car?', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    NumberOfChildrenVisiting = st.number_input('Children Visiting (<5 yrs)', 0, 5, 0)
    MonthlyIncome            = st.number_input('Monthly Income (INR)', 5000, 500000, 50000, step=1000)
    TypeofContact            = st.selectbox('Type of Contact', ['Company Invited', 'Self Enquiry'])
    Occupation               = st.selectbox('Occupation', ['Free Lancer', 'Large Business', 'Self Employed', 'Small Business', 'Salaried'])
    Gender                   = st.selectbox('Gender', ['Female', 'Male'])

with col3:
    MaritalStatus          = st.selectbox('Marital Status', ['Divorced', 'Married', 'Single'])
    Designation            = st.selectbox('Designation', ['AVP', 'Executive', 'Manager', 'Senior Manager', 'VP'])
    PitchSatisfactionScore = st.slider('Pitch Satisfaction Score', 1, 5, 3)
    ProductPitched         = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'King', 'Standard', 'Super Deluxe'])
    NumberOfFollowups      = st.number_input('Number of Follow-ups', 0, 10, 2)
    DurationOfPitch        = st.number_input('Duration of Pitch (mins)', 1, 120, 15)

st.divider()
if st.button('🔮 Predict Purchase Likelihood', use_container_width=True):
    input_data = pd.DataFrame([{
        'Age': Age,
        'TypeofContact':  0 if TypeofContact == 'Company Invited' else 1,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': {'Free Lancer':0,'Large Business':1,'Self Employed':2,'Small Business':3,'Salaried':4}[Occupation],
        'Gender': 0 if Gender == 'Female' else 1,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': {'Basic':0,'Deluxe':1,'King':2,'Standard':3,'Super Deluxe':4}[ProductPitched],
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': {'Divorced':0,'Married':1,'Single':2}[MaritalStatus],
        'NumberOfTrips': NumberOfTrips,
        'Passport': Passport,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': OwnCar,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': {'AVP':0,'Executive':1,'Manager':2,'Senior Manager':3,'VP':4}[Designation],
        'MonthlyIncome': MonthlyIncome,
    }])[FEATURES]   # enforce exact trained column order

    prob = model.predict_proba(input_data)[0, 1]

    if prob >= 0.45:
        st.success(f'✅ Customer LIKELY to purchase the Wellness Tourism Package!')
        st.balloons()
    else:
        st.error(f'❌ Customer UNLIKELY to purchase the Wellness Tourism Package.')
    st.metric(label='Purchase Probability', value=f'{prob:.2%}')
