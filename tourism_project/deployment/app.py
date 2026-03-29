import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download trained model from HF Hub
model_path = hf_hub_download(
    repo_id='rknv1984/tourism-project-model',
    filename='best-tourism-model-v1.joblib'
)
model = joblib.load(model_path)

st.title('Tourism Package Purchase Prediction')
st.write('Predict whether a customer will purchase the Wellness Tourism Package.')

# Customer Details
st.subheader('Customer Details')
Age                      = st.number_input('Age', min_value=18, max_value=100, value=30)
CityTier                 = st.selectbox('City Tier', [1, 2, 3])
NumberOfPersonVisiting   = st.number_input('Number of Persons Visiting', min_value=1, max_value=10, value=2)
PreferredPropertyStar    = st.selectbox('Preferred Property Star', [1, 2, 3, 4, 5])
NumberOfTrips            = st.number_input('Number of Trips per Year', min_value=0, max_value=20, value=2)
Passport                 = st.selectbox('Has Passport?', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
OwnCar                   = st.selectbox('Owns a Car?', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
NumberOfChildrenVisiting = st.number_input('Number of Children Visiting', min_value=0, max_value=5, value=0)
MonthlyIncome            = st.number_input('Monthly Income', min_value=5000, max_value=500000, value=50000)

# Categorical inputs encoded to match LabelEncoder mapping in prep.py
TypeofContact = st.selectbox('Type of Contact', ['Company Invited', 'Self Enquiry'])
TypeofContact_enc = 0 if TypeofContact == 'Company Invited' else 1

Occupation = st.selectbox('Occupation', ['Free Lancer', 'Large Business', 'Self Employed', 'Small Business', 'Salaried'])
occ_map = {'Free Lancer':0, 'Large Business':1, 'Self Employed':2, 'Small Business':3, 'Salaried':4}
Occupation_enc = occ_map[Occupation]

Gender = st.selectbox('Gender', ['Female', 'Male'])
Gender_enc = 0 if Gender == 'Female' else 1

MaritalStatus = st.selectbox('Marital Status', ['Divorced', 'Married', 'Single'])
marital_map = {'Divorced':0, 'Married':1, 'Single':2}
MaritalStatus_enc = marital_map[MaritalStatus]

Designation = st.selectbox('Designation', ['AVP', 'Executive', 'Manager', 'Senior Manager', 'VP'])
desig_map = {'AVP':0, 'Executive':1, 'Manager':2, 'Senior Manager':3, 'VP':4}
Designation_enc = desig_map[Designation]

# Customer Interaction Details
st.subheader('Customer Interaction Details')
PitchSatisfactionScore = st.slider('Pitch Satisfaction Score', min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'King', 'Standard', 'Super Deluxe'])
prod_map = {'Basic':0, 'Deluxe':1, 'King':2, 'Standard':3, 'Super Deluxe':4}
ProductPitched_enc = prod_map[ProductPitched]

NumberOfFollowups = st.number_input('Number of Follow-ups', min_value=0, max_value=10, value=2)
DurationOfPitch   = st.number_input('Duration of Pitch (minutes)', min_value=1, max_value=120, value=15)

# Build input DataFrame — exact column order must match training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'TypeofContact': TypeofContact_enc,
    'Occupation': Occupation_enc,
    'Gender': Gender_enc,
    'MaritalStatus': MaritalStatus_enc,
    'Designation': Designation_enc,
    'ProductPitched': ProductPitched_enc,
    'Passport': Passport,
    'OwnCar': OwnCar,
}])

# Predict on button click
if st.button('Predict'):
    pred_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(pred_proba >= 0.45)
    if prediction == 1:
        st.success('The customer is LIKELY to purchase the Wellness Tourism Package.')
    else:
        st.error('The customer is UNLIKELY to purchase the Wellness Tourism Package.')
    st.info(f'Purchase Probability: {pred_proba:.2%}')
