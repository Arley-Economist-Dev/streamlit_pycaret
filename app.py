import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

st.set_page_config(page_title="Insurance Charges Prediction App")

@st.cache_data #(allow_output_mutation=True)
def get_model():
    return load_model('my_first_pipeline')

def predict(model, df):
    predictions = predict_model(model, data=df)
    return predictions['prediction_label'][0]

model = get_model()


st.title("Insurance Charges Prediction App")
st.markdown("Enter your personal details to get  aprediction of your insurance\
    charges. This is a simple app showcasing the abilities of the PyCaret\
    regression module, based on Streamlit.")

form = st.form("charges")
age = form.number_input('Age', min_value=1, max_value=100, value=25)
sex = form.radio('Sex', ['Male',"Female"])
bmi = form.number_input('BMI', min_value=10.0,max_value=50.0, value=20.0)
children = form.slider('Children', min_value=0, max_value=10, value=0)
region_list=['Southwest','Northwest','Northeast','Southeast']
region = form.selectbox('Region', region_list)
if form.checkbox('smoker'):
    smoker = 'yes'
else:
    smoker = 'no'
    
predict_button = form.form_submit_button('Predict')

input_dict = {'age':age,'sex':sex.lower(),'bmi':bmi,'children':children,
              'smoker':smoker,'region':region.lower()}

input_df = pd.DataFrame([input_dict])

if predict_button:
    out = predict(model, input_df)
    st.success('The predicted charges are ${:.2f}'.format(out))

# import pandas as pd
# import streamlit as st
# from pycaret.regression import load_model, predict_model

# st.set_page_config(page_title="Insurance Charges Prediction App")

# def get_model():
#     return load_model('my_first_pipeline')

# def predict(model, df):
#     predictions = predict_model(model, data=df)
#     return predictions['prediction_label'][0]

# model = get_model()

# st.title("Aplicación de predicción de gastos de seguros")
# st.markdown("Introduzca sus datos personales para obtener una predicción de los gastos de su seguro.\
#     Esta es una sencilla aplicación que muestra las capacidades del módulo de regresión PyCaret basado en Streamlit.")

# form = st.form("charges")
# age = form.number_input('Edad', min_value=1, max_value=100, value=25)
# sex = form.radio('Sexo', ['Male',"Female"])
# bmi = form.number_input('BMI', min_value=10.0,max_value=50.0, value=20.0)
# children = form.slider('Cantidad de hijos', min_value=0, max_value=10, value=0)
# region_list=['Southwest','Northwest','Northeast','Southeast']
# region = form.selectbox('Region', region_list)
# if form.checkbox('Fumador'):
#     smoker = 'Si'
# else:
#     smoker = 'No'

# predict_button = form.form_submit_button('Predict')

# input_dict = {'age':age,'sex':sex.lower(),'bmi':bmi,'children':children,
#               'smoker':smoker,'region':region.lower()}

# input_df = pd.DataFrame([input_dict])

# if predict_button:
#     # Use st.cache_data to cache the data, not the model
#     @st.cache_data(hash_funcs={input_df: lambda df: df.to_json()})
#     def predict_with_data(model, df):
#         return predict(model, df)

#     out = predict_with_data(model, input_df)
#     st.success('The predicted charges are ${:.2f}'.format(out))
