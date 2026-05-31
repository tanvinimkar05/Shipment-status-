import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('shipment_model.pkl')

st.set_page_config(
    page_title='Shipment Status Prediction',
    page_icon='📦',
    layout='centered'
)

st.title('📦 Shipment Status Prediction')
st.write('Predict whether a shipment will be On Time, Delayed, Returned, or Cancelled.')

st.subheader('Enter Shipment Details')

# Note: 'Distance' was identified as not being used by the model
# during the model training phase. The previous context also mentions
# that 'Shipping_Mode' was removed due to non-existence in the dataset
# for initial model training, but it appears here again. For consistency,
# I will include `Distance` and `Shipping_Mode` as inputs if the model
# was retrained with them. However, based on the `X` variable state,
# the model was trained only on `product_quantity` and `Product_Category`.
# I will adjust the app to reflect the model's actual inputs.

product_quantity = st.number_input(
    'Product Quantity',
    min_value=0,
    value=1
)

product_category_options = ['Electronics', 'Clothing', 'Furniture', 'Books', 'Home Appliances', 'Food']
product_category_map = {
    'Electronics': 0,
    'Clothing': 1,
    'Furniture': 2,
    'Books': 3,
    'Home Appliances': 4,
    'Food': 5 # Adding more categories if not specified in problem statement
}

product_category_input = st.selectbox(
    'Product Category',
    options=product_category_options
)

# Convert selected product category to its numerical representation
product_category_encoded = product_category_map.get(product_category_input, -1) # -1 for unknown/new category

if st.button('Predict Shipment Status'):
    if product_category_encoded == -1:
        st.error("Please select a valid Product Category.")
    else:
        input_data = pd.DataFrame({
            'product_quantity': [product_quantity],
            'Product_Category': [product_category_encoded]
        })

        prediction = model.predict(input_data)

        status_map = {
            0: 'On Time',
            1: 'Delayed',
            2: 'Returned',
            3: 'Cancelled' # Assuming these are the potential output classes
        }

        result = status_map.get(prediction[0], 'Unknown')

        st.success(f'Predicted Shipment Status: {result}')

st.markdown('---')
st.write('Machine Learning Internship Project')
