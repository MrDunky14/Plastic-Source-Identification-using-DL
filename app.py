import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
# Remove these imports if no longer directly used for fitting:
# from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
import os
import joblib # Import joblib for loading

# --- THIS LINE MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Deep Earth Sentinel: Microplastic Source Predictor", layout="wide")

# --- Configuration for Streamlit App ---
MODEL_PATH = 'best_deep_earth_sentinel_model.keras'
PREPROCESSOR_PATH = 'fitted_preprocessor.joblib' # New path for preprocessor
LABEL_ENCODER_PATH = 'fitted_label_encoder.joblib' # New path for label encoder

# Define column names used for X during training, in their original order (before preprocessing)
FEATURE_COLUMNS_ORIGINAL = [
    'product_label',
    'product_size',
    'brand_name',
    'manufacturer_country',
    'manufacturer_name',
    'bottle_count'
]

# --- Load the Trained Model, Preprocessor, and LabelEncoder ---
@st.cache_resource # Cache the loading of these resources
def load_all_models_and_preprocessors():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return model, preprocessor, label_encoder
    except Exception as e:
        st.error(f"Error loading required files: {e}")
        st.stop()

model, preprocessor, label_encoder = load_all_models_and_preprocessors()

# --- Recreate unique categories for selectboxes directly from label_encoder ---
# This ensures selectbox options match what the preprocessor and model expect
# We need to get the unique categories that the preprocessor was fitted on.
# Iterating through the transformers of the preprocessor:
# This part is a bit tricky, if original_df is still needed for selectbox options,
# we need to ensure its column types and contents match the training data.
# For now, let's assume `original_df` (if loaded) is consistent.
# The `astype(str)` fix should handle floats/strings for selectbox population.
try:
    # Assuming 'merged_plastic_bottle_waste.csv' is still used for selectbox options
    full_df_path = os.path.join(os.path.dirname(__file__), 'merged_plastic_bottle_waste.csv')
    if not os.path.exists(full_df_path):
        st.error(f"Error: Could not find '{full_df_path}'. Please ensure merged_plastic_bottle_waste.csv is in the same directory as app.py")
        st.stop()
    original_df = pd.read_csv(full_df_path)
    # Ensure correct column types in original_df if not already done
    # You might want to apply type casting here if your original raw data has inconsistencies
    # E.g., original_df['some_col'] = original_df['some_col'].astype(str)

except Exception as e:
    st.error(f"Error loading original_df for selectbox options: {e}")
    st.info("You might need to adjust the path or ensure the file exists.")
    st.stop()


# --- Streamlit App Interface ---
st.title("🌎 Deep Earth Sentinel: Microplastic Scan Country Predictor")
st.markdown("## Identify the likely scan country of plastic bottle waste based on its attributes.")
st.write("---")

st.sidebar.header("Input Bottle Attributes")


# Create input widgets for each feature
input_data = {}
for col in FEATURE_COLUMNS_ORIGINAL:
    # Determine if the feature is categorical by checking preprocessor's transformers
    # This is a more robust way to ensure consistency with what preprocessor expects
    is_categorical_feature = False
    for transformer_name, transformer, cols_affected in preprocessor.transformers_:
        if transformer_name == 'cat' and col in cols_affected.tolist(): # Check if col is in the list of categorical columns for OneHotEncoder
            is_categorical_feature = True
            break

    if is_categorical_feature:
        # Get unique categories from the preprocessor's fitted OneHotEncoder
        # This is the most crucial part to ensure consistency
        try:
            # Find the index of the column in the original categorical features list
            col_index_in_ohe = original_df[preprocessor.named_transformers_['cat'].feature_names_in_.tolist()].columns.get_loc(col)
            unique_categories = sorted(preprocessor.named_transformers_['cat'].categories_[col_index_in_ohe].tolist())
        except Exception as e:
            st.warning(f"Could not retrieve categories for {col} from preprocessor, falling back to original_df: {e}")
            # Fallback to previous method if preprocessor categories are hard to access directly
            # Make sure this part continues to use .astype(str) to avoid sorting errors
            unique_categories = sorted(original_df[col].astype(str).unique().tolist())


        input_data[col] = st.sidebar.selectbox(f"Select {col.replace('_', ' ').title()}:", unique_categories)
    elif col == 'bottle_count': # Specific check for numerical bottle_count
        input_data[col] = st.sidebar.number_input(f"Enter {col.replace('_', ' ').title()}:", min_value=1, value=1, step=1)
    else: # Default for other numerical features if any
        input_data[col] = st.sidebar.text_input(f"Enter {col.replace('_', ' ').title()}:")


# Convert input_data to a DataFrame for preprocessing
input_df = pd.DataFrame([input_data])

# --- Prediction Logic ---
if st.sidebar.button("Predict Scan Country"):
    try:
        # Preprocess the input data
        # Ensure column order matches training data
        input_processed = preprocessor.transform(input_df[FEATURE_COLUMNS_ORIGINAL])

        # Make prediction
        prediction_probs = model.predict(input_processed)[0]
        predicted_class_index = np.argmax(prediction_probs)
        predicted_scan_country = label_encoder.inverse_transform([predicted_class_index])[0]

        st.success(f"## Predicted Scan Country: **{predicted_scan_country}**")

        st.subheader("Prediction Probabilities:")
        # Display probabilities for top N classes for better insight
        top_n = 5
        top_indices = prediction_probs.argsort()[-top_n:][::-1]
        top_probs = prediction_probs[top_indices]
        top_countries = label_encoder.inverse_transform(top_indices)

        prob_df = pd.DataFrame({
            "Country": top_countries,
            "Probability": [f"{p:.2%}" for p in top_probs]
        })
        st.table(prob_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure all input fields are correctly filled and 'merged_plastic_bottle_waste.csv' is accessible.")

st.write("---")
st.markdown("Model trained as part of the 'Deep Earth Sentinel: AI-Powered Microplastic Pollution Source Identification and Forecasting' project.")
st.markdown("Developed by Krishna")