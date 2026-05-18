from IPython.display import display, HTML
from ipywidgets import widgets
from ifri_mini_ml_lib.preprocessing.preparation.encoding import OrdinalEncoder
import numpy as np
import pandas as pd

explicit_categories = [
    ["Junior", "Mid-Level", "Senior", "Director"],
    ["High School", "Bachelor", "Master", "PhD"],
    ["Startup", "SME", "Enterprise"]
]


seniority_widget = widgets.Dropdown(
    options=["Junior", "Mid-Level", "Senior", "Director", "Intern (UNKNOWN)"],
    value="Junior",
    description="Seniority:"
)

education_widget = widgets.Dropdown(
    options=["High School", "Bachelor", "Master", "PhD", "PostDoc (UNKNOWN)"],
    value="Bachelor",
    description="Education:"
)

company_widget = widgets.Dropdown(
    options=["Startup", "SME", "Enterprise", "NGO (UNKNOWN)"],
    value="Startup",
    description="Company Size:"
)

strategy_widget = widgets.ToggleButtons(
    options=['use_encoded_value', 'error'],
    value='use_encoded_value',
    description='Unknown Mode:',
    button_style='info'
)

fill_value_widget = widgets.IntSlider(
    value=-1,
    min=-5,
    max=99,
    step=1,
    description='Fallback Code:'
)

def run_interactive_pipeline(seniority, education, company, strategy, fill_value):
    print("=" * 70)
    print(" LIVE PREPROCESSING PIPELINE INFERENCE ENGINE")
    print("=" * 70)
    

    clean_s = seniority.split(" ")[0]
    clean_e = education.split(" ")[0]
    clean_c = company.split(" ")[0]
    
    X_input = np.array([[clean_s, clean_e, clean_c]])
    

    input_df = pd.DataFrame(X_input, columns=["Seniority", "Education", "Company Size"])
    print("\n[STEP 1] Raw Streaming Feature Input Vector:")
    display(input_df)
    

    interactive_encoder = OrdinalEncoder(
        categories=explicit_categories,
        handle_unknown=strategy,
        unknown_value=fill_value
    )
    

    dummy_train = np.array([["Junior", "High School", "Startup"]])
    interactive_encoder.fit(dummy_train)
    

    try:

        X_encoded = interactive_encoder.transform(X_input)
        encoded_df = pd.DataFrame(X_encoded, columns=["Encoded_Col_0", "Encoded_Col_1", "Encoded_Col_2"])
        
        print("\n[STEP 2] Forward Engineering Mapping Result (.transform()):")
        display(encoded_df)
        

        X_inverted = interactive_encoder.inverse_transform(X_encoded)
        inverted_df = pd.DataFrame(X_inverted, columns=["Recovered_Col_0", "Recovered_Col_1", "Recovered_Col_2"])
        
        print("\n[STEP 3] Structural Pipeline Decoding Output (.inverse_transform()):")
        display(inverted_df)
        
    except ValueError as schema_exception:

        print("\n🛑 PIPELINE EXECUTION CRITICAL CRASH DETECTED!")
        print(f"Error Diagnostic Signature: {schema_exception}")
        print("\nAction Required: Update training definitions or change Unknown Mode to 'use_encoded_value'.")
