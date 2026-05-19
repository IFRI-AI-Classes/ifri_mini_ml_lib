from IPython.display import display, HTML
from ipywidgets import widgets
from ifri_mini_ml_lib.preprocessing.preparation.encoding import OrdinalEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ifri_mini_ml_lib.preprocessing.preparation.splitting import DataSplitter


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


def demo_splitting(method, test_size, k_folds, seed):
    # Interactively visualizes the result of each splitting method.
    np.random.seed(0)
    n_demo = 100
    X_d = pd.DataFrame({'x': np.random.randn(n_demo),
                         'y': np.random.randn(n_demo)})
    y_d = pd.Series(np.where(np.random.rand(n_demo) < 0.25, 1, 0))
    dates_d = pd.date_range('2023-01-01', periods=n_demo, freq='D')
    X_d_t = X_d.copy(); X_d_t.index = dates_d

    sp = DataSplitter(seed=seed)
    fig, ax = plt.subplots(figsize=(11, 3))

    if method == 'train_test_split':
        Xtr, Xte, ytr, yte = sp.train_test_split(X_d, y_d, test_size=test_size)
        sizes  = [len(Xtr), len(Xte)]
        labels = [f'Train\n{len(Xtr)} ({len(Xtr)/n_demo:.0%})',
                  f'Test\n{len(Xte)} ({len(Xte)/n_demo:.0%})']
        colors = ['#4CAF50', '#F44336']
        left = 0
        for sz, lbl, col in zip(sizes, labels, colors):
            ax.barh(0, sz, left=left, height=0.5, color=col)
            ax.text(left + sz/2, 0, lbl, ha='center', va='center',
                    fontweight='bold', color='white', fontsize=11)
            left += sz
        ax.set_title('train_test_split — Data distribution', fontweight='bold', fontsize=13)

    elif method == 'stratified_train_test_split':
        Xtr, Xte, ytr, yte = sp.stratified_train_test_split(X_d, y_d, test_size=test_size)
        pct_orig = (y_d == 1).mean() * 100
        pct_tr   = (ytr  == 1).mean() * 100
        pct_te   = (yte  == 1).mean() * 100
        cats = ['Original dataset', 'Train', 'Test']
        p0   = [100 - pct_orig, 100 - pct_tr, 100 - pct_te]
        p1   = [pct_orig, pct_tr, pct_te]
        x_pos = np.arange(3)
        ax.bar(x_pos, p0, color='#4CAF50', label='Class 0')
        ax.bar(x_pos, p1, bottom=p0, color='#F44336', label='Class 1')
        for xi, v0, v1 in zip(x_pos, p0, p1):
            ax.text(xi, v0/2,       f'{v0:.1f}%', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)
            ax.text(xi, v0 + v1/2,  f'{v1:.1f}%', ha='center', va='center',
                    color='white', fontweight='bold', fontsize=10)
        ax.set_xticks(x_pos); ax.set_xticklabels(cats)
        ax.set_ylabel('Proportion (%)'); ax.legend()
        ax.set_title('stratified_train_test_split — Preserved proportions', fontweight='bold', fontsize=13)

    elif method == 'temporal_train_test_split':
        Xtr, Xte, ytr, yte = sp.temporal_train_test_split(X_d_t, y_d, test_size=test_size)
        y_plot = pd.Series(np.cumsum(np.random.randn(n_demo)), index=dates_d)
        ax.plot(y_plot[Xtr.index], color='#4CAF50', linewidth=2,
                label=f'Train ({len(Xtr)} pts)')
        ax.plot(y_plot[Xte.index], color='#F44336', linewidth=2,
                label=f'Test ({len(Xte)} pts)')
        ax.axvline(x=Xte.index[0], color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Date')
        ax.legend()
        ax.set_title('temporal_train_test_split — Chronological split', fontweight='bold', fontsize=13)

    elif method == 'k_fold_split':
        folds = sp.k_fold_split(X_d, y_d, k=k_folds)
        fold_size = n_demo // k_folds
        for i in range(k_folds):
            for j in range(k_folds):
                color = '#F44336' if j == i else '#4CAF50'
                ax.barh(i, fold_size, left=j*fold_size, height=0.6, color=color, edgecolor='white')
                ax.text(j*fold_size + fold_size/2, i,
                        'TEST' if j == i else 'train',
                        ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white')
        ax.set_yticks(range(k_folds))
        ax.set_yticklabels([f'Cycle {i+1}' for i in range(k_folds)])
        ax.set_title(f'k_fold_split — Test fold rotation (k={k_folds})', fontweight='bold', fontsize=13)

    ax.set_xlim(0, n_demo if method != 'stratified_train_test_split' else None)
    if method not in ('stratified_train_test_split', 'k_fold_split', 'temporal_train_test_split'):
        ax.axis('off')
    plt.tight_layout()
    plt.show()
