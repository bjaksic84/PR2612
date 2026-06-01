import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Konfiguracija strani
st.set_page_config(page_title="Lending Risk Evaluator", page_icon="🏦", layout="wide")
st.title("Napovedovanje tveganja posojil: Odvisni vs. Neodvisni sistem")

# 2. Nalaganje podatkov in modelov
@st.cache_resource
def load_assets():
    dep_model = joblib.load('models/xgb_model.pkl')
    indep_model = joblib.load('models/xgb_independent_model.pkl')
    dep_features = joblib.load('models/dependent_features.pkl')
    indep_features = joblib.load('models/independent_features.pkl')
    test_df = pd.read_csv('data/test_final.csv')
    return dep_model, indep_model, dep_features, indep_features, test_df

dep_model, indep_model, dep_features, indep_features, test_df = load_assets()

# Izvleci pravo tabelo X
target_col = [col for col in test_df.columns if 'loan_status' in col.lower()][0]
test_X = test_df.drop(columns=[target_col])

if 'risk_cluster' in dep_features and 'risk_cluster' not in test_X.columns:
    test_X['risk_cluster'] = 0 

# 3. STRANSKA VRSTICA (Sidebar) za vnos uporabnika
st.sidebar.header("Izbira stranke")
sample_idx = st.sidebar.number_input("Izberi ID stranke (0 - 3999):", min_value=0, max_value=len(test_X)-1, value=42)

base_client = test_X.iloc[sample_idx].copy()
resnicno_stanje = test_df.iloc[sample_idx][target_col]

st.sidebar.markdown("---")
THRESHOLD = st.sidebar.slider("Strogost - Prag tveganja (Threshold)", 0.10, 0.90, 0.40, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Dinamično spreminjanje stranke (Z-scores):")

novo_posojilo = st.sidebar.slider("Znesek posojila (Z-score)", -3.0, 5.0, value=float(base_client.get('loan_amnt', 0.0)), step=0.1)
noba_doba = st.sidebar.slider("Doba odplačevanja (Z-score)", -3.0, 5.0, value=float(base_client.get('term', 0.0)), step=0.1)
nov_dohodek = st.sidebar.slider("Letni dohodek (Z-score)", -3.0, 5.0, value=float(base_client.get('annual_inc', 0.0)), step=0.1)
novo_dti = st.sidebar.slider("DTI (Z-score)", -3.0, 5.0, value=float(base_client.get('dti', 0.0)), step=0.1)
delovna_doba = st.sidebar.slider("Status zaposlitve (Z-score)", -3.0, 5.0, value=float(base_client.get('emp_length', 0.0)), step=0.1)

fico_val = float(base_client.get('fico_avg', 0.0))
if 'fico_avg' in base_client:
    fico_val = st.sidebar.slider("FICO Ocena (Z-score)", -3.0, 5.0, value=float(base_client.get('fico_avg', 0.0)), step=0.1)

client_custom = base_client.copy()
client_custom['loan_amnt'] = novo_posojilo
client_custom['term'] = noba_doba
client_custom['annual_inc'] = nov_dohodek
client_custom['dti'] = novo_dti
client_custom['emp_length'] = delovna_doba

if 'fico_avg' in client_custom:
    client_custom['fico_avg'] = fico_val

# Izdelava DataFrame
client_dep_df = pd.DataFrame([client_custom]).reindex(columns=dep_model.feature_names_in_, fill_value=0)
client_indep_df = pd.DataFrame([client_custom]).reindex(columns=indep_model.feature_names_in_, fill_value=0)

st.markdown(f"**Dejansko stanje te stranke v testni bazi:** {'🔴 Neplačnik (1)' if resnicno_stanje == 1 else '🟢 Odplačal kredit (0)'}")
st.markdown("---")

# 4. GLAVNI DEL
stolpci = st.columns(2)
with stolpci[0]:
    st.header("🏢 ODVISNI MODEL")
    st.caption("Uporablja tudi FICO, Grade, Int Rate")
    
    prob_dep = dep_model.predict_proba(client_dep_df)[0][1]
    st.progress(float(prob_dep), text=f"Tveganje: {prob_dep*100:.1f}%")
    
    if prob_dep >= THRESHOLD:
        st.error(f"ZAVRNJENO! Verjetnost neplačila je {prob_dep*100:.0f}%")
    else:
        st.success(f"ODOBRENO! Stranka je varna.")

with stolpci[1]:
    st.header("⚖️ NEODVISNI MODEL")
    st.caption("Slepi test: Brez FICO, Grade in Obresti")
    
    prob_indep = indep_model.predict_proba(client_indep_df)[0][1]
    st.progress(float(prob_indep), text=f"Tveganje: {prob_indep*100:.1f}%")
    
    if prob_indep >= THRESHOLD:
        st.error(f"ZAVRNJENO! Verjetnost neplačila je {prob_indep*100:.0f}%")
    else:
        st.success(f"ODOBRENO! Stranka je varna.")
