import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Konfiguracija strani in Session State (za "Kaj-če" tabelo)
st.set_page_config(page_title="Lending Risk Evaluator", page_icon="🏦", layout="wide")
st.title("Napovedovanje tveganja posojil: Odvisni vs. Neodvisni sistem")

if 'history' not in st.session_state:
    st.session_state.history = []

# 2. Nalaganje podatkov in modelov v predpomnilnik (cache), da aplikacija deluje hitro
@st.cache_resource
def load_assets():
    # Modeli
    dep_model = joblib.load('models/xgb_model.pkl')
    indep_model = joblib.load('models/xgb_independent_model.pkl')
    
    # Seznami stolpcev
    dep_features = joblib.load('models/dependent_features.pkl')
    indep_features = joblib.load('models/independent_features.pkl')
    
    # Podatki za simulacijo strank
    test_df = pd.read_csv('data/test_final.csv')
    
    # SHAP Explainerji
    expl_dep = shap.TreeExplainer(dep_model)
    expl_indep = shap.TreeExplainer(indep_model)
    
    # Razveljavi standardizacijo
    scaler = joblib.load('models/scaler.pkl')
    
    return dep_model, indep_model, dep_features, indep_features, test_df, expl_dep, expl_indep, scaler

# Naloži vse
dep_model, indep_model, dep_features, indep_features, test_df, expl_dep, expl_indep, scaler = load_assets()

# Pomožni funkciji za pretvorbo števil iz z-scores v dejanske vrednosti in obratno
def unscale(val, feat):
    if feat in scaler.feature_names_in_:
        idx = list(scaler.feature_names_in_).index(feat)
        return float(val * scaler.scale_[idx] + scaler.mean_[idx])
    return val

def scale(val, feat):
    if feat in scaler.feature_names_in_:
        idx = list(scaler.feature_names_in_).index(feat)
        return float((val - scaler.mean_[idx]) / scaler.scale_[idx])
    return val

# Izvleci pravo tabelo X
target_col = [col for col in test_df.columns if 'loan_status' in col.lower()][0]
test_X = test_df.drop(columns=[target_col])
# Ker manjka risk_cluster, zagotovimo da obstaja, če obstaja v dep_features
from sklearn.cluster import KMeans
if 'risk_cluster' in dep_features and 'risk_cluster' not in test_X.columns:
    # Quick fix: vstavimo osnovni cluster (ali ga potegnemo iz druge datoteke) da rešimo SHAP
    test_X['risk_cluster'] = 0 

# 3. STRANSKA VRSTICA (Sidebar) za vnos uporabnika
st.sidebar.header("Izbira in prilagoditev stranke")

# Izbira bazne stranke iz testne množice
st.sidebar.markdown("Ker imamo 68 atributov, najprej naložimo naključen profil, ki ga lahko obkrojimo:")
sample_idx = st.sidebar.number_input("Izberi ID stranke (0 - 3999):", min_value=0, max_value=len(test_X)-1, value=42)

base_client = test_X.iloc[sample_idx].copy()
resnicno_stanje = test_df.iloc[sample_idx][target_col]

st.sidebar.markdown("---")
st.sidebar.subheader("🏦 Nastavitve banke:")
THRESHOLD = st.sidebar.slider("Strogost - Prag tveganja (Threshold)", 0.10, 0.90, 0.40, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Dinamično spreminjanje stranke:")

# Preberemo izhodiščne SKALIRANE vrednosti in jih pretvorimo za UI:
base_dti = unscale(base_client.get('dti', 0), 'dti')
base_inc = unscale(base_client.get('annual_inc', 0), 'annual_inc')
base_fico = unscale(base_client.get('fico_avg', 0), 'fico_avg')
base_hist = unscale(base_client.get('credit_history_years', 0), 'credit_history_years')
base_revol = unscale(base_client.get('revol_bal', 0), 'revol_bal')
base_loan = unscale(base_client.get('loan_amnt', 0), 'loan_amnt')
base_term = unscale(base_client.get('term', 0), 'term')
base_emp = unscale(base_client.get('emp_length', 0), 'emp_length')
base_int = unscale(base_client.get('int_rate', 0), 'int_rate') # Potrebujemo za izračun obroka

# Interaktivni sliderji za ključne vrednosti z REALNIMI enotami ($ in %)
novo_posojilo = st.sidebar.number_input("Znesek posojila ($)", 500.0, 50000.0, value=max(500.0, float(base_loan)), step=1000.0)
noba_doba = st.sidebar.slider("Doba odplačevanja (meseci)", 36, 60, int(max(36, min(60, float(base_term)))), step=12)
nov_dohodek = st.sidebar.number_input("Letni dohodek ($)", min_value=1000.0, max_value=1000000.0, value=max(1000.0, float(base_inc)), step=5000.0)
novo_dti = st.sidebar.slider("DTI (Dolg proti prihodkom v %)", 0.0, 60.0, max(0.0, min(60.0, float(base_dti))), step=0.5)
delovna_doba = st.sidebar.slider("Status zaposlitve (leta)", 0, 10, int(max(0, min(10, float(base_emp)))), step=1)

fico_val = base_fico
if 'fico_avg' in base_client:
    fico_val = st.sidebar.slider("FICO Ocena (Uporabi SAMO odvisni model!)", 300.0, 850.0, max(300.0, min(850.0, float(base_fico))), step=5.0)

zgodovina_kredita = st.sidebar.slider("Kreditna zgodovina (leta)", 0.0, 40.0, max(0.0, min(40.0, float(base_hist))), step=1.0)
stanje_kredita = st.sidebar.number_input("Stanje na obrokih - Revolving ($)", 0.0, value=max(0.0, float(base_revol)), step=1000.0)
sentiment = st.sidebar.slider("Sentiment NLP opisa (-1 negativno, 1 pozitivno)", -1.0, 1.0, float(base_client.get('desc_sentiment_score', 0.0)), step=0.1)

# Dinamični izračun novega mesečnega obroka (installment) ob spremembi posojila/dobe
r = base_int / 1200.0
if r > 0:
    nova_obveznost = novo_posojilo * (r * (1 + r)**noba_doba) / ((1 + r)**noba_doba - 1)
else:
    nova_obveznost = novo_posojilo / noba_doba

st.sidebar.markdown(f"**ℹ️ Preračunan mesečni obrok:** ${nova_obveznost:.2f}")

# Posodobitev izbranega vektorja stranke z vrnitvijo SKALIRANIH vrednosti nazaj v DataFrame
client_custom = base_client.copy()
client_custom['loan_amnt'] = scale(novo_posojilo, 'loan_amnt')
client_custom['term'] = scale(noba_doba, 'term')
client_custom['installment'] = scale(nova_obveznost, 'installment')
client_custom['annual_inc'] = scale(nov_dohodek, 'annual_inc')
client_custom['dti'] = scale(novo_dti, 'dti')
client_custom['emp_length'] = scale(delovna_doba, 'emp_length')
client_custom['credit_history_years'] = scale(zgodovina_kredita, 'credit_history_years')
client_custom['revol_bal'] = scale(stanje_kredita, 'revol_bal')
client_custom['desc_sentiment_score'] = sentiment

if 'fico_avg' in client_custom:
    client_custom['fico_avg'] = scale(fico_val, 'fico_avg')

# Izdelava DataFrame za oba modela s strogim upoštevanjem imen stolpcev, ki jih prihajajo direktno iz učenja
client_dep_df = pd.DataFrame([client_custom]).reindex(columns=dep_model.feature_names_in_, fill_value=0)
client_indep_df = pd.DataFrame([client_custom]).reindex(columns=indep_model.feature_names_in_, fill_value=0)

# Resnično stanje za primerjavo
st.markdown(f"**Dejansko stanje te stranke v testni bazi:** {'🔴 Neplačnik (1)' if resnicno_stanje == 1 else '🟢 Odplačal kredit (0)'}")

# Prikaz ključnih NLP besed, ki opisujejo izbrano stranko
nlp_cols = [c for c in base_client.index if 'tfidf_' in c and base_client[c] > 0]
if len(nlp_cols) > 0:
    cleaned_words = [w.replace('tfidf_emp_', '').replace('tfidf_desc_', '').replace('tfidf_', '') for w in nlp_cols]
    st.info(f"📝 **NLP profil te stranke (izvlečen iz test_final.csv):** {', '.join(cleaned_words)}")

st.markdown("---")

# 4. GLAVNI DEL (Razcep v dva stolpca)
stolpci = st.columns(2)
col1 = stolpci[0]
col2 = stolpci[1]

with col1:
    st.header("🏢 ODVISNI MODEL")
    st.caption("Uporablja tudi FICO, Grade, Int Rate")
    
    prob_dep = dep_model.predict_proba(client_dep_df)[0][1]
    st.progress(float(prob_dep), text=f"Tveganje: {prob_dep*100:.1f}%")
    
    if prob_dep >= THRESHOLD:
        st.error(f"ZAVRNJENO! Verjetnost neplačila je nad mejo ({THRESHOLD*100:.0f}%)")
    else:
        st.success(f"ODOBRENO! Verjetnost neplačila je varna.")
        
    st.markdown("**SHAP Waterfall graf (kako so atributi dodajali tveganje):**")
    shap_val_dep = expl_dep(client_dep_df)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    shap.waterfall_plot(shap_val_dep[0], show=False)
    st.pyplot(fig)
    plt.clf()

with col2:
    st.header("⚖️ NEODVISNI MODEL")
    st.caption("Slepi test: Brez FICO, Grade in Obresti")
    
    prob_indep = indep_model.predict_proba(client_indep_df)[0][1]
    st.progress(float(prob_indep), text=f"Tveganje: {prob_indep*100:.1f}%")
    
    if prob_indep >= THRESHOLD:
        st.error(f"ZAVRNJENO! Verjetnost neplačila je nad mejo ({THRESHOLD*100:.0f}%)")
    else:
        st.success(f"ODOBRENO! Verjetnost neplačila je varna.")
        
    st.markdown("**SHAP Waterfall graf (kako so atributi dodajali tveganje):**")
    shap_val_indep = expl_indep(client_indep_df)
    
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    shap.waterfall_plot(shap_val_indep[0], show=False)
    st.pyplot(fig2)
    plt.clf()

# 5. Zgodovina ("What-If" Analysis)
st.markdown("---")
st.subheader("📚 Tabela simulacij (What-If analiza)")
if st.button("💾 Shrani trenutno oceno v tabelo"):
    st.session_state.history.append({
        "Stranka ID": sample_idx,
        "Prag": f"{THRESHOLD*100}%",
        "Znesek ($)": round(novo_posojilo, 0),
        "Dohodek ($)": round(nov_dohodek, 0),
        "DTI (%)": round(novo_dti, 1),
        "FICO": int(fico_val),
        "Tveganje - Odvisni": f"{prob_dep*100:.1f}%",
        "Tveganje - Neodvisni": f"{prob_indep*100:.1f}%"
    })

if len(st.session_state.history) > 0:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    if st.button("🗑️ Počisti zgodovino"):
        st.session_state.history = []
        st.rerun()
