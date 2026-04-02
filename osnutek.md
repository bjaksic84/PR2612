# Napovedovanje neplačila posojil s strojnim učenjem

**Projektna naloga pri predmetu Podatkovno rudarjenje (2025/26)**

**Avtorji**: Bojan Jakšić, Tilen Butara, Martin Lazar

---

## Opis problema

Pri odobravanju potrošniških posojil je ključno vprašanje, ali bo posojilojemalec posojilo v celoti poplačal ali pa bo prišlo do neplačila (*default*). Napačna ocena tveganja vodi v finančne izgube za posojilodajalca. Cilj projekta je s pomočjo metod podatkovnega rudarjenja in strojnega učenja zgraditi napovedni model, ki na podlagi lastnosti posojilojemalca in posojila oceni verjetnost neplačila.

## Cilji in raziskovalna vprašanja

1. **Kateri dejavniki so najmočneje povezani z neplačilom posojila?** (npr. obrestna mera, kreditna ocena, razmerje dolga do dohodka)
2. **Ali je mogoče iz opisnih podatkov (besedilo, naziv zaposlitve) pridobiti dodaten signal za oceno tveganja?**
3. **Kateri napovedni model daje najboljše rezultate?** Primerjava logistične regresije, naključnih gozdov in algoritma XGBoost.
4. **Ali se posojilojemalce da smiselno razvrstiti v skupine tveganja** z uporabo metode K-means?
5. **Kako razložljive so napovedi modela?** Uporaba metode SHAP za razlago posameznih napovedi.

## Vir in oblika podatkov

Uporabljamo prosto dostopen (*open data*) nabor podatkov **Lending Club Loan Data**, objavljen na platformi Kaggle:

- **Vir**: [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Izvor podatkov**: Lending Club je ameriška platforma za medsebojno posojanje (*peer-to-peer lending*), ki je podatke o posojilih javno objavljala.
- **Oblika**: ena datoteka CSV (~1,2 GB), ki vsebuje podatke o odobrenih posojilih med letoma 2007 in 2018.
- **Obseg**: ~2,2 mio. vrstic in ~150 stolpcev. Za potrebe projekta bomo podatke filtrirali in vzorčili na **20.000 vrstic** ter obdržali **~40 izbranih stolpcev**.
- **Ciljna spremenljivka**: `loan_status` — ohranimo le posojila s statusom *Fully Paid* (poplačano → 0) ali *Charged Off* (neplačilo → 1).

Med izbranimi značilkami so:
- **finančne lastnosti posojila**: znesek, obrestna mera, obrok, rok odplačevanja,
- **lastnosti posojilojemalca**: letni dohodek, razmerje dolga do dohodka (DTI), kreditna ocena (FICO), lastništvo nepremičnine, namen posojila, delovna doba,
- **kreditna zgodovina**: število odprtih računov, stanje revolving dolga, število zamud pri plačilih,
- **besedilni podatki**: opis posojila in naziv zaposlitve.

## Načrt izvedbe

| Faza | Vsebina |
|------|---------|
| 0 | Nalaganje podatkov, filtriranje, vzorčenje, razdelitev na učno in testno množico |
| 1 | Raziskovalna analiza (EDA), obdelava manjkajočih vrednosti, inženiring značilk, kodiranje, skaliranje |
| 2 | Iskanje asociacijskih pravil (Apriori) in izbira značilk (sekvenčna selekcija) |
| 3 | NLP — analiza besedilnih polj s TF-IDF in demonstracija modela za analizo sentimenta (HuggingFace) |
| 4 | Gručenje (K-means), klasifikacija (logistična regresija, naključni gozdovi, XGBoost), primerjava modelov |
| 5 | Razložljivost modela z metodo SHAP, sinteza ugotovitev |
| 6 | Primerjava in ovrednotenje rezultatov z obstoječo literaturo ter ugotovitvami sorodnih študij |

## Podrobnejši načrt uvodnih faz

Za lažjo delitev dela in takojšen začetek projekta so začetne faze razdelane podrobneje. Tako imamo zagotovljeno enotno izhodišče za vse nadaljnje faze (NLP, Apriori, modeliranje).

### Faza 0: Nalaganje podatkov in vzorčenje (Notebook `00_data_loading.ipynb`)
- **Cilj**: Pripraviti obvladljivo in čisto začetno bazo iz izhodiščne ~1,2 GB datoteke.
- **Koraki**:
  1. **Branje podatkov** iz `data/raw/accepted_2007_to_2018Q4.csv` z uporabo `pandas` parametra `low_memory=False`.
  2. **Filtriranje ciljne spremenljivke (`loan_status`)**: ohranimo samo vrstice *Fully Paid* (kodirano kot 0) in *Charged Off* (kodirano kot 1).
  3. **Izbira atributov**: obdržimo ~40 ključnih stolpcev (npr. znesek, dti, dohodek, fico, emp_title, desc). Stolpec `addr_state` odstranimo (naša odločitev za manj dummy spremenljivk).
  4. **Stratificirano vzorčenje**: izluščimo natanko natanko 20.000 vrstic. Obvezno mora ohranjati originalno razmerje med plačanimi in neplačanimi posojili.
  5. **Razdelitev**: stratificirana delitev na učno (80 %) in testno (20 %) množico.
- **Rezultat**: Datoteke `lending_club_20k.csv`, `train_raw.csv` in `test_raw.csv` shranjene v imeniku `data/`.

### Faza 1: Raziskovalna analiza in predprocesiranje (Notebook `01_eda_preprocessing.ipynb`)
- **Cilj**: Razumeti vzorec in pripraviti numerično bazo brez manjkajočih vrednosti, primerno za nevronske mreže in XGBoost.
- **EDA (Eksploratorna analiza - preverjanje na vzorcu):**
  1. Vizualizacije deleža manjkajočih podatkov in porazdelitev ciljne spremenljivke.
  2. Histokrami ključnih numeričnih značilk (dohodek, dog, FICO...) s prekrivanjem barv glede na `loan_status` (razred).
  3. Korelacijska matrika numeričnih značilk.
  4. Analiza prisotnosti besedilnih preučevanih atributov (`desc`, `emp_title`).
- **Predprocesiranje (Učenje oz. fit zgolj na *učni* množici, transformiranje pa na obeh):**
  1. **Manjkajoči podatki**: Numerične z < 30 % manjkajočih imputiramo z mediano. Časovne atribute (`mths_since_last_delinq`) nadomestimo z mediano IN uvedemo binarno zastavico, če je bil podatek sprva manjkajoč. Besedilom dodelimo *prazno (\"\")* ali *"unknown"*.
  2. **Inženiring značilk**: povprečen FICO rezultat `fico_avg`, dolžina kreditne zgodovine v letih (parsiranje `earliest_cr_line`), posojilo-na-dohodek, obrok-na-dohodek.
  3. **Kodiranje**: razred in podrazred posojila ordinalno, kategorije vrste lastništva in namena v One-Hot.
  4. **Skaliranje**: `StandardScaler` za vse numerične vrstice.
- **Rezultat**: Pripravljeni in skalirani `train_processed.csv` in `test_processed.csv` podatki ter shranjeni scaler objekti v mapo `models/`.

### Faza 2: Iskanje asociacijskih pravil in izbira značilk (Notebook `02_pattern_mining.ipynb`)
- **Cilj**: Odkriti skrite vzorce med dejavniki (Apriori) ter zmanjšati dimenzionalnost prostora (izbor najpomembnejših značilk).
- **Asociacijska pravila (Association Rule Mining):**
  1. **Diskretizacija**: Pretvorba zveznih spremenljivk (dohodek, višina posojila, DTI) v diskretne intervale (npr. 'nizek', 'srednji', 'visok') z uporabo binninga.
  2. **Apriori algoritem**: Uporaba knjižnice `mlxtend` za iskanje pogostih naborov postavk (Frequent Itemsets).
  3. **Izvleček pravil**: Pridobivanje asociacijskih pravil s poudarkom na tistih, ki za posledico (*consequent*) vsebujejo informacijo o neplačilu.
  4. **Analiza:** Interpretacija metrik, kot so *podpora (support)*, *zaupanje (confidence)* in *dvig (lift)*.
- **Izbira značilk (Feature Selection):**
  1. Uporaba modelov (npr. naključni gozd za *feature importances*) na predprocesiranem naboru iz Faze 1.
  2. Odstranitev odvečnih značilk za preprečevanje prevelikega prilagajanja (*overfitting*).
- **Rezultat**: Optimiziran nabor pomembnih stolpcev za končno modeliranje (`train_selected.csv`, `test_selected.csv`).


### Faza 3: NLP in analiza besedilnih polj (Notebook `03_nlp_risk_scoring.ipynb`)
- **Cilj**: Iz besedilnih podatkov (`desc` in `emp_title`) pridobiti dodatne informativne značilke (signale o tveganju) za končno modeliranje.
- **Obdelava s TF-IDF (`emp_title`)**:
  1. Čiščenje besedila (pretvorba v male črke, odstranitev ločil in posebnosti).
  2. Uporaba `TfidfVectorizer` (iz `scikit-learn`) za izluščitev najpomembnejših ključnih besed iz nazivov zaposlitev (npr. top 20-50 besed kot novih stolpcev).
- **Analiza sentimenta s HuggingFace (`desc`)**:
  1. Za daljša besedila (opise razlogov za posojilo) bomo izvedli analizo sentimenta z uporabo enega od vnaprej naučenih modelov (npr. `transformers` knjižnica, pipeline za sentiment).
  2. Priključitev verjetnosti negativnega/pozitivnega sentimenta kot novo značilko (`desc_sentiment_score`).
- **Združevanje**:
  1. Pridobljene numerične in NLP značilke se priključijo k osnovni strukturi podatkov (`train_selected.csv` / `test_selected.csv`).
  2. Prvotni tekstovni besedilni stolpci se odstranijo.
- **Rezultat**: Podatkovni datoteki (`train_final.csv`, `test_final.csv`) z združenimi finančnimi in NLP značilkami, pripravljeni za končno modeliranje.

### Faza 4: Gručenje in klasifikacijsko modeliranje (Notebook `04_modeling.ipynb`)
- **Cilj**: Razviti končne modele za oceno tveganja neplačila in segmentirati stranke v skupine tveganja.
- **Gručenje (K-Means)**:
  1. Uporaba algoritma K-Means za nabor podatkov (`train_final.csv`), iskanje optimalnega števila gruč z metodo komolca (Elbow method).
  2. Profiliranje posameznih gruč in ustvarjanje nove značilke `risk_cluster`.
- **Klasifikacija**:
  1. Začetna priprava osnovnega (baseline) modela: Logistična regresija.
  2. Implementacija naprednejših modelov s prečnim preverjanjem (GridSearch/RandomSearch): Naključni gozd in XGBoost.
  3. Reševanje neuravnoteženosti razredov (angl. *class imbalance*) z uporabo uteži (scale_pos_weight ipd.).
- **Evalvacija**:
  1. Modeli bodo ocenjeni na testni množici (`test_final.csv`).
  2. Prikaz matrike zmede (confusion matrix) in izračun metrik, kot so točnost (Accuracy), natančnost (Precision), priklic (Recall), mera F1 ter AUC-ROC krivulja.
- **Rezultat**: Optimalen model shranjen kot `xgb_model.pkl` za nadaljnje pojasnjevanje ob pomoči SHAP ter vnos v vizualno aplikacijo.

### Faza 5: Razložljivost modela z metodo SHAP (Notebook `05_explainability.ipynb`)
- **Cilj**: Omogočiti popolno transparentnost in razumljivost črne skrinjice znanja XGBoost modela preko globalnih in lokalnih SHAP vplivov.
- **Globalna razložljivost**:
  1. Uporaba `shap.TreeExplainer` na izbranem drevesnem modelu `xgb_model.pkl`.
  2. Izračun in izris zbirnega grafa (`summary_plot`, `beeswarm`), ki nakazuje kateri atributi na ravni celotne populacije najbolj pripomorejo k povišanemu tveganju.
- **Lokalna razložljivost (Individualne posojilne odločitve)**:
  1. Identifikacija posameznika z zelo nizkim tveganjem za neplačilo (dober primer).
  2. Identifikacija posameznika z zelo visokim tveganjem za neplačilo (slab primer).
  3. Izris `waterfall_plot` za ta dva specifična primera z namenom razumevanja, zakaj sta uvrščena v ti kategoriji napovedi.
- **Interaktivna aplikacija (Streamlit)**:
  1. Priprava Python datoteke (`app.py`), ki zažene aplikacijo s pomočjo ogrodja Streamlit.
  2. Aplikacija omogoča vnos poljubnih lastnosti posojilojemalca (ali naključen izbor vzorca iz predpripravljene CSV baze) in interaktivno napove tveganje, hkrati pa vmesniški prikaz izriše ustrezen SHAP graf (`waterfall`).
- **Rezultat**: Shranjene in posredovane zaledne vizualizacije vplivov, ki jih preko front-end aplikacije za končne uporabnike enostavno prevajajo do netehničnih uslužbencev v banki.


### Faza 6: Primerjava in ovrednotenje z literaturo (Notebook `06_literature_comparison.ipynb`)
- **Cilj**: Kritično in znanstveno primerjati uspešnost naše implementacije z obstoječimi akademskimi izsledki in referencami iz domene strojnega učenja na področju tveganj.
- **Aktivnosti**:
  1. Identifikacija in popis 3 do 5 ključnih strokovnih študij (Google Scholar / IEEE), ki so temeljile na podobnih posojilnih podatkih.
  2. Kvantitativna in kvalitativna primerjalna tabela metrik našega modela (ROC-AUC, izračunane F1-mere) proti tistim v literaturi.
  3. Analiza vpliva specifik našega projekta (NLP značilke iz tekstovnih opisov, segmentacija strank) ter ovrednotenje njune doprinesene vrednosti.
- **Rezultat**: Sklepne ugotovitve, omejitve preučevanja in teoretično utemeljen zaključek celotne projektne naloge.

## Pričakovani rezultati in prikaz

Primarni rezultat projekta je napovedni model, ki za posamezno posojilo vrne **verjetnost neplačila** (vrednost med 0 in 1) skupaj z razvrstitvijo posojilojemalca v eno od skupin tveganja (nizko / srednje / visoko). Pričakujemo, da bo model XGBoost dosegel najboljšo ločljivost med razredi (AUC-ROC > 0,75), logistična regresija pa bo služila kot interpretabilna izhodišče za primerjavo.

Poleg numeričnih metrik bomo z metodo SHAP prikazali, kateri dejavniki so pri posamezni napovedi odločilni — kar omogoča razlago napovedi tudi nestrokovnjakom.

Rezultati in model bodo predstavljeni v obliki interaktivne spletne aplikacije, razvite z ogrodjem **Streamlit**. Aplikacija bo omogočala:
- vnos parametrov hipotetičnega posojila in takojšnjo napoved verjetnosti neplačila,
- prikaz SHAP grafov za razlago napovedi,
- pregled ključnih vizualizacij iz analize (porazdelitve, ROC krivulje, gručenje).

## Orodja

Python (pandas, scikit-learn, mlxtend, XGBoost, SHAP, HuggingFace Transformers, Streamlit), Jupyter Notebook.
