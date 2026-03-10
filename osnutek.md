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

## Pričakovani rezultati in prikaz

Primarni rezultat projekta je napovedni model, ki za posamezno posojilo vrne **verjetnost neplačila** (vrednost med 0 in 1) skupaj z razvrstitvijo posojilojemalca v eno od skupin tveganja (nizko / srednje / visoko). Pričakujemo, da bo model XGBoost dosegel najboljšo ločljivost med razredi (AUC-ROC > 0,75), logistična regresija pa bo služila kot interpretabilna izhodišče za primerjavo.

Poleg numeričnih metrik bomo z metodo SHAP prikazali, kateri dejavniki so pri posamezni napovedi odločilni — kar omogoča razlago napovedi tudi nestrokovnjakom.

Rezultati in model bodo predstavljeni v obliki interaktivne spletne aplikacije, razvite z ogrodjem **Streamlit**. Aplikacija bo omogočala:
- vnos parametrov hipotetičnega posojila in takojšnjo napoved verjetnosti neplačila,
- prikaz SHAP grafov za razlago napovedi,
- pregled ključnih vizualizacij iz analize (porazdelitve, ROC krivulje, gručenje).

## Orodja

Python (pandas, scikit-learn, mlxtend, XGBoost, SHAP, HuggingFace Transformers, Streamlit), Jupyter Notebook.
