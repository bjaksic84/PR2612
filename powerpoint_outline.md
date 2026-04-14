# Predstavitev projekta: Napovedovanje neplačila posojil s strojnim učenjem

## SLIDE 1

### [Section 1] Naslov
**Napovedovanje tveganja neplačila posojil s pomočjo strojnega učenja**

### [Section 2] Podatki
* **Vir podatkov:** Zbirka podatkov »Lending Club« (2007–2018), pridobljeno iz [Lending Club public data/Kaggle]. Uporabljamo en osrednji združen vir.
* **Prvotni namen zbiranja:** Avtorji/platforma so podatke zbirali za arhiviranje posojilnih transakcij, spremljanje bonitetnih ocen in upravljanje lastnega portfelja na peer-to-peer finančnem trgu.
* **Tip in obseg podatkov:** 
  * Tabularni podatki s 151 atributi (številski, kategorični), vključno z besedilnimi polji (opis posojila, naziv zaposlitve).
  * Izvorna baza vsebuje več kot 2 milijona zapisov. Za ta projekt je bilo izvedeno stratificirano vzorčenje na natanko **20.000 vrstic** za zagotavljanje računske obvladljivosti.
  * Binarna klasifikacija: 2 razreda (Plačano `0` in Neplačano `1`).
* **Težave in manjkajoči zapisi:** Močno asimetrična porazdelitev ciljne spremenljivke (~80 % : 20 %). Prisotnost močnega šuma in manjkajočih vrednosti v več kot 100 netemeljnih atributih (npr. meseci od zadnje zamude ipd.).
* **Predprocesiranje:**
  * Filtriranje iz 151 na 45 relevantnih značilk.
  * Sistematična imputacija manjkajočih vrednosti in standardizacija številskih atributov.
  * *One-Hot Encoding* kategoričnih spremenljivk.
  * NLP/TF-IDF vektorizacija besedilnih polj za zajem tveganih ključnih besed.

### [Section 3] Glavna vprašanja/cilji podatkovnega rudarjenja
* **Cilj 1:** Kateri finančni in demografski pokazatelji (npr. obrestna mera, DTI, lastništvo nepremičnine) najmočneje korelirajo z zvišanim tveganjem za neplačilo posojila?
* **Cilj 2:** Kateri algoritem strojnega učenja (Logistična regresija, Random Forest, XGBoost) najučinkoviteje in z najvišjim priklicem zmore napovedati te tvegane profile glede na močno asimetrijo razredov?


---

## SLIDE 2

### [Section 1 - Half] Podroben opis metod 
* **Iskanje vzorcev in asociacij (Apriori):** Kategorične in diskretizirane atribute analiziramo v obliki trditev, da izluščimo tiste kombinacije profilov, ki najpogosteje ostanejo neplačani.
* **Odrastitev dimenzij (Feature Selection):** Uporaba Naključnega gozda (Random Forest) za identifikacijo najvplivnejših spremenljivk, s čimer oklestimo atributno mizo na 19 ključnih indikatorjev (po NLP obdelavi in uporabi K-Means pa se razširi na konelnih 68 atributov).
* **Segmentacija ne-nadzorovanega učenja:** Z algoritmom K-Means ugotovimo obnašanje podatkov v n-dimenzionalnem prostoru in jih združimo v optimalne 3 gruče (*risk clusters*).
* **Napovedno modeliranje (Klasifikacija):** Na učnem in testnem naboru (z asimetričnim uteževanjem razredov) zoperstavimo interpretabilno **Logistično regresijo**, globoki **Naključni gozd** ter zmogljivi **XGBoost**. Uspešnost ocenjujemo primarno preko ploščine pod krivuljo (ROC-AUC) ter metrike priklica (Recall) za razred tveganih strank.

### [Section 2 - Half] Dosedanje ugotovitve in rezultati
* **Ugotevitve profilov (Apriori):** Tipičen posameznik, podvržen neplačilu kredita, je **najemnik** (nima lastne nepremičnine), najema posojilo za **konsolidacijo dolgov** in ima določeno **visoko obrestno mero**.
* **Dimenzije:** Pokazalo se je, da celotno usodo posojila v veliki meri diktira razmerje dolga proti dohodku (DTI), posameznikova kreditna ocena (FICO) in višina obrestne mere.
* **Paradoks pri klasifikaciji (Zmaga preprostosti):** Naši podatki niso zelo zapleteni, zato preprosta in hitra **Logistična regresija** presenetljivo premaga napredni *XGBoost*. Regresija uspešneje prepozna slabe posojilojemalce (najde jih 64 %, XGBoost pa 61 %).

---

## SLIDE 3 (Optional)

### Rezultati v grafih in odprta vprašanja
* **Grafikalna podpora:** 
  * *(Tukaj vstavite Slika 2 iz vmesnega poročila – Feature Importances bar chart)*
  * *(Tukaj vstavite Slika 3 iz vmesnega poročila – ROC Krivulja z AUC metrikami)*
* **Dvodimenzionalna skalabilnost:** Izvozili smo tako interpretabilen model (Logistično) kot globok model (XGBoost). Odprto vprašanje ostaja: "Pri katerem volumnu podatkov in količini anomalij v polni *Lending Club* bazi z 2.000.000 vpisi bi XGBoost pokazal svojo dejansko prevlado in prekosil regresijo?"