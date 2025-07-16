# Proiect 9 – Predicția șomajului în România (SVM și Rețele Neuronale)

Acest repository conține un proiect de analiză predictivă realizat în R, care vizează modelarea probabilității ca un respondent să fie șomer pe baza unor variabile socio-demografice și ocupaționale, folosind seturi de date LFS pentru România (2010–2013).

### 1. Prelucrarea datelor

Datele au fost combinate din fișierele brute RO_LFS_2010_Y.csv, RO_LFS_2011_Y.csv, RO_LFS_2012_Y.csv și RO_LFS_2013_Y.csv. Pașii de prelucrare includ:
- etichetarea variabilei țintă (șomer = 1 dacă ILOSTAT = 2)
- selecția variabilelor relevante (vârstă, sex, educație, regiune, statut ocupațional etc.)
- împărțirea în seturi de antrenare și testare
- curățarea datelor și eliminarea nivelurilor redundante

### 2. Model SVM (Support Vector Machine)

Pe un eșantion de 5000 de cazuri:
- modelul SVM a fost antrenat cu kernel radial
- s-au evaluat: matricea de confuzie, AUC, și curba ROC
- s-au generat grafice pentru analiza performanței

### 3. Model MLP (Rețea neuronală)

Pe un eșantion de 5000 (train) și 2000 (test):
- rețeaua a fost antrenată cu 4 neuroni ascunși
- s-au standardizat datele (center & scale)
- s-au evaluat metrice precum: accuracy, F1-score, ROC-AUC
- au fost generate grafice de tip matrice de confuzie și ROC comparativ

### 4. Evaluarea performanței

Pentru ambele modele au fost comparate:
- Matricea de confuzie (vizualizată cu ggplot2)
- Scorurile AUC din pROC
- Performanța generală pe baza curbei ROC

### Fișiere

proiect.R – codul sursă scris în limbajul R, care prelucrează datele, antrenează modelele și generează predicții

train.csv, test.csv – fișiere CSV cu seturile de antrenare și testare (versiuni reduse)

## Proiect realizat în echipă
