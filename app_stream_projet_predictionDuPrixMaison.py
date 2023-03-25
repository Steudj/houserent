from gettext import install
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.write(""" # Prediction du prix d'une maison""")

#import dataset
data=pd.read_csv(r"data/train.csv")
st.write("### Head of the train DataFrame")
st.write(data.head())

text = "Les données fournies dans ce lien sont relatives aux prix de vente de maisons à Ames, Iowa. Elles sont divisées en deux fichiers : un fichier de formation (train.csv) contenant les données pour l'entraînement des modèles de prédiction, et un fichier de test (test.csv) pour lesquelles les participants à la compétition Kaggle devront prédire les prix de vente.\n\nLe fichier de formation contient 1460 observations et 81 variables. Les variables représentent différents aspects de la maison, tels que la taille du lot, la qualité de la construction, l'âge de la maison, les caractéristiques des pièces, etc. La variable cible est \"SalePrice\", qui représente le prix de vente final de la maison.\n\nLes données peuvent être analysées à l'aide de différentes techniques d'analyse exploratoire des données, telles que :\n\nAnalyse univariée : Cette technique permet de comprendre la distribution de chaque variable dans le jeu de données. Elle peut être réalisée à l'aide de techniques telles que l'histogramme, la boîte à moustaches et le diagramme en barres.\n\nAnalyse bivariée : Cette technique permet d'analyser la relation entre deux variables. Elle peut être réalisée à l'aide de techniques telles que le diagramme de dispersion et le coefficient de corrélation.\n\nAnalyse multivariée : Cette technique permet d'analyser la relation entre plusieurs variables. Elle peut être réalisée à l'aide de techniques telles que l'analyse en composantes principales (ACP) et l'analyse factorielle discriminante (AFD).\n\nEn analysant les données, il peut être possible de déterminer quelles variables ont un impact significatif sur le prix de vente de la maison. En outre, il peut être possible de découvrir des tendances, des modèles ou des relations intéressantes entre les variables. Ces résultats peuvent être utilisés pour développer des modèles de prédiction plus précis."

st.write(text)


st.write(data.info())


#Vérification des valeurs manquantes
st.write("### Vérification des valeurs manquantes")
st.write(data.isna().sum())

text = "Nous constatons la présence de valeurs manquantes dans plusieurs colonnes du dataframe"
st.write(text)

#Observations sur les donnees
st.write("### Observations sur les donnees")
dfFeatures = []
df = data
for i in df.columns:
    dfFeatures.append([i, df[i].nunique(), df[i].drop_duplicates().values, df[i].count(), df[i].dtype])
# create dataframe from summary
df_summary = pd.DataFrame(dfFeatures, columns = ['Features', 'Unique', 'Values', 'count', 'Type'])

# display dataframe as table in Streamlit app
#st.write(df_summary)


# Histogrammes pour les variables numériques
# st.write("### Histogrammes pour les variables numériques")

num_vars = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
            'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
            '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
            'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
            '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']

for var in num_vars:
    fig, ax = plt.subplots()
    ax.hist(data[var], bins=30)
    ax.set_title(var)
    st.pyplot(fig) 



# Diagrammes en barres pour les variables catégorielles
st.write("### Diagrammes en barres pour les variables catégorielles")







# Diagrammes en barres pour les variables catégorielles
cat_vars = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
            'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
            'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 
            'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
            'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
            'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
            'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

for var in cat_vars:
    plt.figure(figsize=(8, 6))
    data[var].value_counts().plot(kind='bar')
    plt.title(var)
    st.pyplot()

st.write("Pour la colonne Street nous avons plus de 1400 rue pavées et moins de 5 rues non pavées.")
st.write("Pour Alley on compte 50 Grvl et 43 Pave.")
st.write("Pour LotShape on compte 900 Reg, 420 IR1.")
st.write("Pour Utilities nous avons plus de 1400 logements qui bénéficient de toutes les infrastructures et moins de 5 logements qui n'en bénéficient pas.")

# Boxplot des données numeriques
st.write("### Boxplot des données numeriques")

def boxplot(data, numeric):
    fig = plt.figure(figsize=(40,40))
    for i, col in enumerate(numeric):
        ax = fig.add_subplot((len(num_vars)//2)+1, 2, i+1)
        sns.boxplot(data=data, x=col, ax=ax, palette="husl")
        ax.set_title(f"Boxplot of {col}")
    st.pyplot(fig)
    
boxplot(data, num_vars)

text = "Nous observons la présence de valeurs aberrantes dans notre jeu de données,donc nous allons lister les colonnes impactées et corriger celles-ci."
st.write(text)

# Detection des outliers
st.write("### Detection des outliers")
def outliers_detection(data, col):
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    lower = data[col].quantile(0.25) - (1.5*IQR)
    upper = data[col].quantile(0.75) + (1.5*IQR)

    outliers = data.index[(data[col] < lower) | (data[col] > upper)]

    return outliers

features_outliers = list()
for col in num_vars:
    if len(outliers_detection(data, col)) != 0:
        features_outliers.append(col)

st.write("features_outliers")


#Copie du dataframe
df1 = data[['LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'YearBuilt',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'BsmtFullBath',
 'BsmtHalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageCars',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal',
 'SalePrice']]


# Remplacement des valeurs atypiques
st.write("### Detection des outliers et remplacement des valeurs atypiques")
def replaceAttypicalValues():
    for col in df1.columns:
        IQR = df1[col].quantile(0.75) - df1[col].quantile(0.25)
        Lower = df1[col].quantile(0.25) - (1.5*IQR)
        upper = df1[col].quantile(0.75) + (1.5*IQR)
        
        for i in range(len(data[col].values)):
            if(df1[col].values[i]>upper):
                df1[col].values[i]=upper
            if (df1[col].values[i]<Lower):
                df1[col].values[i]=Lower
replaceAttypicalValues()

st.write("#### Boxplot avec les valeurs atypiques remplacées")
boxplot(df1, df1)
text = "Les valeurs atypiques ont bien été remplaçées"




# Tracer les diagrammes en barres pour les variables catégorielles
# Plot countplots for categorical variables
# fig, axs = plt.subplots(ncols=3, nrows=15, figsize=(30, 100))
# for i, var in enumerate(cat_vars):
#     row = i // 3
#     col = i % 3
#     sns.countplot(data[var], ax=axs[row, col])
#     axs[row, col].set_title(var)
#     axs[row, col].tick_params(axis='x', labelrotation=90)

# # Show the plot in Streamlit
# st.pyplot(fig)

# Tracer les diagrammes en barres pour les variables catégorielles
# fig, axs = plt.subplots(ncols=3, nrows=15, figsize=(30, 100))

# # Loop through the variables and create the countplot
# for i, var in enumerate(cat_vars):
#     row = i // 3
#     col = i % 3
#     sns.countplot(data[var], ax=axs[row, col])
#     axs[row, col].set_title(var)
#     axs[row, col].tick_params(axis='x', labelrotation=90)

# # Display the plot in Streamlit
# st.pyplot(fig)


import streamlit as st

st.write("MSZoning : Identifie la classification générale de zonage de la vente.")
st.write("1. Environ 1200 proprietes sont classifiees dans la zone Résidentielle à faible densité")
st.write("2. 200 proprietes sont en zone Résidentielle de moyenne densité")
st.write("3. 50 proprietes sont dans un Village flottant résidentiel")
st.write("4. Environ 5 logements sont Résidentiels haute densité")
st.write("")
st.write("Frontage du lot : Pieds linéaires de la rue reliée à la propriété")
st.write("LotArea : Taille du lot en pieds carrés")
st.write("Street : Type de route d'accès à la propriété")
st.write("1. 99% des proprietes sont accessibles via des routes pavees")
st.write("2. 1% de logements accessibles par des routes faites de cailloux")
st.write("")
st.write("Alley : Type de ruelle donnant accès à la propriété")
st.write("1. 50 logements accesiibles par Gravel")
st.write("2. 40 accessibles par Pavé")
st.write("")
st.write("LotShape : Forme générale de la propriété")
st.write("1. 1000 terrains ont une forme Réguliere")
st.write("2. 450 terrains ont une forme Légèrement irréguliere")
st.write("3. 10  Modérément irrégulier")
st.write("4. 5 Irrégulier")
st.write("")
st.write("LandContour : Allure du terrain")
st.write("1.1400 proprietes ont un terrain Presque plat/niveau")
st.write("2. 100 terrains possedent une  pente - Montée rapide et significative du niveau de la rue au bâtiment")
st.write("3. 100 terrains possedent une Pente importante d'un côté à l'autre.")
st.write("4. 25 proprietes sont sur un terrain à Faible dépression")
st.write("")
st.write("Utilities : Type de services publics disponibles")
st.write("1. 1300 proprietes disposent de Tous les services publics (E,G,W,& S)")
st.write("2. 1 propriete possede Électricité et gaz seulement")
st.write("")
st.write("LotConfig : Configuration du terrain")
st.write("1. 1100logements ont un Lot intérieur")
st.write("2. 250 logements ont un Lot d'angle")
st.write("3. 100 logements CulDSac Cul-de-sac")
st.write("4. 50 possedent une Façade sur 2 côtés de la propriété")
st.write("5. 5 possedent une Façade sur 3 côtés de la propriété")
st.write("")
st.write("LandSlope: Pente de la propriété")
st.write("Gtl Pente douce")
st.write("Mod Pente modérée")
st.write("Sev Severe Slope")
st.write("")

st.write("Neighborhood : Emplacements physiques dans les limites de la ville d'Ames")

st.write("""
    3. en dernier viennent les proprietes situees à Blueste Bluestem   
    2. 150 proprietes sont situees CollgCr College Creek
    1. 300 proprietes sont localisees à (elles representent la majorite des logements) NWAmes Northwest Ames
""")

st.write("Condition1 : Proximité de diverses conditions")
st.write("""
    Artery Adjacent à une rue artérielle
    1. 100Feedr Adjacent à une rue de desserte	
    2. 1300Normal	
    RRNn A moins de 200' d'une voie ferrée nord-sud
    RRAn Adjacent à une voie ferrée nord-sud
    PosN Près d'un élément positif hors site - parc, ceinture verte, etc.
    PosA Adjacent à un élément positif hors site
    RRNe À moins de 200 pieds d'une voie ferrée est-ouest
    RRAe Adjacent à une voie ferrée est-ouest
""")

st.write("Condition2 : Proximité de diverses conditions (si plus d'une est présente)")
st.write("""
    Arterial Adjacent à une rue artérielle
    Feedr Adjacent à une rue de desserte	
    Norm Normal	
    RRNn A moins de 200' d'une voie ferrée nord-sud
    RRAn Adjacent à une voie ferrée nord-sud
    PosN Près de positif o
""")

st.write("BldgType: Type of dwelling")
st.write("""
    1Fam	Single-family Detached	
    2FmCon	Two-family Conversion; originally built as one-family dwelling
    Duplx	Duplex
    TwnhsE	Townhouse End Unit
    TwnhsI	Townhouse Inside Unit
""")

         
        



st.write("#### Analyse Bivariée")


# Diviser les variables en groupes de trois
variable_groups = [data.columns[i:i+3] for i in range(0, len(data.columns), 3)]

# Tracer les graphiques de relations pour chaque groupe de variables
for group in variable_groups:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i, variable in enumerate(group):
        sns.scatterplot(x=variable, y='SalePrice', data=data, ax=axes[i])
    st.pyplot(fig)


st.write("### Analyse Multivarié")
corramt = data.corr()
fig, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(corramt, vmax=0.8, square=True, fmt='.2f', cmap='coolwarm', ax=ax)

st.pyplot(fig)
text = "Cette figure représente la matrice de correlation de notre jeu de données"
st.write(text)

st.write("#### Affichage des variables hautement correlées avec la variable réponse SalePrice")

text = "Les variables fortement correlées avec SalePrice sont OverallQual, GrLivArea, GarageCars, GarageArea,TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd "
st.write(text)
