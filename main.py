import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Carregamento do dataset

def load_data(filename):
    data = pd.read_csv(filename, sep=",")
    return data
passengers = load_data("titanic_v2.csv")

#Informação dataset

def basic_info(passengers):
    print("\nPrimeiras linhas:")
    print(passengers.head())

    print("\nInformação geral:")
    print(passengers.info())

    print("\nValores nulos:")
    print(passengers.isnull().sum())

#Pré-processamento de dados para o modelo
def logistic_regression(passengers):

    # Converter variável categórica (sex)
    if passengers["Sex"].dtype == "object":
        passengers["Sex"] = passengers["Sex"].map({"male": 0, "female": 1})

    # Eliminar nulos
    passengers_model = passengers[["Survived", "Pclass", "Sex"]].dropna()

    # Baralhar os dados
    passengers_model = passengers_model.sample(frac=1, random_state=42).reset_index(drop=True)

    # Divisão dos dados
    split = int(0.7 * len(passengers_model))
    train = passengers_model.iloc[:split]
    test = passengers_model.iloc[split:]

    #Modelo de regressão logistica ajustado
    model = smf.logit("Survived ~ Pclass + Sex", data=train).fit()
    print("\nResumo do modelo:")
    print(model.summary())

    # Previsões
    y_pred_prob = model.predict(test)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    y_test = test["Survived"]

    # Matriz de confusão
    conf_matrix = pd.crosstab(y_test, y_pred)

    print("\nMatriz de Confusão:")
    print(conf_matrix)

    # Precisão
    accuracy = (y_test == y_pred).mean()
    print("\nTaxa Global de Acerto:", round(accuracy, 4))


def main():

    passengers = load_data("titanic_v2.csv")

    # Seleção de colunas 
    cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    passengers = passengers[cols].copy()

    basic_info(passengers)

    # Tratamento simples de valores nulos
    passengers["Age"].fillna(passengers["Age"].median(), inplace=True)
    passengers["Embarked"].fillna(passengers["Embarked"].mode()[0], inplace=True)

    logistic_regression(passengers)


if __name__ == "__main__":
    main()


## Conclusões:

# O modelo acertou 78% das previsões no conjunto de teste.
# Observa-se que a variável "Sex" tem forte influência na sobrevivência.
# Para melhorar, podemos incluir variáveis contínuas como Age e Fare.