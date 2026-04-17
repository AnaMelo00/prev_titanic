### Previsão da sobrevivência no Titanic com Regressão Logística:
 Implementar um script de Python que utilize o dataset “titanic_v2.csv” para construir um modelo de regressão logística, com o intuito de prever se um passageiro sobreviveu ou não ao trágico desastre do Titanic. 

 ### Estrutura do projeto:

Pasta EXERCICIO_TITANIC contém os seguintes ficheiros:

README:  Ficheiro-resumo do projeto e objetivos do mesmo e conclusões gerais.

__init__.py: Ficheiro que permite importar módulos. 

 titanic.ipynb: Ficheiro com gráficos e interpretação de resultados.

 main.py: Ficheiro que tem por base o script do projeto.

## Fases do projeto:

1. Importação das bibliotecas necessárias. 

2. Carregamento do datase (“titanic_v2.csv” ). 

3. Pré-processamento dos dados.

4. Divisão dos dados e treino do modelo.

5. Previsão e avaliação do modelo.

 ### Como correr o projeto:

1. Abrir um terminal
2. Instalar bibliotecas necessárias

```bash
pip install numpy as np
pip install pandas as pd
pip install seaborn as sns
pip install statsmodels.api as sm
pip install statsmodels.formula. api as smf
```
3. Ir até à pasta do projeto (main.py)
4. Executar o script
```bash
python main.py
```

## Dependências

Instalação e importação das bibliotecas:
 1) Numpy
 2) Pandas
 3) Seaborn 
 4) Statsmodel (e statsmodel.formula.api)
 
```bash
pip install numpy as np
pip install pandas as pd
pip install seaborn as sns
pip install statsmodels.api as sm
pip install statsmodels.formula. api as smf

```

## Conclusões:

- O modelo acertou 78% das previsões no conjunto de teste.
- Observa-se que a variável "Sex" tem forte influência na sobrevivência (mulheres com mais chance de sobreviver e passageiros da classe 1).
- Para melhorar, podemos incluir variáveis contínuas como Age e Fare e analisar mais detalhadamente.
