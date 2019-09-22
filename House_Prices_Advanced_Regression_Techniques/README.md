# House Prices Advanced Regression Techniques
https://www.kaggle.com/c/house-prices-advanced-regression-techniques
--------------------------------------------------------------
Melhor resultado obtido até agora (Root Mean Squared Logarithm Error) = 0.17055

--------------------------------------------------------------
### 1. Requisitos:
 - Ubuntu 16.04
 - [Miniconda](https://conda.io/miniconda.html) - Python 3.7 - Linux 64-bit (Bash installer)
 - [JupyterLab](https://jupyter.org/install) - Instalar e rodar o Jupyter Notebook
    ```sh
    $ jupyter notebook
    ```
 - [Pip](https://pip.pypa.io/en/stable/installing/) - Instalador *get-pip.py*
 - Dependências em `requirements.txt`:
    ```sh
    $ pip install --no-cache-dir -r requirements.txt 
    ```

### 2. Estrutura de diretórios:
 - **codigo**: contém codigo Python no notebook(*.ipynb*) com a solução ```futuramente subirei numa estrutura de projeto Python com refatoração do codigo``` 
 - **data**: contém todos os dados de treino original e teste, além dos datasets transformados que são gerados no script, tudo em *.CSV* 
 - **understanding_info**: contém arquivos com informações referentes a pré-análise dos dados(*features_evaluation.csv*) e suas descrições (*data_description*), além de imagens

### 3. Entendendo o problema:
> Avaliar o quanto cada uma das features do dataset influencia no preço da casa **em Ames**

- Classificação de relevância de cada feature no preço de venda de uma casa em Ames(EUA)
    -  Avaliar as features e seus significados de acordo com seu contexto.
        - Exemplo 1:  O que significa número de cômodos "above grade"? Isso está relcionado com a estrura das casas americanas conforme mostra a figura *house_structure_usa.jpg*, e parece ser bem padrão nos EUA ter 2 "andares".Features relacionadas a "above grade": ['GrLivArea'],['HalfBath'], ['FullBath'], ['Bedroom'], ['Kitchen']
        - Exemplo 2: Quão importante é ter um bom sistema de aquecimento para uma casa em Ames? Pelo gráfico de temperatura de Ames em *climate_ames.jpeg* é possível ver que é frio o ano todo, logo isso é bem mais relevante em Ames do que aqui no Brasil. Features relacionadas: ['Fireplaces'],['FireplaceQu'],['HeatingQC'],['CentralAir']
- Mapear features que estão fortemente vinculadas a qualquer outra, podendo ser até redudantes através do arquivo de descrição *data_description.txt*

### 4. Data Cleaning:
- Features com Missing data: 
	- ['LotFrontage']	==> muitas samples
	- ['GarageYrBlt'] 	==> muitas samples
	- ['MasVnrArea']  ==> mesmas 8 samples com missing data
	- ['MasVnrType'] 	==> mesmas 8 samples com missing data
- Outliers: 
    - Amostras cujo SalePrice é grande demasiado, eles representam apenas 1% do dataset
    - Do 450.000 ate os 700.000 há muitos "espaços vazios" de valores >> perigo de possível overfit durante o treino para esse intervalo de valores

### 5. Feature Engineering: 
 - Quanto mais features diminuir melhor -> Curse of Dimensionality - temos apenas 1460 amostras totais para 79 features
 - Excluir features rotuladas com baixa relevancia no arquivo *features_evaluation.csv*
 - Excluir amostras cujas features tem baixa relevancia e muito pouco missing data(neste caso somente tem 8 amostras totais excluídas)
 - Criação de novas features com base na combinação daquelas que sao fortemente relacionadas no arquivo *features_evaluation.csv*
 - Feature Selection: através da correlação entre elas e do teste chi-quadrado, selecionando as 15 features com menos independencia com SalePrice, analisando se nao tem redundancia entre elas
- Transformação dos valores de features categoricas em numericas: numeros inteiros associados a cada categoria
- Feature Scaling: mean normalization apenas com as features originalmente numericas, as categoricas nao foram normalizadas já que seu conjunto possível de valores discretos são pequenos
### 6. Construção dos Modelos Preditivos:
- Algoritmos usados para fazer predição dos valores de SalePrice
    - SVR - Supporting Vector Regression
    - Ridge Regression (Regressao Linear com fator de Regularização)
    - Random Forest Regressor
- Uso de Cross Over Validation com 10 folds: seleciona melhor modelo de 1 dos 10 folds
	- 10 folds para ter 90% do total para treino e apenas 10% para validacaos
    - Critério da escolha: RMSE entre os valores preditos naquele fold com os da validação são menores
- Escolha do modelo final Random Forest: tem propriedade de mesmo com features correlacionadas entre si gerar bom resultado, pois devido aos varios numeros de arvores que compensam o efeito de correlacao das features
