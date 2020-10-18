'''
Sampling Bias

Quando geramos um conjunto aleatório de dados, eles serão aleatório. 
Acontece, que muitas vezes precisamos definir parâmetros de aleatoriedade que queremos.
Por exemplo, uma empresa decide fazer uma pesquisa e chama 1.000 pessoas de forma aleatória 
para responde-la por telefone. 
Essa amostra DEVE representar a população. 
Se nos EUA 51.3% da população é de mulheres e 48.7% são de homens, uma pesquisa bem conduzida deve 
manter essa proporção na amostra: 513 mulheres e 487 homens.
Se for utilizado apenas amostras aleatórias, haverá chance de enviesar o resultado.

Para resolver isso, precisamos saber a proporção de cada amostra. Por exemplo:

Homens entre 20-30 anos (20% da amostra)
Homens entre 31-50 anos (15% da amostra)
Mulheres entre 20-30 anos (32% da amostra) ....

O conjunto de Train/Test tem que respeitar essa proporção na aleatoriedade. Senão, corre risco dele
pegar apenas valores da classe mais representativa (de forma aleatóra), ou até pior, pegar muitas amostras
das classes menos representativa (né aleatório)

Para isso precisamos fazer um binning dividindo por categorias
'''

def binning(df, column, n_bins, labels=False):
    bins = np.linspace(min(df[column]), max(df[column]), n_bins+1)
    if labels:
        return pd.cut(df[column], bins, labels)
    else:
        lab = []
        for x in range(n_bins):
            lab.append(x+1)
        return pd.cut(df[column], bins, labels = lab)    
    

'''
É bom plotar para ver a distribuição.

Agora que temos uma categoria que estratifica a distribuição da feature em binnings, 
podemos estratificar o conjunto de 'test' da mesma forma que a amostra com a classe 
StratifiedShuffleSplit do sk-learn:
'''

# É preciso dividir o conjunto em train test antes. O DF que utilizamos é o 'binning'.

def stratified_shuffle(df_binning, column):
    from sklearn.model_selection import StratifiedShuffleSplit
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(df_binning, df_binning[column]):
        strat_train_set = df_binning.loc[train_index]
        strat_test_set = df_binning.loc[test_index]
    
    return strat_train_set, strat_test_set

# Retorna uma tupla com 2 DATAFRAMES. Posição[0] = treino, Posição[1]=teste. Precisamos armazenar em variáveis:
# ex1.: variavel1, variavel2 = stratified_shuffle(df_binning, column)
# ex2.: strat_train_set, strat_test_set = stratified_shuffle(housing, 'income_cat') 
# Testando a distribuição: strat_test_set[column].value_counts() / len(strat_test_set)

'''
Comparando as proporções do BINNING com o TRAIN/TEST gerada acima:
Vamos comparar a distribuição BINNING, TRAIN_TEST do DF BINNING e TRAIN_TEST aleatório 
(para vermos a diferença entre o aleatório e o BINNING).

Uma função busca informações dentro da outra. Comparamos a distribuição do conjunto de TEST
'''

def cat_proportions(data, column):
    return data[column].value_counts() / len(data)

def compare_props(df, strat_df, column):
    
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) # aleatório
    
    compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": cat_proportions(strat_test_set, column),
    "Random": cat_proportions(test_set, column),
    }).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    
    return compare_props

# Exemplo: compare_props(housing, strat_test_set, 'income_cat')
# Depois é preciso excluir a variável dummy (income_cat) do conjunto TEST e TRAIN 
# para retornar o dataset ao original.


# Como dropar a coluna em mais de um DF (ex.: dropar no test, train e validation)
# Falta criar a função
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# Separar o DF em categórico e numerico (falta criar a função)
def cat_num(df):
    numerical_features = train.select_dtypes([np.number]).columns.tolist()
    categorical_features = train.select_dtypes(exclude = [np.number,np.datetime64]).columns.tolist()
    return numerical_features, categorical_features



# Analizando e plotando histograma de coluna numérica e vendo suas caractersticas
def describe_column(df, column):
    print(df[column].describe())
    sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
    sns.distplot(
        df[column], norm_hist=False, kde=True
    ).set(xlabel=column, ylabel='Qtd');

# Histograma de TODAS as colunas numéricas do DF.
#train[numerical_features].hist(figsize=(30, 35), layout=(12, 4));

k = len(train[numerical_features].columns)
n = 3
m = (k - 1) // n + 1 ## Floor Division (also called Integer Division)
fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))
for i, (name, col) in enumerate(train[numerical_features].items()):
    r, c = i // n, i % n
    ax = axes[r, c]
    col.hist(ax=ax)
    ax2 = col.plot.kde(ax=ax, secondary_y=True, title=name)
    ax2.set_ylim(0)

fig.tight_layout()



# Analyzing Relationships Between Numerical Variables and the target ['revneue']
fig, ax = plt.subplots(10, 4, figsize=(30, 35))
for variable, subplot in zip(numerical_features, ax.flatten()):
    sns.regplot(x=train[variable], y=train['revenue'], ax=subplot)


# Analyzing Relationships between the Categorical Variables and the target ['revenue']
fig, ax = plt.subplots(3, 1, figsize=(40, 30))
for var, subplot in zip(categorical_features, ax.flatten()):
    sns.boxplot(x=var, y='revenue', data=train, ax=subplot)



'''
Plottting funcs
'''

def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    '''
    Helper function that gives a quick summary of a given column of categorical data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data, y would be the count
    y: str. vertical axis to plot the labels of categorical data, x would be the count
    hue: str. if you want to compare it another variable (usually the target variable)
    palette: array-like. Colour of the plot
    Returns
    =======
    Quick Stats of the data and also the count plot
    '''
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
    plt.show()

# ex.: Target Variable: Survival
#    c_palette = ['tab:blue', 'tab:orange']
#    categorical_summarized(train_df, y = 'Survived', palette=c_palette)

# # Feature Variable: Gender
# categorical_summarized(train_df, y = 'Sex', hue='Survived', palette=c_palette)


def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    '''
    Helper function that gives a quick summary of quantattive data
    Arguments
    =========
    dataframe: pandas dataframe
    x: str. horizontal axis to plot the labels of categorical data (usually the target variable)
    y: str. vertical axis to plot the quantitative data
    hue: str. if you want to compare it another categorical variable (usually the target variable if x is another variable)
    palette: array-like. Colour of the plot
    swarm: if swarm is set to True, a swarm plot would be overlayed
    Returns
    =======
    Quick Stats of the data and also the box plot of the distribution
    '''
    series = dataframe[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe,
                      palette=palette, ax=ax)

    plt.show()

# # univariate analysis
# quantitative_summarized(dataframe= train_df, y = 'Age', palette=c_palette, verbose=False, swarm=True)

# bivariate analysis with target variable
# quantitative_summarized(dataframe= train_df, y = 'Age', x = 'Survived', palette=c_palette, verbose=False, swarm=True)

# # multivariate analysis with Embarked variable and Pclass variable
# quantitative_summarized(dataframe= train_df, y = 'Age', x = 'Embarked', hue = 'Pclass', palette=c_palette3, verbose=False, swarm=False)


# Trabalhando com imagens (ou não) e função LOC.

import ipywidgets as widgets
from ipywidgets import interact, interact_manual

df.loc[df['claps'] > 500]

@interact
def show_articles_more_than(column='claps', x=5000):
    return df.loc[df[column] > x]


# Interact with specification of arguments
@interact
def show_articles_more_than(column=['claps', 'views', 'fans', 'reads'], 
                            x=(10, 100000, 10)):
    return df.loc[df[column] > x]

# vendo imagens dentro de um diretório

import os
from IPython.display import Image
@interact
def show_images(file=os.listdir('images/')):
    display(Image(fdir+file))


# Plotaando Widgets
import cufflinks as cf

@interact
def scatter_plot(x=list(df.select_dtypes('number').columns), 
                 y=list(df.select_dtypes('number').columns)[1:],
                 theme=list(cf.themes.THEMES.keys()), 
                 colorscale=list(cf.colors._scales_names.keys())):
    
    df.iplot(kind='scatter', x=x, y=y, mode='markers', 
             xTitle=x.title(), yTitle=y.title(), 
             text='title',
             title=f'{y.title()} vs {x.title()}',
            theme=theme, colorscale=colorscale)


































def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating
    '''
    for col in cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(columns=col, axis=1),
                            pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)],
                           axis=1)
        except:
            continue
    return df;





