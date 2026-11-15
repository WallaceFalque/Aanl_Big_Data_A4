import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(BASE, "dados", "zomato.csv")
ARQUIVO_LIMPO = os.path.join(BASE, "dados", "zomato_limpo.csv")
PASTA_GRAFICOS = os.path.join(BASE, "graficos")
PASTA_RELATORIOS = os.path.join(BASE, "relatorios")
SEMENTE = 42

AZUL = "#2a78d6"
LARANJA = "#eb6834"
VERDE_AGUA = "#1baf7a"
CINZA = "#8a8984"
TEXTO = "#0b0b0b"
TEXTO_SEC = "#52514e"
FUNDO = "#fcfcfb"
CORES_CATEGORICAS = [AZUL, LARANJA, VERDE_AGUA]
MAPA_SEQUENCIAL = LinearSegmentedColormap.from_list("azul", ["#eef4fc", "#8db8ea", "#2a78d6", "#123f75"])
MAPA_DIVERGENTE = LinearSegmentedColormap.from_list("div", ["#d95926", "#f3f2ef", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": FUNDO,
    "axes.facecolor": FUNDO,
    "savefig.facecolor": FUNDO,
    "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": TEXTO_SEC,
    "axes.titlecolor": TEXTO,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": TEXTO_SEC,
    "ytick.color": TEXTO_SEC,
    "font.size": 10,
    "legend.frameon": False,
    "figure.dpi": 110,
})

relatorio = []


def registrar(texto=""):
    print(texto)
    relatorio.append(texto)


def titulo_etapa(nome):
    registrar("")
    registrar("=" * 78)
    registrar(nome)
    registrar("=" * 78)


def salvar(fig, nome):
    caminho = os.path.join(PASTA_GRAFICOS, nome)
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    registrar(f"  [grafico] graficos/{nome}")


def barras_horizontais(serie, titulo, xlabel, nome, formato="{:.0f}", cor=AZUL):
    serie = serie.sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, len(serie) * 0.38)))
    barras = ax.barh(serie.index.astype(str), serie.values, color=cor, height=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    limite = serie.max()
    for barra, valor in zip(barras, serie.values):
        ax.text(barra.get_width() + limite * 0.01, barra.get_y() + barra.get_height() / 2,
                formato.format(valor), va="center", fontsize=8, color=TEXTO_SEC)
    ax.set_xlim(0, limite * 1.12)
    salvar(fig, nome)


def obter():
    titulo_etapa("ETAPA 1 - OBTER (Obtain)")
    colunas = ["name", "address", "online_order", "book_table", "rate", "votes", "location",
               "rest_type", "dish_liked", "cuisines", "approx_cost(for two people)",
               "listed_in(type)", "listed_in(city)"]
    df = pd.read_csv(ARQUIVO_DADOS, usecols=colunas)
    registrar(f"Fonte: Kaggle - Zomato Bangalore Restaurants (arquivo {os.path.basename(ARQUIVO_DADOS)})")
    registrar(f"Registros carregados: {len(df):,}")
    registrar(f"Colunas utilizadas: {len(df.columns)}")
    registrar("Colunas descartadas na leitura: url, phone, reviews_list, menu_item (texto livre/identificadores)")
    return df


def limpar(df):
    titulo_etapa("ETAPA 2 - LIMPAR (Scrub)")
    df = df.rename(columns={
        "name": "nome", "address": "endereco", "online_order": "pedido_online",
        "book_table": "reserva_mesa", "rate": "nota", "votes": "votos", "location": "bairro",
        "rest_type": "tipo_restaurante", "dish_liked": "pratos_curtidos", "cuisines": "culinarias",
        "approx_cost(for two people)": "custo_dois", "listed_in(type)": "categoria_listagem",
        "listed_in(city)": "regiao",
    })
    registrar(f"Valores ausentes antes da limpeza:")
    for coluna, qtd in df.isna().sum().items():
        if qtd > 0:
            registrar(f"  {coluna:<22} {qtd:>7,} ({qtd / len(df):.1%})")

    df["nota"] = (df["nota"].astype(str).str.replace("/5", "", regex=False).str.strip()
                  .replace({"NEW": np.nan, "-": np.nan, "nan": np.nan}))
    df["nota"] = pd.to_numeric(df["nota"], errors="coerce")
    df["custo_dois"] = pd.to_numeric(df["custo_dois"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["pedido_online"] = (df["pedido_online"] == "Yes").astype(int)
    df["reserva_mesa"] = (df["reserva_mesa"] == "Yes").astype(int)

    antes = len(df)
    df = df.dropna(subset=["bairro", "tipo_restaurante", "culinarias", "custo_dois"])
    registrar(f"Removidos por ausencia de bairro/tipo/culinaria/custo: {antes - len(df):,}")

    df["tipo_principal"] = df["tipo_restaurante"].str.split(",").str[0].str.strip()
    df["culinaria_principal"] = df["culinarias"].str.split(",").str[0].str.strip()
    df["qtd_culinarias"] = df["culinarias"].str.split(",").apply(len)
    df["votos_log"] = np.log1p(df["votos"])
    df["faixa_preco"] = pd.cut(df["custo_dois"], bins=[0, 300, 600, 1000, 2000, np.inf],
                               labels=["Ate 300", "301-600", "601-1000", "1001-2000", "Acima de 2000"])

    restaurantes = df.drop_duplicates(subset=["nome", "endereco"]).copy()
    registrar(f"Registros apos limpeza (com listagens repetidas): {len(df):,}")
    registrar(f"Restaurantes unicos (nome + endereco): {len(restaurantes):,}")
    registrar(f"Restaurantes sem nota (NEW ou '-'): {restaurantes['nota'].isna().sum():,}")

    q1, q3 = restaurantes["custo_dois"].quantile([0.25, 0.75])
    limite_sup = q3 + 1.5 * (q3 - q1)
    registrar(f"Outliers de custo (IQR, acima de {limite_sup:.0f}): "
              f"{(restaurantes['custo_dois'] > limite_sup).sum():,} restaurantes "
              f"(mantidos: representam o segmento premium)")

    restaurantes.drop(columns=["endereco", "pratos_curtidos"]).to_csv(ARQUIVO_LIMPO, index=False)
    registrar(f"Base limpa exportada: dados/{os.path.basename(ARQUIVO_LIMPO)}")
    return df, restaurantes


def explorar(df, rest):
    titulo_etapa("ETAPA 3 - EXPLORAR (Explore)")
    avaliados = rest.dropna(subset=["nota"])

    desc = rest[["nota", "votos", "custo_dois", "qtd_culinarias"]].describe().T
    desc.to_csv(os.path.join(PASTA_RELATORIOS, "estatisticas_descritivas.csv"))
    registrar("Estatisticas descritivas (restaurantes unicos):")
    registrar(desc.round(2).to_string())
    registrar(f"Assimetria da nota: {avaliados['nota'].skew():.2f} | "
              f"Assimetria do custo: {rest['custo_dois'].skew():.2f} | "
              f"Assimetria dos votos: {rest['votos'].skew():.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(avaliados["nota"], bins=np.arange(1.75, 5.05, 0.1), color=AZUL, edgecolor=FUNDO, linewidth=1)
    ax.axvline(avaliados["nota"].mean(), color=TEXTO, linestyle="--", linewidth=1.5)
    ax.text(avaliados["nota"].mean() + 0.03, ax.get_ylim()[1] * 0.92,
            f"media {avaliados['nota'].mean():.2f}", color=TEXTO, fontsize=9)
    ax.set_title("Distribuicao das notas dos restaurantes")
    ax.set_xlabel("Nota (0 a 5)")
    ax.set_ylabel("Quantidade de restaurantes")
    salvar(fig, "01_distribuicao_notas.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(rest["custo_dois"].clip(upper=3000), bins=30, color=AZUL, edgecolor=FUNDO, linewidth=1)
    ax.axvline(rest["custo_dois"].median(), color=TEXTO, linestyle="--", linewidth=1.5)
    ax.text(rest["custo_dois"].median() + 30, ax.get_ylim()[1] * 0.92,
            f"mediana {rest['custo_dois'].median():.0f}", color=TEXTO, fontsize=9)
    ax.set_title("Distribuicao do custo para duas pessoas (limitado a 3000)")
    ax.set_xlabel("Custo aproximado para duas pessoas (INR)")
    ax.set_ylabel("Quantidade de restaurantes")
    salvar(fig, "02_distribuicao_custo.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, coluna, rotulo in zip(axes, ["pedido_online", "reserva_mesa"], ["Pedido online", "Reserva de mesa"]):
        contagem = rest[coluna].value_counts().reindex([1, 0])
        ax.bar(["Sim", "Nao"], contagem.values, color=[AZUL, CINZA], width=0.6)
        for i, v in enumerate(contagem.values):
            ax.text(i, v * 1.01, f"{v:,}\n({v / contagem.sum():.0%})", ha="center", va="bottom", fontsize=9, color=TEXTO_SEC)
        ax.set_title(rotulo)
        ax.set_ylim(0, contagem.max() * 1.2)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Quantidade de restaurantes")
    salvar(fig, "03_servicos_oferecidos.png")
    registrar(f"Oferecem pedido online: {rest['pedido_online'].mean():.1%} | "
              f"Oferecem reserva de mesa: {rest['reserva_mesa'].mean():.1%}")

    top_bairros = rest["bairro"].value_counts().head(15)
    barras_horizontais(top_bairros, "Top 15 bairros com mais restaurantes", "Quantidade de restaurantes",
                       "04_top_bairros.png")
    registrar(f"Bairro com mais restaurantes: {top_bairros.index[0]} ({top_bairros.iloc[0]:,})")

    top_tipos = rest["tipo_principal"].value_counts().head(12)
    barras_horizontais(top_tipos, "Tipos de estabelecimento mais comuns", "Quantidade de restaurantes",
                       "05_top_tipos.png")
    registrar(f"Tipo mais comum: {top_tipos.index[0]} ({top_tipos.iloc[0] / len(rest):.1%} dos restaurantes)")

    culinarias = rest["culinarias"].str.split(",").explode().str.strip().value_counts().head(15)
    barras_horizontais(culinarias, "Top 15 culinarias oferecidas", "Restaurantes que oferecem",
                       "06_top_culinarias.png")
    registrar(f"Culinaria mais oferecida: {culinarias.index[0]} ({culinarias.iloc[0] / len(rest):.1%} dos restaurantes)")

    redes = rest["nome"].value_counts().head(15)
    barras_horizontais(redes, "Top 15 redes com mais unidades", "Numero de unidades", "07_top_redes.png")
    registrar(f"Maior rede: {redes.index[0]} ({redes.iloc[0]} unidades)")

    pratos = df.drop_duplicates(subset=["nome", "endereco"])["pratos_curtidos"].dropna()
    pratos = pratos.str.split(",").explode().str.strip().value_counts().head(15)
    barras_horizontais(pratos, "Top 15 pratos mais curtidos pelos clientes", "Restaurantes onde o prato e destaque",
                       "08_top_pratos.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, coluna, rotulo in zip(axes, ["pedido_online", "reserva_mesa"], ["Pedido online", "Reserva de mesa"]):
        grupos = [avaliados.loc[avaliados[coluna] == 1, "nota"], avaliados.loc[avaliados[coluna] == 0, "nota"]]
        caixa = ax.boxplot(grupos, tick_labels=["Sim", "Nao"], patch_artist=True, widths=0.5,
                           medianprops={"color": TEXTO, "linewidth": 2},
                           flierprops={"marker": "o", "markersize": 3, "markerfacecolor": CINZA,
                                       "markeredgecolor": "none", "alpha": 0.4})
        for patch, cor in zip(caixa["boxes"], [AZUL, CINZA]):
            patch.set_facecolor(cor)
            patch.set_alpha(0.85)
            patch.set_edgecolor(FUNDO)
        ax.set_title(f"Nota x {rotulo}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Nota")
    salvar(fig, "09_nota_por_servico.png")
    for coluna, rotulo in [("pedido_online", "pedido online"), ("reserva_mesa", "reserva de mesa")]:
        medias = avaliados.groupby(coluna)["nota"].mean()
        registrar(f"Nota media com {rotulo}: {medias.get(1, np.nan):.2f} | sem: {medias.get(0, np.nan):.2f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(avaliados["custo_dois"], avaliados["nota"], s=10, color=AZUL, alpha=0.25, edgecolors="none")
    coef = np.polyfit(avaliados["custo_dois"], avaliados["nota"], 1)
    xs = np.linspace(avaliados["custo_dois"].min(), avaliados["custo_dois"].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color=LARANJA, linewidth=2, label="Tendencia linear")
    ax.set_title("Custo para dois x Nota")
    ax.set_xlabel("Custo aproximado para duas pessoas (INR)")
    ax.set_ylabel("Nota")
    ax.legend(loc="lower right")
    salvar(fig, "10_custo_vs_nota.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(avaliados["votos"] + 1, avaliados["nota"], s=10, color=AZUL, alpha=0.25, edgecolors="none")
    ax.set_xscale("log")
    ax.set_title("Popularidade (votos) x Nota")
    ax.set_xlabel("Numero de votos (escala log)")
    ax.set_ylabel("Nota")
    salvar(fig, "11_votos_vs_nota.png")

    por_categoria = df.dropna(subset=["nota"]).groupby("categoria_listagem").agg(
        nota_media=("nota", "mean"), custo_medio=("custo_dois", "mean"), qtd=("nome", "count"))
    barras_horizontais(por_categoria["nota_media"], "Nota media por categoria de listagem", "Nota media",
                       "12_nota_por_categoria.png", formato="{:.2f}")
    barras_horizontais(por_categoria["custo_medio"], "Custo medio por categoria de listagem", "Custo medio (INR)",
                       "13_custo_por_categoria.png")
    registrar("Resumo por categoria de listagem:")
    registrar(por_categoria.sort_values("nota_media", ascending=False).round(2).to_string())

    bairros_grandes = avaliados["bairro"].value_counts()
    bairros_grandes = bairros_grandes[bairros_grandes >= 50].index
    nota_bairro = avaliados[avaliados["bairro"].isin(bairros_grandes)].groupby("bairro")["nota"].mean()
    barras_horizontais(nota_bairro.sort_values(ascending=False).head(15),
                       "Top 15 bairros por nota media (min. 50 restaurantes)", "Nota media",
                       "14_melhores_bairros.png", formato="{:.2f}")
    custo_bairro = rest[rest["bairro"].isin(bairros_grandes)].groupby("bairro")["custo_dois"].mean()
    barras_horizontais(custo_bairro.sort_values(ascending=False).head(15),
                       "Top 15 bairros mais caros (min. 50 restaurantes)", "Custo medio para dois (INR)",
                       "15_bairros_mais_caros.png")
    registrar(f"Bairro com melhor nota media: {nota_bairro.idxmax()} ({nota_bairro.max():.2f})")
    registrar(f"Bairro mais caro: {custo_bairro.idxmax()} (INR {custo_bairro.max():.0f})")

    faixa = avaliados.groupby("faixa_preco", observed=True)["nota"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    barras = ax.bar(faixa.index.astype(str), faixa.values, color=AZUL, width=0.6)
    for barra, v in zip(barras, faixa.values):
        ax.text(barra.get_x() + barra.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9, color=TEXTO_SEC)
    ax.set_ylim(0, 5)
    ax.set_title("Nota media por faixa de preco")
    ax.set_xlabel("Custo para duas pessoas (INR)")
    ax.set_ylabel("Nota media")
    ax.grid(axis="x", visible=False)
    salvar(fig, "16_nota_por_faixa_preco.png")

    tipos_top = rest["tipo_principal"].value_counts().head(8).index
    tabela = (avaliados[avaliados["tipo_principal"].isin(tipos_top)]
              .pivot_table(index="tipo_principal", columns="faixa_preco", values="nota", aggfunc="mean", observed=True))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    imagem = ax.imshow(tabela.values, cmap=MAPA_SEQUENCIAL, aspect="auto", vmin=3.0, vmax=4.5)
    ax.set_xticks(range(len(tabela.columns)), tabela.columns.astype(str))
    ax.set_yticks(range(len(tabela.index)), tabela.index)
    ax.grid(False)
    for i in range(tabela.shape[0]):
        for j in range(tabela.shape[1]):
            valor = tabela.values[i, j]
            if not np.isnan(valor):
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if valor > 3.9 else TEXTO)
    fig.colorbar(imagem, ax=ax, label="Nota media")
    ax.set_title("Nota media por tipo de estabelecimento e faixa de preco")
    salvar(fig, "17_mapa_tipo_faixa.png")

    numericas = avaliados[["nota", "votos_log", "custo_dois", "pedido_online", "reserva_mesa", "qtd_culinarias"]]
    correlacao = numericas.corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    imagem = ax.imshow(correlacao.values, cmap=MAPA_DIVERGENTE, vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlacao)), correlacao.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(correlacao)), correlacao.columns)
    ax.grid(False)
    for i in range(len(correlacao)):
        for j in range(len(correlacao)):
            ax.text(j, i, f"{correlacao.values[i, j]:.2f}", ha="center", va="center", fontsize=9, color=TEXTO)
    fig.colorbar(imagem, ax=ax, label="Correlacao de Pearson")
    ax.set_title("Matriz de correlacao")
    salvar(fig, "18_matriz_correlacao.png")
    registrar("Correlacao com a nota:")
    registrar(correlacao["nota"].drop("nota").sort_values(ascending=False).round(3).to_string())


def montar_atributos(base):
    top_bairros = base["bairro"].value_counts().head(20).index
    top_tipos = base["tipo_principal"].value_counts().head(10).index
    top_culinarias = base["culinaria_principal"].value_counts().head(12).index
    tabela = base[["pedido_online", "reserva_mesa", "votos_log", "custo_dois", "qtd_culinarias"]].copy()
    categoricas = pd.DataFrame({
        "bairro": base["bairro"].where(base["bairro"].isin(top_bairros), "Outros"),
        "tipo": base["tipo_principal"].where(base["tipo_principal"].isin(top_tipos), "Outros"),
        "culinaria": base["culinaria_principal"].where(base["culinaria_principal"].isin(top_culinarias), "Outros"),
    })
    return pd.concat([tabela, pd.get_dummies(categoricas, dtype=int)], axis=1)


def modelar_clusterizacao(avaliados):
    registrar("")
    registrar("-- 4.1 Clusterizacao (K-Means) --")
    variaveis = ["nota", "custo_dois", "votos_log", "pedido_online", "reserva_mesa"]
    escalador = StandardScaler()
    matriz = escalador.fit_transform(avaliados[variaveis])

    amostra = np.random.default_rng(SEMENTE).choice(len(matriz), size=min(5000, len(matriz)), replace=False)
    ks = range(2, 9)
    inercias, silhuetas = [], []
    for k in ks:
        modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit(matriz)
        inercias.append(modelo.inertia_)
        silhuetas.append(silhouette_score(matriz[amostra], modelo.labels_[amostra]))
        registrar(f"  k={k}  inercia={modelo.inertia_:,.0f}  silhueta={silhuetas[-1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(list(ks), inercias, color=AZUL, linewidth=2, marker="o", markersize=8)
    axes[0].set_title("Metodo do cotovelo")
    axes[0].set_xlabel("Numero de clusters (k)")
    axes[0].set_ylabel("Inercia")
    axes[1].plot(list(ks), silhuetas, color=AZUL, linewidth=2, marker="o", markersize=8)
    axes[1].set_title("Coeficiente de silhueta")
    axes[1].set_xlabel("Numero de clusters (k)")
    axes[1].set_ylabel("Silhueta media")
    salvar(fig, "19_escolha_k.png")

    k_final = 3
    registrar(f"k escolhido: {k_final} (equilibrio entre cotovelo, silhueta e interpretabilidade de negocio)")
    modelo = KMeans(n_clusters=k_final, n_init=10, random_state=SEMENTE).fit(matriz)
    avaliados = avaliados.copy()
    avaliados["cluster"] = modelo.labels_

    perfil = avaliados.groupby("cluster").agg(
        restaurantes=("nome", "count"), nota_media=("nota", "mean"), custo_medio=("custo_dois", "mean"),
        votos_medianos=("votos", "median"), pedido_online=("pedido_online", "mean"),
        reserva_mesa=("reserva_mesa", "mean"))
    ordem = perfil.sort_values("custo_medio").index
    nomes = {}
    for posicao, cluster in enumerate(ordem):
        linha = perfil.loc[cluster]
        if linha["reserva_mesa"] > 0.5:
            nomes[cluster] = "Premium com reserva"
        elif linha["pedido_online"] > 0.5:
            nomes[cluster] = "Populares com delivery online"
        else:
            nomes[cluster] = "Tradicionais sem delivery online"
    if len(set(nomes.values())) < k_final:
        nomes = {cluster: f"Segmento {i + 1}" for i, cluster in enumerate(ordem)}
    perfil["segmento"] = perfil.index.map(nomes)
    perfil = perfil.loc[ordem]
    perfil.to_csv(os.path.join(PASTA_RELATORIOS, "perfil_clusters.csv"))
    registrar("Perfil dos clusters:")
    registrar(perfil.round(2).to_string())

    componentes = PCA(n_components=2, random_state=SEMENTE).fit_transform(matriz)
    fig, ax = plt.subplots(figsize=(9, 6))
    for cor, cluster in zip(CORES_CATEGORICAS, ordem):
        mascara = avaliados["cluster"].values == cluster
        ax.scatter(componentes[mascara, 0], componentes[mascara, 1], s=12, color=cor, alpha=0.45,
                   edgecolors="none", label=f"{nomes[cluster]} ({mascara.sum():,})")
    ax.set_title("Segmentos de restaurantes (projecao PCA)")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend(loc="best", markerscale=2)
    salvar(fig, "20_clusters_pca.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    rotulos = [nomes[c] for c in ordem]
    for ax, coluna, titulo, formato in zip(axes, ["nota_media", "custo_medio", "votos_medianos"],
                                           ["Nota media", "Custo medio (INR)", "Votos (mediana)"],
                                           ["{:.2f}", "{:.0f}", "{:.0f}"]):
        valores = perfil[coluna].values
        barras = ax.bar(range(len(rotulos)), valores, color=CORES_CATEGORICAS[:len(rotulos)], width=0.6)
        for barra, v in zip(barras, valores):
            ax.text(barra.get_x() + barra.get_width() / 2, v * 1.01, formato.format(v), ha="center",
                    va="bottom", fontsize=9, color=TEXTO_SEC)
        ax.set_xticks(range(len(rotulos)), [r.replace(" ", "\n", 1) for r in rotulos], fontsize=8)
        ax.set_title(titulo)
        ax.set_ylim(0, valores.max() * 1.18)
        ax.grid(axis="x", visible=False)
    salvar(fig, "21_perfil_clusters.png")
    return perfil


def modelar_classificacao(avaliados):
    registrar("")
    registrar("-- 4.2 Classificacao: restaurante bem avaliado (nota >= 4.0) --")
    alvo = (avaliados["nota"] >= 4.0).astype(int)
    registrar(f"Proporcao de bem avaliados: {alvo.mean():.1%}")
    atributos = montar_atributos(avaliados)
    x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, alvo, test_size=0.25,
                                                            random_state=SEMENTE, stratify=alvo)
    escalador = StandardScaler().fit(x_treino)
    modelos = {
        "Regressao Logistica": (LogisticRegression(max_iter=2000, class_weight="balanced"), True),
        "Arvore de Decisao": (DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEMENTE), False),
        "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=14, class_weight="balanced",
                                                 n_jobs=-1, random_state=SEMENTE), False),
    }
    resultados, curvas, melhor = [], {}, None
    for nome, (modelo, escalar) in modelos.items():
        treino = escalador.transform(x_treino) if escalar else x_treino
        teste = escalador.transform(x_teste) if escalar else x_teste
        modelo.fit(treino, y_treino)
        previsto = modelo.predict(teste)
        probabilidade = modelo.predict_proba(teste)[:, 1]
        linha = {
            "modelo": nome,
            "acuracia": accuracy_score(y_teste, previsto),
            "precisao": precision_score(y_teste, previsto),
            "recall": recall_score(y_teste, previsto),
            "f1": f1_score(y_teste, previsto),
            "auc": roc_auc_score(y_teste, probabilidade),
        }
        resultados.append(linha)
        curvas[nome] = roc_curve(y_teste, probabilidade)
        if melhor is None or linha["f1"] > melhor[1]["f1"]:
            melhor = (modelo, linha, previsto)
    tabela = pd.DataFrame(resultados).set_index("modelo")
    tabela.to_csv(os.path.join(PASTA_RELATORIOS, "metricas_classificacao.csv"))
    registrar(tabela.round(3).to_string())
    registrar(f"Melhor modelo (F1): {melhor[1]['modelo']}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for cor, (nome, (fpr, tpr, _)) in zip(CORES_CATEGORICAS, curvas.items()):
        ax.plot(fpr, tpr, color=cor, linewidth=2, label=f"{nome} (AUC {tabela.loc[nome, 'auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=CINZA, linestyle="--", linewidth=1)
    ax.set_title("Curva ROC - modelos de classificacao")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.legend(loc="lower right")
    salvar(fig, "22_curva_roc.png")

    matriz = confusion_matrix(y_teste, melhor[2])
    fig, ax = plt.subplots(figsize=(6, 5))
    imagem = ax.imshow(matriz, cmap=MAPA_SEQUENCIAL)
    ax.set_xticks([0, 1], ["Nota < 4.0", "Nota >= 4.0"])
    ax.set_yticks([0, 1], ["Nota < 4.0", "Nota >= 4.0"])
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matriz[i, j]:,}", ha="center", va="center", fontsize=12,
                    color="white" if matriz[i, j] > matriz.max() / 2 else TEXTO)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusao - {melhor[1]['modelo']}")
    fig.colorbar(imagem, ax=ax)
    salvar(fig, "23_matriz_confusao.png")

    floresta = modelos["Random Forest"][0]
    importancia = pd.Series(floresta.feature_importances_, index=atributos.columns).sort_values(ascending=False).head(15)
    barras_horizontais(importancia, "Variaveis mais importantes - Random Forest (classificacao)", "Importancia",
                       "24_importancia_classificacao.png", formato="{:.3f}")
    registrar("Variaveis mais importantes (classificacao):")
    registrar(importancia.head(8).round(3).to_string())
    return tabela, importancia


def modelar_regressao(avaliados):
    registrar("")
    registrar("-- 4.3 Regressao: previsao da nota do restaurante --")
    atributos = montar_atributos(avaliados)
    alvo = avaliados["nota"]
    x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, alvo, test_size=0.25, random_state=SEMENTE)
    modelos = {
        "Regressao Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=16, min_samples_leaf=2,
                                               n_jobs=-1, random_state=SEMENTE),
    }
    resultados, previsoes = [], {}
    for nome, modelo in modelos.items():
        modelo.fit(x_treino, y_treino)
        previsto = modelo.predict(x_teste)
        previsoes[nome] = previsto
        resultados.append({
            "modelo": nome,
            "MAE": mean_absolute_error(y_teste, previsto),
            "RMSE": np.sqrt(mean_squared_error(y_teste, previsto)),
            "R2": r2_score(y_teste, previsto),
        })
    base_ingenua = np.full(len(y_teste), y_treino.mean())
    resultados.append({"modelo": "Baseline (media)", "MAE": mean_absolute_error(y_teste, base_ingenua),
                       "RMSE": np.sqrt(mean_squared_error(y_teste, base_ingenua)), "R2": r2_score(y_teste, base_ingenua)})
    tabela = pd.DataFrame(resultados).set_index("modelo")
    tabela.to_csv(os.path.join(PASTA_RELATORIOS, "metricas_regressao.csv"))
    registrar(tabela.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, (nome, previsto) in zip(axes, previsoes.items()):
        ax.scatter(y_teste, previsto, s=10, color=AZUL, alpha=0.3, edgecolors="none")
        ax.plot([1.8, 5], [1.8, 5], color=LARANJA, linewidth=2, label="Previsao perfeita")
        ax.set_title(f"{nome} (R2 = {tabela.loc[nome, 'R2']:.3f})")
        ax.set_xlabel("Nota real")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Nota prevista")
    salvar(fig, "25_real_vs_previsto.png")

    residuos = y_teste - previsoes["Random Forest"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuos, bins=40, color=AZUL, edgecolor=FUNDO, linewidth=1)
    ax.axvline(0, color=TEXTO, linestyle="--", linewidth=1.5)
    ax.set_title("Distribuicao dos residuos - Random Forest")
    ax.set_xlabel("Residuo (nota real - nota prevista)")
    ax.set_ylabel("Frequencia")
    salvar(fig, "26_residuos_regressao.png")

    importancia = pd.Series(modelos["Random Forest"].feature_importances_,
                            index=atributos.columns).sort_values(ascending=False).head(15)
    barras_horizontais(importancia, "Variaveis mais importantes - Random Forest (regressao)", "Importancia",
                       "27_importancia_regressao.png", formato="{:.3f}")
    coeficientes = pd.Series(modelos["Regressao Linear"].coef_, index=atributos.columns)
    registrar("Coeficientes da regressao linear (variaveis principais):")
    registrar(coeficientes[["pedido_online", "reserva_mesa", "votos_log", "custo_dois", "qtd_culinarias"]]
              .round(4).to_string())
    return tabela


def modelar(rest):
    titulo_etapa("ETAPA 4 - MODELAR (Model)")
    avaliados = rest.dropna(subset=["nota"]).reset_index(drop=True)
    registrar(f"Base de modelagem: {len(avaliados):,} restaurantes unicos com nota")
    perfil = modelar_clusterizacao(avaliados)
    metricas_classe, importancia = modelar_classificacao(avaliados)
    metricas_reg = modelar_regressao(avaliados)
    return perfil, metricas_classe, importancia, metricas_reg


def interpretar(df, rest, perfil, metricas_classe, importancia, metricas_reg):
    titulo_etapa("ETAPA 5 - INTERPRETAR (iNterpret) - Relatorio gerencial")
    avaliados = rest.dropna(subset=["nota"])
    online = avaliados.groupby("pedido_online")["nota"].mean()
    reserva = avaliados.groupby("reserva_mesa")["nota"].mean()
    correlacao_custo = avaliados[["custo_dois", "nota"]].corr().iloc[0, 1]
    correlacao_votos = avaliados[["votos_log", "nota"]].corr().iloc[0, 1]
    melhor_classe = metricas_classe["f1"].idxmax()
    melhor_reg = metricas_reg.drop("Baseline (media)")["R2"].idxmax()
    bairros = rest["bairro"].value_counts()
    saturacao = bairros.head(5).sum() / len(rest)

    achados = [
        f"1. Mercado: {len(rest):,} restaurantes unicos em Bangalore; a nota media e {avaliados['nota'].mean():.2f} "
        f"e a mediana de custo para dois e INR {rest['custo_dois'].median():.0f}.",
        f"2. Concentracao: os 5 maiores bairros concentram {saturacao:.1%} dos restaurantes "
        f"(lider: {bairros.index[0]}), indicando alta concorrencia nessas regioes.",
        f"3. Reserva de mesa: quem oferece tem nota media {reserva.get(1):.2f} contra {reserva.get(0):.2f}; "
        f"o servico esta associado a experiencias mais completas (e mais caras).",
        f"4. Pedido online: nota media {online.get(1):.2f} com o servico vs {online.get(0):.2f} sem; "
        f"{rest['pedido_online'].mean():.0%} dos restaurantes ja vendem online.",
        f"5. Preco x qualidade: correlacao custo-nota de {correlacao_custo:.2f} (moderada); preco alto ajuda, "
        f"mas nao garante boa avaliacao.",
        f"6. Popularidade: correlacao entre log(votos) e nota de {correlacao_votos:.2f}; engajamento do cliente "
        f"e o sinal mais forte de qualidade percebida.",
        f"7. Segmentacao: o K-Means encontrou {len(perfil)} perfis: "
        + "; ".join(f"{linha.segmento} ({int(linha.restaurantes):,} rest., nota {linha.nota_media:.2f}, "
                    f"INR {linha.custo_medio:.0f})" for linha in perfil.itertuples()) + ".",
        f"8. Classificacao: o melhor modelo ({melhor_classe}) identifica restaurantes com nota >= 4.0 com "
        f"F1 = {metricas_classe.loc[melhor_classe, 'f1']:.3f} e AUC = {metricas_classe.loc[melhor_classe, 'auc']:.3f}. "
        f"Variavel mais relevante: {importancia.index[0]}.",
        f"9. Regressao: o melhor modelo ({melhor_reg}) preve a nota com MAE = {metricas_reg.loc[melhor_reg, 'MAE']:.3f} "
        f"e R2 = {metricas_reg.loc[melhor_reg, 'R2']:.3f}, contra MAE = "
        f"{metricas_reg.loc['Baseline (media)', 'MAE']:.3f} do baseline pela media.",
    ]
    recomendacoes = [
        "- Priorizar estrategias de engajamento (pedir avaliacoes, programas de fidelidade): votos sao o principal "
        "preditor de nota.",
        "- Novos negocios devem evitar bairros saturados ou entrar neles com diferencial claro de culinaria/preco.",
        "- Oferecer reserva de mesa em restaurantes de ticket medio/alto, onde o servico se associa a notas maiores.",
        "- Manter presenca no delivery online: e padrao de mercado e o publico ja espera o canal.",
        "- Usar o modelo de classificacao para triagem de parceiros/restaurantes com potencial de boa avaliacao, "
        "e os clusters para campanhas de marketing segmentadas.",
    ]
    registrar("Principais achados:")
    for linha in achados:
        registrar(linha)
    registrar("")
    registrar("Recomendacoes para tomada de decisao:")
    for linha in recomendacoes:
        registrar(linha)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    indicadores = metricas_classe[["acuracia", "precisao", "recall", "f1", "auc"]]
    largura = 0.25
    posicoes = np.arange(len(indicadores.columns))
    for i, (cor, (nome, linha)) in enumerate(zip(CORES_CATEGORICAS, indicadores.iterrows())):
        ax.bar(posicoes + (i - 1) * largura, linha.values, width=largura - 0.02, color=cor, label=nome)
    ax.set_xticks(posicoes, ["Acuracia", "Precisao", "Recall", "F1", "AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparativo dos modelos de classificacao")
    ax.set_ylabel("Valor da metrica")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    salvar(fig, "28_comparativo_classificacao.png")


def main():
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    os.makedirs(PASTA_RELATORIOS, exist_ok=True)
    registrar("PROJETO DE ANALISE DE DADOS - RESTAURANTES ZOMATO (BANGALORE)")
    registrar("Processo de ciencia de dados OSEMN: Obter, Limpar, Explorar, Modelar, Interpretar")
    bruto = obter()
    df, restaurantes = limpar(bruto)
    explorar(df, restaurantes)
    perfil, metricas_classe, importancia, metricas_reg = modelar(restaurantes)
    interpretar(df, restaurantes, perfil, metricas_classe, importancia, metricas_reg)
    with open(os.path.join(PASTA_RELATORIOS, "relatorio_gerencial.txt"), "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(relatorio))
    print("\nRelatorio salvo em relatorios/relatorio_gerencial.txt")


if __name__ == "__main__":
    main()
