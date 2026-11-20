import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, r2_score,
)

BASE = os.path.dirname(os.path.abspath(__file__))
ARQUIVO_DADOS = os.path.join(BASE, "dados", "zomato.csv")
ARQUIVO_LIMPO = os.path.join(BASE, "dados", "zomato_limpo.csv")
PASTA_IMAGENS = os.path.join(BASE, "imagens")
PASTA_RELATORIOS = os.path.join(BASE, "relatorios")
SEMENTE = 42

AZUL = "#2a78d6"
LARANJA = "#eb6834"
VERDE_AGUA = "#1baf7a"
CINZA = "#8a8984"
TEXTO = "#0b0b0b"
TEXTO_SEC = "#52514e"
FUNDO = "#fcfcfb"
CORES = [AZUL, LARANJA, VERDE_AGUA]
ATRIBUTOS = ["pedido_online", "reserva_mesa", "votos_log", "custo_dois", "qtd_culinarias"]

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
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "axes.axisbelow": True,
    "xtick.color": TEXTO_SEC,
    "ytick.color": TEXTO_SEC,
    "font.size": 10,
    "legend.frameon": False,
    "figure.dpi": 110,
})

relatorio = []
contador_imagens = [0]


def registrar(texto=""):
    print(texto)
    relatorio.append(texto)


def titulo_etapa(nome):
    registrar("")
    registrar("=" * 78)
    registrar(nome)
    registrar("=" * 78)


def preparar_pastas():
    os.makedirs(PASTA_IMAGENS, exist_ok=True)
    os.makedirs(PASTA_RELATORIOS, exist_ok=True)
    for arquivo in os.listdir(PASTA_IMAGENS):
        if arquivo.lower().endswith(".png"):
            os.remove(os.path.join(PASTA_IMAGENS, arquivo))


def salvar(fig, nome):
    contador_imagens[0] += 1
    arquivo = f"{contador_imagens[0]:02d}_{nome}.png"
    for ax in fig.axes:
        if ax.get_legend_handles_labels()[0] and ax.get_legend() is None:
            ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_IMAGENS, arquivo), bbox_inches="tight")
    registrar(f"  [grafico] imagens/{arquivo}")
    fig.canvas.manager.set_window_title(arquivo)
    plt.show()
    plt.close(fig)


def histograma(valores, bins, titulo, xlabel, nome):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(valores, bins=bins, color=AZUL, edgecolor=FUNDO, label="Restaurantes por intervalo")
    ax.axvline(valores.median(), color=TEXTO, linestyle="--", linewidth=1.5,
               label=f"Mediana ({valores.median():.1f})")
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Quantidade de restaurantes")
    salvar(fig, nome)


def barras(serie, titulo, xlabel, nome, formato="{:.0f}", horizontal=True):
    fig, ax = plt.subplots(figsize=(9, 5))
    if horizontal:
        serie = serie.sort_values()
        retangulos = ax.barh(serie.index.astype(str), serie.values, color=AZUL, height=0.7, label=xlabel)
        ax.bar_label(retangulos, [formato.format(v) for v in serie.values], padding=3, fontsize=8, color=TEXTO_SEC)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    else:
        retangulos = ax.bar(serie.index.astype(str), serie.values, color=AZUL, width=0.6, label=xlabel)
        ax.set_ylim(0, serie.max() * 1.2)
        ax.bar_label(retangulos, [formato.format(v) for v in serie.values], padding=3, fontsize=9, color=TEXTO_SEC)
        ax.set_ylabel(xlabel)
        ax.grid(axis="x", visible=False)
    ax.set_title(titulo)
    salvar(fig, nome)


def obter():
    titulo_etapa("ETAPA 1 - OBTER (Obtain)")
    colunas = ["name", "address", "online_order", "book_table", "rate", "votes", "location",
               "rest_type", "cuisines", "approx_cost(for two people)"]
    df = pd.read_csv(ARQUIVO_DADOS, usecols=colunas)
    registrar("Fonte: Kaggle - Zomato Bangalore Restaurants (dados/zomato.csv)")
    registrar(f"Registros carregados: {len(df):,}")
    registrar(f"Colunas utilizadas: {len(df.columns)}")
    return df


def limpar(df):
    titulo_etapa("ETAPA 2 - LIMPAR (Scrub)")
    df = df.rename(columns={
        "name": "nome", "address": "endereco", "online_order": "pedido_online",
        "book_table": "reserva_mesa", "rate": "nota", "votes": "votos", "location": "bairro",
        "rest_type": "tipo_restaurante", "cuisines": "culinarias",
        "approx_cost(for two people)": "custo_dois",
    })
    registrar("Valores ausentes antes da limpeza:")
    for coluna, qtd in df.isna().sum().items():
        if qtd > 0:
            registrar(f"  {coluna:<18} {qtd:>7,} ({qtd / len(df):.1%})")

    for coluna in df.select_dtypes(include=["object", "string"]).columns:
        df[coluna] = df[coluna].astype("string").str.strip().replace("", pd.NA)

    df["nota"] = df["nota"].str.replace("/5", "", regex=False).str.strip()
    df["nota"] = pd.to_numeric(df["nota"], errors="coerce").astype(float)
    df["custo_dois"] = pd.to_numeric(df["custo_dois"].str.replace(",", "", regex=False), errors="coerce").astype(float)
    df["pedido_online"] = (df["pedido_online"] == "Yes").astype(int)
    df["reserva_mesa"] = (df["reserva_mesa"] == "Yes").astype(int)
    registrar("Conversao de tipos: nota '4.1/5' -> 4.1 | custo '1,200' -> 1200 | Yes/No -> 1/0")

    antes = len(df)
    df = df.dropna(subset=["nome", "bairro", "tipo_restaurante", "culinarias", "custo_dois"])
    registrar(f"Removidos por dados obrigatorios ausentes: {antes - len(df):,}")

    antes = len(df)
    df = df.drop_duplicates(subset=["nome", "endereco"])
    registrar(f"Removidas listagens repetidas do mesmo restaurante: {antes - len(df):,}")

    df["tipo_principal"] = df["tipo_restaurante"].str.split(",").str[0].str.strip()
    df["qtd_culinarias"] = df["culinarias"].str.split(",").str.len()
    df["votos_log"] = np.log1p(df["votos"])
    df["faixa_preco"] = pd.cut(df["custo_dois"], bins=[0, 300, 600, 1000, 2000, np.inf],
                               labels=["Ate 300", "301-600", "601-1000", "1001-2000", "Acima de 2000"])

    registrar(f"Restaurantes unicos apos limpeza: {len(df):,}")
    registrar(f"Restaurantes sem nota (novos ou sem avaliacao): {df['nota'].isna().sum():,}")
    registrar("Amostra da base tratada:")
    registrar(df[["nome", "bairro", "tipo_principal", "nota", "votos", "custo_dois"]].head(5).to_string(index=False))

    df.drop(columns=["endereco"]).to_csv(ARQUIVO_LIMPO, index=False)
    registrar("Base limpa exportada: dados/zomato_limpo.csv")
    return df.reset_index(drop=True)


def explorar(rest):
    titulo_etapa("ETAPA 3 - EXPLORAR (Explore)")
    avaliados = rest.dropna(subset=["nota"])

    desc = rest[["nota", "votos", "custo_dois"]].describe().T
    desc.to_csv(os.path.join(PASTA_RELATORIOS, "estatisticas_descritivas.csv"))
    registrar("Estatisticas descritivas:")
    registrar(desc.round(2).to_string())

    histograma(avaliados["nota"], np.arange(1.75, 5.05, 0.1), "Distribuicao das notas", "Nota (0 a 5)",
               "distribuicao_notas")
    histograma(rest["custo_dois"].clip(upper=3000), 30, "Distribuicao do custo para duas pessoas",
               "Custo para duas pessoas (INR, limitado a 3000)", "distribuicao_custo")

    for coluna, rotulo in [("pedido_online", "pedido online"), ("reserva_mesa", "reserva de mesa")]:
        percentual = rest[coluna].map({1: "Sim", 0: "Nao"}).value_counts(normalize=True).reindex(["Sim", "Nao"]) * 100
        barras(percentual, f"Restaurantes com {rotulo}", "% dos restaurantes", f"servico_{coluna}",
               formato="{:.1f}%", horizontal=False)
        medias = avaliados.groupby(coluna)["nota"].mean()
        registrar(f"Com {rotulo}: {rest[coluna].mean():.1%} | nota media com: {medias[1]:.2f} | sem: {medias[0]:.2f}")

    top_bairros = rest["bairro"].value_counts().head(10)
    barras(top_bairros, "Top 10 bairros com mais restaurantes", "Quantidade de restaurantes", "top_bairros")
    registrar(f"Bairro com mais restaurantes: {top_bairros.index[0]} ({top_bairros.iloc[0]:,})")

    culinarias = rest["culinarias"].str.split(",").explode().str.strip().value_counts().head(10)
    barras(culinarias, "Top 10 culinarias oferecidas", "Restaurantes que oferecem", "top_culinarias")
    registrar(f"Culinaria mais oferecida: {culinarias.index[0]}")

    faixa = avaliados.groupby("faixa_preco", observed=True)["nota"].mean()
    barras(faixa, "Nota media por faixa de preco", "Nota media", "nota_por_faixa_preco",
           formato="{:.2f}", horizontal=False)
    registrar("Nota media por faixa de preco:")
    registrar(faixa.round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(avaliados["custo_dois"], avaliados["nota"], s=10, color=AZUL, alpha=0.25, edgecolors="none",
               label="Restaurante avaliado")
    coef = np.polyfit(avaliados["custo_dois"], avaliados["nota"], 1)
    xs = np.linspace(avaliados["custo_dois"].min(), avaliados["custo_dois"].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color=LARANJA, linewidth=2, label="Tendencia linear")
    ax.set_title("Custo para dois x Nota")
    ax.set_xlabel("Custo para duas pessoas (INR)")
    ax.set_ylabel("Nota")
    salvar(fig, "custo_vs_nota")

    correlacao = avaliados[["nota"] + ATRIBUTOS].corr()["nota"].drop("nota").sort_values(ascending=False)
    registrar("Correlacao das variaveis com a nota:")
    registrar(correlacao.round(3).to_string())
    return correlacao


def clusterizar(avaliados):
    registrar("")
    registrar("-- 4.1 Clusterizacao (K-Means) --")
    variaveis = ["nota", "custo_dois", "votos_log", "pedido_online", "reserva_mesa"]
    matriz = StandardScaler().fit_transform(avaliados[variaveis])

    k = 3
    ks = list(range(2, 9))
    inercias = [KMeans(n_clusters=n, n_init=10, random_state=SEMENTE).fit(matriz).inertia_ for n in ks]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, inercias, color=AZUL, linewidth=2, marker="o", markersize=8, label="Inercia por k")
    ax.scatter([k], [inercias[ks.index(k)]], s=160, color=LARANJA, zorder=3, label=f"k escolhido ({k})")
    ax.set_title("Metodo do cotovelo - escolha do numero de clusters")
    ax.set_xlabel("Numero de clusters (k)")
    ax.set_ylabel("Inercia")
    salvar(fig, "metodo_cotovelo")

    avaliados = avaliados.copy()
    avaliados["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit_predict(matriz)
    perfil = avaliados.groupby("cluster").agg(
        restaurantes=("nome", "count"), nota_media=("nota", "mean"), custo_medio=("custo_dois", "mean"),
        votos_medianos=("votos", "median"), pedido_online=("pedido_online", "mean"),
        reserva_mesa=("reserva_mesa", "mean")).sort_values("custo_medio")
    perfil["segmento"] = [f"Segmento {i + 1}" for i in range(k)]
    perfil.to_csv(os.path.join(PASTA_RELATORIOS, "perfil_clusters.csv"))
    registrar(f"k escolhido: {k}")
    registrar("Perfil dos clusters (ordenado por custo medio):")
    registrar(perfil.round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cor, (cluster, linha) in zip(CORES, perfil.iterrows()):
        grupo = avaliados[avaliados["cluster"] == cluster]
        ax.scatter(grupo["custo_dois"], grupo["nota"], s=10, color=cor, alpha=0.4, edgecolors="none",
                   label=f"{linha['segmento']} ({len(grupo):,})")
    ax.set_title("Segmentos de restaurantes (K-Means)")
    ax.set_xlabel("Custo para duas pessoas (INR)")
    ax.set_ylabel("Nota")
    ax.legend(markerscale=2)
    salvar(fig, "clusters")
    return perfil


def classificar(avaliados):
    registrar("")
    registrar("-- 4.2 Classificacao (Regressao Logistica): restaurante bem avaliado (nota >= 4.0) --")
    alvo = (avaliados["nota"] >= 4.0).astype(int)
    registrar(f"Proporcao de bem avaliados: {alvo.mean():.1%}")
    x_treino, x_teste, y_treino, y_teste = train_test_split(avaliados[ATRIBUTOS], alvo, test_size=0.25,
                                                            random_state=SEMENTE, stratify=alvo)
    escalador = StandardScaler().fit(x_treino)
    modelo = LogisticRegression(class_weight="balanced").fit(escalador.transform(x_treino), y_treino)
    previsto = modelo.predict(escalador.transform(x_teste))
    metricas = pd.Series({
        "acuracia": accuracy_score(y_teste, previsto),
        "precisao": precision_score(y_teste, previsto),
        "recall": recall_score(y_teste, previsto),
        "f1": f1_score(y_teste, previsto),
    })
    metricas.to_frame("valor").to_csv(os.path.join(PASTA_RELATORIOS, "metricas_classificacao.csv"))
    registrar(metricas.round(3).to_string())

    matriz = confusion_matrix(y_teste, previsto)
    fig, ax = plt.subplots(figsize=(6, 5))
    imagem = ax.imshow(matriz, cmap="Blues")
    fig.colorbar(imagem, ax=ax, label="Quantidade de restaurantes")
    rotulos = ["Nota < 4.0", "Nota >= 4.0"]
    ax.set_xticks([0, 1], rotulos)
    ax.set_yticks([0, 1], rotulos)
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matriz[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if matriz[i, j] > matriz.max() / 2 else TEXTO)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusao - Regressao Logistica")
    salvar(fig, "matriz_confusao")
    return metricas


def regredir(avaliados):
    registrar("")
    registrar("-- 4.3 Regressao (Regressao Linear): previsao da nota --")
    x_treino, x_teste, y_treino, y_teste = train_test_split(avaliados[ATRIBUTOS], avaliados["nota"],
                                                            test_size=0.25, random_state=SEMENTE)
    modelo = LinearRegression().fit(x_treino, y_treino)
    previsto = modelo.predict(x_teste)
    metricas = pd.Series({
        "MAE": mean_absolute_error(y_teste, previsto),
        "R2": r2_score(y_teste, previsto),
        "MAE baseline (media)": mean_absolute_error(y_teste, np.full(len(y_teste), y_treino.mean())),
    })
    metricas.to_frame("valor").to_csv(os.path.join(PASTA_RELATORIOS, "metricas_regressao.csv"))
    registrar(metricas.round(3).to_string())
    registrar("Coeficientes:")
    registrar(pd.Series(modelo.coef_, index=ATRIBUTOS).round(4).to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_teste, previsto, s=10, color=AZUL, alpha=0.3, edgecolors="none", label="Restaurante (teste)")
    ax.plot([1.8, 5], [1.8, 5], color=LARANJA, linewidth=2, label="Previsao perfeita")
    ax.set_title(f"Nota real x prevista (R2 = {metricas['R2']:.3f})")
    ax.set_xlabel("Nota real")
    ax.set_ylabel("Nota prevista")
    ax.legend(loc="upper left")
    salvar(fig, "real_vs_previsto")
    return metricas


def modelar(rest):
    titulo_etapa("ETAPA 4 - MODELAR (Model)")
    avaliados = rest.dropna(subset=["nota"]).reset_index(drop=True)
    registrar(f"Base de modelagem: {len(avaliados):,} restaurantes com nota")
    return clusterizar(avaliados), classificar(avaliados), regredir(avaliados)


def interpretar(rest, correlacao, perfil, metricas_classe, metricas_reg):
    titulo_etapa("ETAPA 5 - INTERPRETAR (iNterpret) - Relatorio gerencial")
    avaliados = rest.dropna(subset=["nota"])
    reserva = avaliados.groupby("reserva_mesa")["nota"].mean()
    online = avaliados.groupby("pedido_online")["nota"].mean()
    bairros = rest["bairro"].value_counts()

    registrar("Principais achados:")
    registrar(f"1. Mercado: {len(rest):,} restaurantes; nota media {avaliados['nota'].mean():.2f} e custo mediano "
              f"para dois de INR {rest['custo_dois'].median():.0f}.")
    registrar(f"2. Concentracao: os 5 maiores bairros reunem {bairros.head(5).sum() / len(rest):.1%} dos restaurantes "
              f"(lider: {bairros.index[0]}).")
    registrar(f"3. Reserva de mesa: nota media {reserva[1]:.2f} com o servico contra {reserva[0]:.2f} sem.")
    registrar(f"4. Pedido online: nota media {online[1]:.2f} com o servico contra {online[0]:.2f} sem.")
    registrar(f"5. Variavel mais associada a nota: {correlacao.index[0]} (correlacao {correlacao.iloc[0]:.2f}).")
    registrar("6. Segmentacao (K-Means): "
              + "; ".join(f"{l.segmento}: {int(l.restaurantes):,} rest., nota {l.nota_media:.2f}, INR {l.custo_medio:.0f}"
                          for l in perfil.itertuples()) + ".")
    registrar(f"7. Classificacao: acuracia {metricas_classe['acuracia']:.1%} e F1 {metricas_classe['f1']:.3f} "
              f"para identificar restaurantes com nota >= 4.0.")
    registrar(f"8. Regressao: erro medio de {metricas_reg['MAE']:.3f} ponto na nota (R2 {metricas_reg['R2']:.3f}), "
              f"contra {metricas_reg['MAE baseline (media)']:.3f} usando apenas a media.")
    registrar("")
    registrar("Recomendacoes para tomada de decisao:")
    registrar("- Estimular avaliacoes dos clientes: o volume de votos e o principal sinal de boa nota.")
    registrar("- Oferecer reserva de mesa em restaurantes de ticket medio/alto.")
    registrar("- Evitar bairros saturados ou entrar neles com diferencial claro.")
    registrar("- Usar os segmentos para campanhas de marketing direcionadas.")


def main():
    preparar_pastas()
    registrar("PROJETO DE ANALISE DE DADOS - RESTAURANTES ZOMATO (BANGALORE)")
    registrar("Processo OSEMN: Obter, Limpar, Explorar, Modelar, Interpretar")
    restaurantes = limpar(obter())
    correlacao = explorar(restaurantes)
    perfil, metricas_classe, metricas_reg = modelar(restaurantes)
    interpretar(restaurantes, correlacao, perfil, metricas_classe, metricas_reg)
    with open(os.path.join(PASTA_RELATORIOS, "relatorio_gerencial.txt"), "w", encoding="utf-8") as arquivo:
        arquivo.write("\n".join(relatorio))
    print("\nRelatorio salvo em relatorios/relatorio_gerencial.txt")


if __name__ == "__main__":
    main()
