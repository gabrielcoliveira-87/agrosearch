"""
AgroSearch - Motor de Busca Inteligente (AgroTech Solutions)
============================================================
Protótipo de motor de busca textual em Streamlit.

Executar:  streamlit run agrosearch.py

Restrição do desafio: o índice invertido, o TF, o IDF, o TF-IDF e a
similaridade de cosseno (bônus) são implementados "do zero", usando apenas
a biblioteca padrão do Python (re, math, unicodedata, collections).
O pandas (já instalado junto com o Streamlit) é usado SOMENTE para exibir
tabelas na interface; nenhum cálculo depende dele.

Estrutura do arquivo
    1. Base de documentos
    2. Fase 1 - Pipeline de pré-processamento
    3. Fase 2 - Índice invertido
    4. Fase 3 - TF, IDF, TF-IDF e ranqueamento (+ cosseno bônus)
    5. Interface Streamlit
"""

import math
import re
import unicodedata
from collections import Counter

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 1. BASE DE DOCUMENTOS (hardcode sugerido no enunciado)
# ---------------------------------------------------------------------------
DOCUMENTOS_PADRAO = [
    "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
]

# ---------------------------------------------------------------------------
# 2. FASE 1 - PIPELINE DE PRÉ-PROCESSAMENTO
#    Tokenização -> Normalização -> Stopwords -> Stemming
# ---------------------------------------------------------------------------

# Stopwords do português (já sem acentos, pois são removidas APÓS a normalização)
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos",
    "e", "ou", "para", "pra", "por", "com", "sem", "sobre", "entre",
    "ao", "aos", "que", "se", "como", "mais", "muito", "mas", "ja",
    "ser", "sao", "foi", "pode", "podem", "feito", "e", "eh", "sua", "seu",
    "suas", "seus", "esta", "este", "isso", "essa", "esse", "durante",
}

# Sufixos do português (sem acentos), do mais longo para o mais curto.
# Stemmer simplificado inspirado no RSLP, implementado do zero.
SUFIXOS = sorted(
    [
        "amentos", "imentos", "amento", "imento",
        "adoras", "adores", "adora", "ador",
        "acoes", "acao", "idades", "idade",
        "mente", "osas", "osos", "osa", "oso",
        "icas", "icos", "ica", "ico",
        "ando", "endo", "indo", "aram", "eram",
        "ar", "er", "ir",
    ],
    key=len,
    reverse=True,
)
TAM_MIN_RADICAL = 3  # nunca deixa o radical menor que isso


def tokenizar(texto: str) -> list[str]:
    """Etapa 1 - Tokenização: quebra o texto em palavras (letras/dígitos)."""
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+", texto)


def remover_acentos(palavra: str) -> str:
    """Decompõe (NFD) e descarta os caracteres de acento combinantes."""
    decomposta = unicodedata.normalize("NFD", palavra)
    return "".join(c for c in decomposta if unicodedata.category(c) != "Mn")


def normalizar(token: str) -> str:
    """Etapa 2 - Normalização: minúsculas + remoção de acentos."""
    return remover_acentos(token.lower())


def stem(palavra: str) -> str:
    """Etapa 4 - Stemming (redução ao radical) por remoção de sufixos."""
    # (a) plural: remove o 's' final ("lagartas" -> "lagarta")
    if len(palavra) > 4 and palavra.endswith("s"):
        palavra = palavra[:-1]
    # (b) sufixos derivacionais/verbais mais comuns
    for suf in SUFIXOS:
        if palavra.endswith(suf) and len(palavra) - len(suf) >= TAM_MIN_RADICAL:
            palavra = palavra[: -len(suf)]
            break
    # (c) vogal temática final ("soja" -> "soj", "controle" -> "control")
    if len(palavra) > TAM_MIN_RADICAL + 1 and palavra[-1] in "aeo":
        palavra = palavra[:-1]
    return palavra


def preprocessar(texto: str, usar_stopwords: bool = True, usar_stemming: bool = True) -> list[str]:
    """Executa o pipeline completo e devolve a lista final de termos."""
    tokens = [normalizar(t) for t in tokenizar(texto)]
    if usar_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if usar_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


def pipeline_detalhado(texto: str, usar_stopwords: bool, usar_stemming: bool) -> dict:
    """Devolve o resultado de cada etapa (usado na visualização da Fase 1)."""
    tokens = tokenizar(texto)
    normalizados = [normalizar(t) for t in tokens]
    sem_stop = [t for t in normalizados if t not in STOPWORDS] if usar_stopwords else normalizados
    finais = [stem(t) for t in sem_stop] if usar_stemming else sem_stop
    return {
        "1. Tokenização": tokens,
        "2. Normalização": normalizados,
        "3. Stopwords": sem_stop,
        "4. Stemming": finais,
    }


# ---------------------------------------------------------------------------
# 3. FASE 2 - ÍNDICE INVERTIDO  (Termo -> [IDs de documentos])
# ---------------------------------------------------------------------------
def construir_indice_invertido(docs_tokens: list[list[str]]) -> dict[str, list[int]]:
    """IDs dos documentos começam em 1 (Doc 1 ... Doc N)."""
    indice: dict[str, list[int]] = {}
    for doc_id, tokens in enumerate(docs_tokens, start=1):
        for termo in set(tokens):  # set: cada doc entra uma vez por termo
            indice.setdefault(termo, []).append(doc_id)
    return {termo: sorted(ids) for termo, ids in sorted(indice.items())}


# ---------------------------------------------------------------------------
# 4. FASE 3 - TF, IDF, TF-IDF e RANQUEAMENTO
# ---------------------------------------------------------------------------
def calcular_tf(tokens: list[str], modo: str = "relativo") -> dict[str, float]:
    """
    TF(t,d):
      'relativo' -> f(t,d) / total de termos do documento
      'bruto'    -> f(t,d)  (contagem simples)
    """
    contagem = Counter(tokens)
    total = len(tokens)
    if modo == "bruto" or total == 0:
        return {t: float(c) for t, c in contagem.items()}
    return {t: c / total for t, c in contagem.items()}


def calcular_idf(indice: dict[str, list[int]], n_docs: int, modo: str = "classico") -> dict[str, float]:
    """
    IDF(t):
      'classico'  -> log10(N / df(t))               (0 se o termo está em todos os docs)
      'suavizado' -> log10((1 + N) / (1 + df)) + 1  (nunca é zero)
    """
    idf = {}
    for termo, ids in indice.items():
        df = len(ids)
        if modo == "suavizado":
            idf[termo] = math.log10((1 + n_docs) / (1 + df)) + 1
        else:
            idf[termo] = math.log10(n_docs / df)
    return idf


def vetor_tfidf(tokens: list[str], idf: dict[str, float], modo_tf: str) -> dict[str, float]:
    """Vetor esparso {termo: tf * idf}. Termos fora do vocabulário são ignorados."""
    tf = calcular_tf(tokens, modo_tf)
    return {t: tf[t] * idf[t] for t in tf if t in idf}


def similaridade_cosseno(v1: dict[str, float], v2: dict[str, float]) -> float:
    """cos(θ) = (v1·v2) / (||v1|| · ||v2||) sobre vetores esparsos."""
    produto = sum(v1[t] * v2[t] for t in v1 if t in v2)
    norma1 = math.sqrt(sum(x * x for x in v1.values()))
    norma2 = math.sqrt(sum(x * x for x in v2.values()))
    if norma1 == 0 or norma2 == 0:
        return 0.0
    return produto / (norma1 * norma2)


def buscar(query_tokens, docs_tokens, indice, idf, modo_tf):
    """
    Ranqueia os documentos para a query.
    Score TF-IDF acumulado(d) = Σ_{t ∈ query} TF(t,d) · IDF(t)
    Devolve (lista de resultados, detalhamento por termo).
    """
    n_docs = len(docs_tokens)
    termos_query = list(dict.fromkeys(query_tokens))  # únicos, preservando a ordem
    termos_validos = [t for t in termos_query if t in indice]
    termos_desconhecidos = [t for t in termos_query if t not in indice]

    tfs = [calcular_tf(toks, modo_tf) for toks in docs_tokens]
    vetores_docs = [{t: tf[t] * idf[t] for t in tf} for tf in tfs]
    vetor_q = vetor_tfidf(query_tokens, idf, modo_tf)

    detalhes = []  # uma linha por (termo, documento)
    resultados = []
    for i in range(n_docs):
        doc_id = i + 1
        score = 0.0
        for t in termos_validos:
            tf = tfs[i].get(t, 0.0)
            parcela = tf * idf[t]
            score += parcela
            detalhes.append({"Termo": t, "Doc": doc_id, "TF": tf, "IDF": idf[t], "TF-IDF": parcela})
        resultados.append(
            {
                "doc_id": doc_id,
                "score": score,
                "cosseno": similaridade_cosseno(vetor_q, vetores_docs[i]),
            }
        )
    return resultados, detalhes, termos_validos, termos_desconhecidos


# ---------------------------------------------------------------------------
# 5. INTERFACE STREAMLIT
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")
    st.title("🌱 AgroSearch – Motor de Busca Inteligente")
    st.caption("AgroTech Solutions · Índice invertido e TF-IDF implementados do zero (sem scikit-learn)")

    # ---------------- Barra lateral: configuração ----------------
    with st.sidebar:
        st.header("⚙️ Configurações do pipeline")
        usar_stopwords = st.checkbox("Remover stopwords", value=True)
        usar_stemming = st.checkbox("Aplicar stemming", value=True)

        st.header("📐 Fórmulas")
        modo_tf = st.selectbox(
            "TF", ["relativo", "bruto"],
            format_func=lambda m: "Frequência relativa (f/total)" if m == "relativo" else "Contagem bruta (f)",
        )
        modo_idf = st.selectbox(
            "IDF", ["classico", "suavizado"],
            format_func=lambda m: "Clássico: log10(N/df)" if m == "classico" else "Suavizado: log10((1+N)/(1+df))+1",
        )
        usar_cosseno = st.checkbox("⭐ Bônus: similaridade de cosseno", value=True)

        st.header("📚 Base de documentos")
        texto_base = st.text_area(
            "Um documento por linha (editável)",
            value="\n".join(DOCUMENTOS_PADRAO),
            height=300,
        )

    documentos = [linha.strip() for linha in texto_base.splitlines() if linha.strip()]
    if not documentos:
        st.error("Adicione pelo menos um documento na barra lateral.")
        return

    # Pré-processa toda a base com a configuração escolhida
    docs_tokens = [preprocessar(d, usar_stopwords, usar_stemming) for d in documentos]
    indice = construir_indice_invertido(docs_tokens)
    idf = calcular_idf(indice, len(documentos), modo_idf)

    aba1, aba2, aba3 = st.tabs(["1️⃣ Pré-processamento", "2️⃣ Índice invertido", "3️⃣ Busca e ranking TF-IDF"])

    # ---------------- Fase 1 ----------------
    with aba1:
        st.subheader("Fase 1 · Pipeline de pré-processamento")
        st.write(
            "Ligue/desligue **stopwords** e **stemming** na barra lateral e "
            "observe o vocabulário mudar."
        )

        # Vocabulário nas 4 combinações, para comparação
        def tamanho_vocab(sw, sm):
            return len({t for d in documentos for t in preprocessar(d, sw, sm)})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vocabulário atual", len(indice))
        c2.metric("Sem stopwords/stemming", tamanho_vocab(False, False))
        c3.metric("Só stopwords", tamanho_vocab(True, False))
        c4.metric("Só stemming", tamanho_vocab(False, True))

        opcoes = [f"Doc {i}" for i in range(1, len(documentos) + 1)]
        escolhido = st.selectbox("Documento para inspecionar", opcoes)
        idx = opcoes.index(escolhido)
        st.info(documentos[idx])
        etapas = pipeline_detalhado(documentos[idx], usar_stopwords, usar_stemming)
        colunas = st.columns(4)
        for col, (nome, toks) in zip(colunas, etapas.items()):
            col.markdown(f"**{nome}** ({len(toks)})")
            col.dataframe(pd.DataFrame({"token": toks}), hide_index=True, use_container_width=True)

        with st.expander("📖 Vocabulário completo (termos únicos)"):
            st.write(", ".join(sorted(indice)))

    # ---------------- Fase 2 ----------------
    with aba2:
        st.subheader("Fase 2 · Índice invertido (Termo → [IDs dos documentos])")
        st.caption(f"{len(indice)} termos · construído a partir dos tokens já pré-processados.")
        modo_vis = st.radio("Visualização", ["Tabela", "JSON"], horizontal=True)
        if modo_vis == "Tabela":
            df_idx = pd.DataFrame(
                {
                    "Termo": list(indice),
                    "Documentos": [", ".join(f"Doc {i}" for i in ids) for ids in indice.values()],
                    "df": [len(ids) for ids in indice.values()],
                    "IDF": [round(idf[t], 4) for t in indice],
                }
            )
            st.dataframe(df_idx, hide_index=True, use_container_width=True)
        else:
            st.json(indice)

    # ---------------- Fase 3 ----------------
    with aba3:
        st.subheader("Fase 3 · Busca e ranqueamento TF-IDF")
        query = st.text_input("Consulta do técnico", value="irrigação da soja", placeholder="ex.: controle de lagartas")

        with st.expander("📘 Fórmulas utilizadas"):
            st.latex(r"TF(t,d)=\frac{f(t,d)}{|d|}" if modo_tf == "relativo" else r"TF(t,d)=f(t,d)")
            st.latex(r"IDF(t)=\log_{10}\frac{N}{df(t)}" if modo_idf == "classico"
                     else r"IDF(t)=\log_{10}\frac{1+N}{1+df(t)}+1")
            st.latex(r"\text{Score}(d)=\sum_{t\in q} TF(t,d)\cdot IDF(t)")
            st.latex(r"\cos(q,d)=\frac{\vec q\cdot\vec d}{\|\vec q\|\,\|\vec d\|}")

        if query.strip():
            q_tokens = preprocessar(query, usar_stopwords, usar_stemming)
            st.write("**Query processada:**", q_tokens if q_tokens else "(vazia após o pré-processamento)")

            resultados, detalhes, validos, desconhecidos = buscar(q_tokens, docs_tokens, indice, idf, modo_tf)
            if desconhecidos:
                st.warning(f"Termos fora do vocabulário (ignorados): {', '.join(desconhecidos)}")

            ranquear_cosseno = False
            if usar_cosseno:
                ranquear_cosseno = st.toggle("Ranquear pelo cosseno (em vez do TF-IDF acumulado)")
            chave = "cosseno" if ranquear_cosseno else "score"
            resultados.sort(key=lambda r: (r[chave], r["score"]), reverse=True)

            if not resultados or resultados[0][chave] <= 0:
                st.error("Nenhum documento relevante encontrado para essa consulta.")
            else:
                venc = resultados[0]
                st.success(f"🏆 Documento vencedor: **Doc {venc['doc_id']}**")
                st.markdown(f"> {documentos[venc['doc_id'] - 1]}")

                linhas = []
                for pos, r in enumerate(resultados, start=1):
                    linha = {
                        "Posição": pos,
                        "": "🏆" if pos == 1 else "",
                        "Documento": f"Doc {r['doc_id']}",
                        "TF-IDF acumulado": round(r["score"], 5),
                    }
                    if usar_cosseno:
                        linha["Cosseno"] = round(r["cosseno"], 5)
                    linha["Trecho"] = documentos[r["doc_id"] - 1]
                    linhas.append(linha)
                st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)

                with st.expander("🔍 Detalhamento: TF, IDF e TF-IDF por termo e documento"):
                    df_det = pd.DataFrame(detalhes)
                    df_det["Doc"] = df_det["Doc"].map(lambda i: f"Doc {i}")
                    for col in ("TF", "IDF", "TF-IDF"):
                        df_det[col] = df_det[col].round(5)
                    st.dataframe(df_det, hide_index=True, use_container_width=True)

                    st.markdown("**Matriz TF-IDF por termo da consulta**")
                    if detalhes:
                        matriz = pd.DataFrame(detalhes).pivot(index="Termo", columns="Doc", values="TF-IDF").round(5)
                        matriz.columns = [f"Doc {c}" for c in matriz.columns]
                        st.dataframe(matriz, use_container_width=True)
        else:
            st.info("Digite uma consulta para buscar nos manuais.")


if __name__ == "__main__":
    main()
