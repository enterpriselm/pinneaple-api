import altair as alt
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# ========= CONFIG =========
st.set_page_config(page_title="PINNeAPPle References", layout="wide")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
DB_PATH = "search_results.db"

# ========= DB FUNCTIONS =========
def get_connection():
    return sqlite3.connect(DB_PATH)

def get_dashboard_data():
    conn = get_connection()
    
    arxiv_count = pd.read_sql("SELECT COUNT(*) FROM arxiv_papers", conn).iloc[0, 0]
    github_count = pd.read_sql("SELECT COUNT(*) FROM github_repos", conn).iloc[0, 0]
    
    area_counts = pd.read_sql("SELECT area, COUNT(*) as total FROM arxiv_papers GROUP BY area", conn)

    df_authors = pd.read_sql("SELECT authors FROM arxiv_papers", conn)
    df_authors["authors"] = df_authors["authors"].str.split(",")
    df_authors = df_authors.explode("authors")
    df_authors["authors"] = df_authors["authors"].str.strip()
    top_authors_papers = df_authors["authors"].value_counts().head(5).reset_index()
    top_authors_papers.columns = ["authors", "total"]

    top_authors_repos = pd.read_sql("""
        SELECT author, COUNT(*) as total 
        FROM github_repos 
        GROUP BY author 
        ORDER BY total DESC LIMIT 5
    """, conn)
    
    conn.close()
    return arxiv_count, github_count, area_counts, top_authors_papers, top_authors_repos


def get_filtered_data(table, filters):
    conn = get_connection()
    query = f"SELECT * FROM {table} WHERE 1=1"
    for key, value in filters.items():
        if value:
            query += f" AND {key} LIKE '%{value}%'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def semantic_search_arxiv(query, top_k=5):
    conn = get_connection()
    cur = conn.cursor()
    query_embedding = embedder.encode(query, normalize_embeddings=True)

    cur.execute("SELECT id, paper_name, abstract, embedding FROM arxiv_papers WHERE embedding IS NOT NULL")
    results = []
    for row in cur.fetchall():
        id_, name, abstract, blob = row
        emb = pickle.loads(blob)
        score = np.dot(query_embedding, emb)
        results.append((score, id_, name, abstract))
    conn.close()
    results.sort(reverse=True)
    return results[:top_k]

# ========= UI =========
def show_dashboard():
    st.title("📊 PINNeAPPle Dashboard")
    arxiv_count, github_count, area_counts, top_authors_papers, top_authors_repos = get_dashboard_data()

    col1, col2 = st.columns(2)
    col1.metric("Total Papers", arxiv_count)
    col2.metric("Total Repositories", github_count)

    st.subheader("📚 Papers by Area")
    chart = alt.Chart(area_counts).mark_bar().encode(
        x=alt.X("total:Q", title="Number of Papers"),
        y=alt.Y("area:N", sort='-x', title="Area"),
        tooltip=["area", "total"]
    ).properties(height=400)

    st.altair_chart(chart, use_container_width=True)

    st.subheader("👨‍🔬 Top Paper Authors")
    st.table(top_authors_papers)

    st.subheader("👨‍💻 Top Repo Authors")
    st.table(top_authors_repos)

def show_papers():
    st.title("📄 ArXiv Papers")
    with st.expander("🔎 Filters"):
        area = st.text_input("Area")
        subarea = st.text_input("Subarea")
        author = st.text_input("Author")
        title = st.text_input("Paper Title")

    filters = {
        "area": area,
        "subarea": subarea,
        "authors": author,
        "paper_name": title,
    }

    df = get_filtered_data("arxiv_papers", filters)
    st.write(f"🔢 Found {len(df)} papers")

    # Paginação
    per_page = 10
    max_page = max(1, (len(df) - 1) // per_page + 1)
    page_number = st.number_input("Page", min_value=1, max_value=max_page, step=1)
    start = (page_number - 1) * per_page
    end = start + per_page
    paginated_df = df.iloc[start:end]

    # Cabeçalho da tabela
    header_cols = st.columns([4, 2, 1, 1, 1])
    header_cols[0].markdown("**Paper Name**")
    header_cols[1].markdown("**Publication Date**")
    header_cols[2].markdown("**Area**")
    header_cols[3].markdown("**Subarea**")
    header_cols[4].markdown("**Details**")

    # Linhas da tabela com botões que atualizam o estado, sem rerun explícito
    for idx, row in paginated_df.iterrows():
        cols = st.columns([4, 2, 1, 1, 1])
        cols[0].markdown(row['paper_name'])
        cols[1].markdown(row["publication_date"])
        cols[2].markdown(row["area"])
        cols[3].markdown(row["subarea"])

        if cols[4].button("🔍", key=f"paper_{row['id']}"):
            st.session_state.page = "paper_detail"
            st.session_state.paper_id = row["id"]
            # Não usar st.rerun() nem experimental_rerun()
            return  # interrompe para a execução detectar a mudança

def show_repositories():
    st.title("📦 GitHub Repositories")
    with st.expander("🔎 Filters"):
        area = st.text_input("Area", key="repo_area")
        subarea = st.text_input("Subarea", key="repo_subarea")
        author = st.text_input("Author", key="repo_author")
        name = st.text_input("Repository Name", key="repo_name")

    filters = {
        "area": area,
        "subarea": subarea,
        "author": author,
        "repo_name": name,
    }

    df = get_filtered_data("github_repos", filters)
    st.write(f"🔢 Found {len(df)} repositories")

    per_page = 10
    max_page = max(1, (len(df) - 1) // per_page + 1)
    page_number = st.number_input("Repo Page", min_value=1, max_value=max_page, step=1)
    start = (page_number - 1) * per_page
    end = start + per_page
    paginated_df = df.iloc[start:end]

    header_cols = st.columns([4, 3, 2, 1, 1])
    header_cols[0].markdown("**Repository Name**")
    header_cols[1].markdown("**Author**")
    header_cols[2].markdown("**URL**")
    header_cols[3].markdown("**Area**")
    header_cols[4].markdown("**Details**")

    for idx, row in paginated_df.iterrows():
        cols = st.columns([4, 3, 2, 1, 1])
        cols[0].markdown(row['repo_name'])
        cols[1].markdown(row["author"])
        cols[2].markdown(row["repo_url"])
        cols[3].markdown(row["area"])

        if cols[4].button("🔍", key=f"repo_{row['id']}"):
            st.session_state.page = "repo_detail"
            st.session_state.repo_id = row["id"]
            return  # interrompe para a execução detectar a mudança

def show_paper_detail(paper_id):
    conn = get_connection()
    row = pd.read_sql(f"SELECT * FROM arxiv_papers WHERE id = {paper_id}", conn).iloc[0]
    conn.close()

    st.title(row["paper_name"])
    st.markdown(f"**Authors:** {row['authors']}")
    st.markdown(f"**Publication Date:** {row['publication_date']}")
    st.markdown(f"**Link:** [Access PDF]({row['url']})")
    st.markdown(f"**Area/Subarea:** {row['area']} / {row['subarea']}")
    st.subheader("📝 Abstract")
    st.write(row["abstract"])

def show_repo_detail(repo_id):
    conn = get_connection()
    row = pd.read_sql(f"SELECT * FROM github_repos WHERE id = {repo_id}", conn).iloc[0]
    conn.close()

    st.title(row["repo_name"])
    st.markdown(f"**Author:** {row['author']}")
    st.markdown(f"**Link:** [GitHub Repo]({row['repo_url']})")
    st.markdown(f"**Area/Subarea:** {row['area']} / {row['subarea']}")
    st.subheader("📖 README")
    if row["readme"]:
        st.markdown(row["readme"])
    else:
        st.warning("No README available.")

def show_semantic_search():
    st.title("🔍 Semantic Search in Papers")
    query = st.text_input("Enter your query:")
    if query:
        results = semantic_search_arxiv(query, top_k=5)
        st.write("### Results:")
        for score, id_, name, abstract in results:
            if st.button(name, key=f"search_{id_}"):
                st.session_state.page = "paper_detail"
                st.session_state.paper_id = id_
                st.rerun()
            st.markdown(f"**Score:** {score:.4f}")
            st.write(abstract)
            st.markdown("---")

# ========= ROUTING =========
def main():
    with st.sidebar:
        st.title("🍍 PINNeAPPle References")
        selected_page = st.radio("Navigation", ["Dashboard", "Papers", "Repositories", "Semantic Search"])
        st.markdown("---")
        st.markdown("© 2025 Enterprise Learning Machines")

    if "page" not in st.session_state or selected_page != st.session_state.page:
        st.session_state.page = selected_page

    if st.session_state.page == "Dashboard":
        show_dashboard()
    elif st.session_state.page == "Papers":
        show_papers()
    elif st.session_state.page == "Repositories":
        show_repositories()
    elif st.session_state.page == "Semantic Search":
        show_semantic_search()
    elif st.session_state.page == "paper_detail":
        show_paper_detail(st.session_state.paper_id)
    elif st.session_state.page == "repo_detail":
        show_repo_detail(st.session_state.repo_id)

if __name__ == "__main__":
    main()