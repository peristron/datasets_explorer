# run command:
#                             streamlit run python_datahub_dataset_relationships_v125.py
#                  directory setup: cd C:\users\oakhtar\documents\pyprojs_local
#   OPTIMIZED for deployment - LKG - as of 11.18.2025
# v125 update: Major UX improvements. The sidebar is now the single source of truth for dataset selection, removing the redundant 'Focus Datasets' widget.
#              The app now provides clear, explicit feedback when no direct relationships are found between selected datasets, preventing user confusion.

import pandas as pd
import re
import streamlit as st
import os
import logging
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import time
import networkx as nx
import plotly.graph_objects as go

# Logging Setup & Warning Suppression
logging.basicConfig(filename='scraper.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# --- Known URLs Constant ---
# Pre-filled list of the 38 known dataset category URLs.
DEFAULT_URLS = """
https://community.d2l.com/brightspace/kb/articles/4752-accommodations-data-sets
https://community.d2l.com/brightspace/kb/articles/4712-activity-feed-data-sets
https://community.d2l.com/brightspace/kb/articles/4723-announcements-data-sets
https://community.d2l.com/brightspace/kb/articles/4767-assignments-data-sets
https://community.d2l.com/brightspace/kb/articles/4519-attendance-data-sets
https://community.d2l.com/brightspace/kb/articles/4520-awards-data-sets
https://community.d2l.com/brightspace/kb/articles/4521-calendar-data-sets
https://community.d2l.com/brightspace/kb/articles/4523-checklist-data-sets
https://community.d2l.com/brightspace/kb/articles/4754-competency-data-sets
https://community.d2l.com/brightspace/kb/articles/4713-content-data-sets
https://community.d2l.com/brightspace/kb/articles/22812-content-service-data-sets
https://community.d2l.com/brightspace/kb/articles/26020-continuous-professional-development-cpd-data-sets
https://community.d2l.com/brightspace/kb/articles/4725-course-copy-data-sets
https://community.d2l.com/brightspace/kb/articles/4524-course-publisher-data-sets
https://community.d2l.com/brightspace/kb/articles/26161-creator-data-sets
https://community.d2l.com/brightspace/kb/articles/4525-discussions-data-sets
https://community.d2l.com/brightspace/kb/articles/4526-exemptions-data-sets
https://community.d2l.com/brightspace/kb/articles/4527-grades-data-sets
https://community.d2l.com/brightspace/kb/articles/4528-intelligent-agents-data-sets
https://community.d2l.com/brightspace/kb/articles/5782-jit-provisioning-data-sets
https://community.d2l.com/brightspace/kb/articles/4714-local-authentication-data-sets
https://community.d2l.com/brightspace/kb/articles/4727-lti-data-sets
https://community.d2l.com/brightspace/kb/articles/4529-organizational-units-data-sets
https://community.d2l.com/brightspace/kb/articles/4796-outcomes-data-sets
https://community.d2l.com/brightspace/kb/articles/4530-portfolio-data-sets
https://community.d2l.com/brightspace/kb/articles/4531-questions-data-sets
https://community.d2l.com/brightspace/kb/articles/4532-quizzes-data-sets
https://community.d2l.com/brightspace/kb/articles/4533-release-conditions-data-sets
https://community.d2l.com/brightspace/kb/articles/33182-reoffer-course-data-sets
https://community.d2l.com/brightspace/kb/articles/4534-role-details-data-sets
https://community.d2l.com/brightspace/kb/articles/4535-rubrics-data-sets
https://community.d2l.com/brightspace/kb/articles/4536-scorm-data-sets
https://community.d2l.com/brightspace/kb/articles/4537-sessions-and-system-access-data-sets
https://community.d2l.com/brightspace/kb/articles/19147-sis-course-merge-data-sets
https://community.d2l.com/brightspace/kb/articles/4538-surveys-data-sets
https://community.d2l.com/brightspace/kb/articles/4540-tools-data-sets
https://community.d2l.com/brightspace/kb/articles/4740-users-data-sets
https://community.d2l.com/brightspace/kb/articles/4541-virtual-classroom-data-sets
""".strip()


def parse_urls_from_text_area(text_block):
    """Takes a block of text and extracts all valid URLs, one per line."""
    logging.info("Parsing text area for URLs...")
    urls = [line.strip() for line in text_block.split('\n') if line.strip()]
    valid_urls = [url for url in urls if url.startswith('http')]
    unique_urls = sorted(list(set(valid_urls)))
    logging.info(f"Found {len(unique_urls)} unique URLs to scrape.")
    return unique_urls

def scrape_table(url, category_name):
    """Scrapes table data from a single dataset page. No longer uses .pkl file caching."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        data = []
        elements = soup.find_all(['h2', 'h3', 'table'])
        current_dataset = category_name
        
        for element in elements:
            if element.name in ['h2', 'h3']: 
                current_dataset = element.text.strip().lower()
            elif element.name == 'table':
                table_headers = [th.text.strip().lower().replace(' ', '_') for th in element.find_all('th')]
                if not table_headers: continue
                for row in element.find_all('tr'):
                    columns_ = row.find_all('td')
                    if len(columns_) != len(table_headers): continue
                    entry = {table_headers[i]: columns_[i].text.strip() for i in range(len(table_headers))}
                    header_map = {'field': 'column_name', 'name': 'column_name', 'type': 'data_type'}
                    entry = {header_map.get(k, k): v for k, v in entry.items()}
                    if 'column_name' not in entry or not entry['column_name']: continue
                    entry['dataset_name'] = current_dataset
                    entry['category'] = category_name
                    data.append(entry)
        if not data:
            logging.warning(f"No data extracted from tables on page: {url}")
            return []
        df = pd.DataFrame(data)
        return df.to_dict('records')
    except Exception as e:
        logging.error(f"Error scraping page {url}: {e}")
        return []

def scrape_and_save_from_list(url_list):
    """Orchestrates scraping all URLs from the provided list."""
    all_data = []
    progress_bar = st.progress(0, "Scraping dataset pages...")
    
    def get_category_from_url(url):
        return re.sub(r'^\d+\s*', '', os.path.basename(url).replace('-data-sets', '').replace('-', ' ')).lower()

    with ThreadPoolExecutor(max_workers=10) as executor:
        args = [(url, get_category_from_url(url)) for url in url_list]
        future_to_url = {executor.submit(scrape_table, *arg): arg[0] for arg in args}
        for i, future in enumerate(future_to_url):
            try:
                result = future.result()
                all_data.extend(result)
            except Exception as e:
                logging.error(f"A scraping thread failed: {e}")
            progress_bar.progress((i + 1) / len(url_list), f"Scraping... {i+1}/{len(url_list)} pages.")
    
    progress_bar.empty()
    if not all_data:
        logging.error("Scraping finished, but no data was collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    expected_cols = ['category', 'dataset_name', 'column_name', 'data_type', 'description', 'key', 'version', 'version_history', 'column_size', 'notes']
    for col in expected_cols:
        if col not in df.columns: df[col] = ''
    
    df = df.fillna('')
    df['is_primary_key'] = df['key'].str.contains('pk', case=False, na=False)
    df['is_foreign_key'] = df['key'].str.contains('fk', case=False, na=False)
    df['foreign_key_references'] = ''
    
    df.to_csv('dataset_metadata.csv', index=False)
    logging.info(f"Scraping complete. Saved {len(df)} rows to dataset_metadata.csv")
    return df

@st.cache_data
def find_pk_fk_joins(df, selected_datasets):
    """Analyzes the dataframe to find all potential PK-FK joins for the selected datasets."""
    if df.empty or not selected_datasets:
        return pd.DataFrame()
        
    pks = df[df['is_primary_key'] == True]
    fks = df[(df['is_foreign_key'] == True) & (df['dataset_name'].isin(selected_datasets))]
    
    if pks.empty or fks.empty:
        return pd.DataFrame()
    
    merged = pd.merge(fks, pks, on='column_name', suffixes=('_fk', '_pk'))
    joins = merged[merged['dataset_name_fk'] != merged['dataset_name_pk']]
    
    if joins.empty:
        return pd.DataFrame()
        
    result = joins[['dataset_name_fk', 'column_name', 'dataset_name_pk', 'category_pk']]
    result.columns = ['Source Dataset', 'Join Column', 'Target Dataset', 'Target Category']
    return result.drop_duplicates().reset_index(drop=True)

def main():
    st.set_page_config(page_title="Dataset Explorer v125", layout="wide", page_icon="🕸️")
    st.markdown("<h2 style='color: #ffffff; text-align: center;'>Dataset Explorer</h2>", unsafe_allow_html=True)
    logging.info("--- Streamlit App Initialized v125 ---")

    # --- Reorganized Sidebar for Better UX ---
    with st.sidebar.expander("STEP 1: Load or Update Data", expanded=True):
        st.info("To get the latest data, click the button below. Add any new URLs to the text box first.")
        pasted_text = st.text_area("URLs to Scrape (one per line):", height=250, key="paste_area", value=DEFAULT_URLS)
        if st.button("Scrape All URLs in Text Area", type="primary"):
            url_list = parse_urls_from_text_area(pasted_text)
            if not url_list:
                st.error("No valid URLs found. Please paste URLs starting with 'http', one per line.")
            else:
                with st.spinner(f"Found {len(url_list)} URLs. Now scraping all pages..."):
                    scrape_and_save_from_list(url_list)
                    st.success("Scrape complete! Refreshing app with new data...")
                    st.rerun() 

    # --- Main Data Loading and Display Logic ---
    columns = pd.DataFrame()
    metadata_file = 'dataset_metadata.csv'

    if os.path.exists(metadata_file):
        try:
            columns = pd.read_csv(metadata_file).fillna('')
        except Exception as e:
            st.error(f"Could not load cached 'dataset_metadata.csv'. To fix, please use the 'Update Data' section. Error: {e}")
            return
    else:
        st.warning("No local data cache found. Please use the 'STEP 1: Load or Update Data' section in the sidebar to load data.")
        return

    # --- V125: UI and Display Code - Streamlined Sidebar ---
    st.sidebar.title("STEP 2: Explore Datasets")
    if columns.empty:
        st.info("No data is loaded.")
        return
        
    categories = sorted(c for c in columns['category'].unique() if c)
    datasets = sorted(d for d in columns['dataset_name'].unique() if d)
    st.sidebar.info(f"Loaded {len(datasets)} datasets across {len(categories)} categories.")
    
    selected_categories = st.sidebar.multiselect("Filter by Category", categories, default=[])
    filtered_datasets = sorted(columns[columns['category'].isin(selected_categories)]['dataset_name'].unique()) if selected_categories else datasets
    # V125: This is now the single source of truth for selection
    selected_datasets = st.sidebar.multiselect(
        "Select Datasets to Explore", 
        filtered_datasets, 
        default=[],
        help="Select datasets here to view their details and graph their connections."
    )
    
    # --- Graph Layout Controls ---
    st.sidebar.subheader("Graph Layout Controls")
    graph_font_size = st.sidebar.slider("Node Font Size", min_value=8, max_value=24, value=12, help="Adjust the text size on the graph nodes.")
    node_separation = st.sidebar.slider("Node Separation", min_value=0.1, max_value=2.5, value=0.9, help="Increase to spread nodes further apart.")
    graph_height = st.sidebar.slider("Graph Height (px)", min_value=500, max_value=1500, value=700, help="Increase the vertical space for the graph.")
    show_edge_labels = st.sidebar.checkbox("Show Join Column Labels", value=False, help="Toggle visibility of the labels on connection lines.")

    # --- Main Page Content ---
    with st.expander("❓ How to Use This Application", expanded=False):
        st.markdown("""
        This tool allows you to explore Brightspace datasets, their columns, and their relationships.
        #### Quick Start
        1.  **Load Data:** If this is your first time, or you want the latest data, use the **'STEP 1: Load or Update Data'** section in the sidebar.
        2.  **Filter & Select:** Use the **'STEP 2: Explore Datasets'** section to filter by category and select datasets.
        3.  **View Details:** The details for your selected datasets will appear below in expandable tables.
        4.  **Explore Connections:** The graph at the bottom will automatically update to show direct relationships between your selected datasets. Use the **Graph Layout Controls** in the sidebar to adjust the view for clarity.

        ---
        **Why is the manual update step needed?** The D2L Community website is a modern web application protected by security measures that block simple automated scripts. This manual-assist approach is the most reliable way to handle these issues, putting you in control of what data gets loaded.
        """)

    st.subheader("Dataset Details")
    if selected_datasets:
        for dataset in selected_datasets:
            with st.expander(f"Details for: **{dataset}**", expanded=False):
                dataset_cols = columns[columns['dataset_name'] == dataset].copy()
                display_cols = ['column_name', 'data_type', 'description', 'key', 'version', 'version_history', 'column_size']
                display_cols_exist = [c for c in display_cols if c in dataset_cols.columns and not dataset_cols[c].astype(str).str.strip().eq('').all()]
                if display_cols_exist: st.dataframe(dataset_cols[display_cols_exist], use_container_width=True, hide_index=True)
                else: st.write("No detailed columns to display for this dataset.")
    else:
        st.info("Select one or more datasets from the sidebar to view their details and explore connections.")

    # --- V125: Graphing and Relationship Section - Rewritten for better UX ---
    st.subheader("Dataset Connection Explorer")
    st.caption("This graph shows direct PK-FK relationships *between* the datasets you selected in the sidebar.")

    # Don't try to graph if not enough datasets are selected
    if len(selected_datasets) < 2:
        st.info("Select at least two datasets in the sidebar to visualize their relationships.")
    else:
        join_data = find_pk_fk_joins(columns, selected_datasets)
        
        G = nx.DiGraph()
        if not join_data.empty:
            for _, row in join_data.iterrows():
                source = row['Source Dataset']
                target = row['Target Dataset']
                # Only add edges between the selected datasets
                if source in selected_datasets and target in selected_datasets:
                    G.add_node(source, type='focus')
                    G.add_node(target, type='focus') # Both are 'focus' now
                    G.add_edge(source, target, label=row['Join Column'])

        # V125: Provide clear feedback if no direct connections are found
        if not G.edges:
            st.warning(f"No direct PK-FK relationships found between the selected datasets: **{', '.join(selected_datasets)}**")
        else:
            with st.expander("View Joinable Relationships Table", expanded=False):
                # Filter join_data to only show the connections that are actually on the graph
                graph_joins = join_data[
                    (join_data['Source Dataset'].isin(G.nodes())) & 
                    (join_data['Target Dataset'].isin(G.nodes()))
                ]
                st.dataframe(graph_joins, use_container_width=True)

            pos = nx.spring_layout(G, k=node_separation, iterations=50) 
            
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0, x1, y1 = pos[edge[0]][0], pos[edge[0]][1], pos[edge[1]][0], pos[edge[1]][1]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.5, color='#888'), hoverinfo='none', mode='lines')

            node_x, node_y, node_text, node_color, node_size, node_hover = [], [], [], [], [], []
            cat_colors = {cat: f"hsl({(hash(cat)*137.5) % 360}, 70%, 60%)" for cat in categories}

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_size.append(30) # All nodes are focus nodes now
                category = columns[columns['dataset_name'] == node]['category'].iloc[0] if not columns[columns['dataset_name'] == node].empty else 'unknown'
                node_color.append(cat_colors.get(category, '#ccc'))
                node_hover.append(f"<b>{node}</b><br>Category: {category}")
                
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                hoverinfo='text', hovertext=node_hover,
                text=node_text, textposition="top center", 
                textfont=dict(size=graph_font_size, color='#fff'),
                marker=dict(showscale=False, color=node_color, size=node_size, line_width=2)
            )
            
            annotations = []
            if show_edge_labels:
                for edge in G.edges(data=True):
                    x0, y0, x1, y1 = pos[edge[0]][0], pos[edge[0]][1], pos[edge[1]][0], pos[edge[1]][1]
                    annotations.append(dict(
                        x=(x0+x1)/2, y=(y0+y1)/2, xref='x', yref='y',
                        text=edge[2].get('label', ''), showarrow=False,
                        font=dict(color="cyan", size=max(8, graph_font_size - 4)),
                        ax=20, ay=-20
                    ))

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
                                annotations=annotations,
                                height=graph_height
                            ))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
