# Dataset Explorer for Brightspace Data Hub

An interactive Streamlit application to visualize dataset relationships from the D2L Brightspace Community documentation. This tool scrapes the official documentation to build an interactive relationship graph, allowing users to explore how different datasets are connected.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://[your-app-url].streamlit.app/)

![App Screenshot](https://[link-to-your-screenshot.png])
*(Optional: After deploying, take a screenshot of your app, upload it to your GitHub repo, and link it here.)*

---

## About The Project

The Brightspace Data Hub provides a wealth of data, but understanding the relationships between its many datasets can be challenging. The official documentation is spread across dozens of pages, making it difficult to see the full picture.

This application solves that problem by:
1.  Scraping all dataset schema information from the D2L Community website.
2.  Identifying Primary Key (PK) and Foreign Key (FK) relationships.
3.  Providing an interactive network graph to visualize how datasets are interconnected.

## Key Features

-   🕸️ **Interactive Relationship Graph:** Visualize how "focus" datasets connect to others via shared keys.
-   📊 **Detailed Dataset Views:** View the columns, data types, and descriptions for any selected dataset in clean, searchable tables.
-   🔄 **On-Demand Data Scraping:** Fetch the latest dataset information directly from the D2L Community with the click of a button.
-   🎨 **Customizable View:** Adjust the graph's node separation, height, font size, and label visibility for optimal clarity.
-   🚀 **Deployment-Ready:** Optimized for easy, free deployment via Streamlit Community Cloud.

## Built With

-   [Streamlit](https://streamlit.io/)
-   [Pandas](https://pandas.pydata.org/)
-   [Plotly](https://plotly.com/python/)
-   [NetworkX](https://networkx.org/)
-   [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)

---

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

-   Python 3.8+
-   pip

### Installation & Local Execution

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/[your-github-username]/[your-repo-name].git
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd [your-repo-name]
    ```

3.  **Install the required Python packages:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```sh
    streamlit run python_datahub_dataset_relationships_v124.py
    ```
    The application will open in your web browser.

---

## Usage

1.  **Load Data:** On first use, or to get the latest data, use the "Scrape All URLs in Text Area" button in the sidebar. This may take a minute.
2.  **Filter & Select:** Use the filters in the sidebar to narrow down the list of datasets, then select one or more to view their details.
3.  **View Details:** Expand the sections on the main page to see the column information for your selected datasets.
4.  **Explore Connections:** At the bottom of the page, choose "Focus" datasets to see them in the graph. The graph will automatically show all related datasets.
5.  **Adjust Layout:** Use the "Graph Layout Controls" in the sidebar to make the graph easier to read.

## Deployment

This application is designed to be deployed on **Streamlit Community Cloud**. To deploy your own version:
1.  Fork this repository or create your own with the same file structure.
2.  Ensure your GitHub repository is public.
3.  Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
4.  Choose your repository and click "Deploy".

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

-   Data sourced from the official [D2L Brightspace Community](https://community.d2l.com/brightspace/kb/articles/4515-data-hub-brightspace-data-sets) documentation.
-   This project is for educational and illustrative purposes.
