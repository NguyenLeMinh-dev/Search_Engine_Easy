# Food Search Engine (SEG_project)

This project is a food and restaurant search engine built from the ground up, covering the entire process from data crawling, processing, model building, and API deployment. The system focuses on providing accurate and semantically relevant search results for culinary queries.

## Main Features

* **Hybrid Search:** Combines the power of keyword-based search and semantic search to return the most relevant results.
* **API Server:** Provides API endpoints via Flask to serve a front-end application.
* **User Authentication:** Includes basic features like registration (`/register`) and login (`/login`).
* **Personalized Saving:**
    Allows users to save (`/save`) and unsave (`/unsave`) their favorite restaurants.

---

## Search Engine Architecture

The search engine (`utils/system_search_engine.py`) is the core of the project, built on a Two-Stage Hybrid Search architecture to optimize for both accuracy and speed.

### Stage 1: Candidate Retrieval

In this stage, the system quickly gathers a large pool of potential candidates (`CANDIDATE_POOL_SIZE`) from two parallel sources:

1.  **Semantic Search:**
    * **Technology:** Uses **FAISS** (`IndexFlatL2`) on a pre-trained, fine-tuned embedding vector store (`finetuned_item_embeddings.npy`).
    * **Purpose:** To find restaurants with content (descriptions, reviews) that are semantically similar to the user's query, even if the keywords don't match exactly.

2.  **Keyword Search:**
    * **Technology:** Uses **BM25Okapi** (`bm25_index`).
    * **Purpose:** To find restaurants that have a high keyword overlap (e.g., "bánh xèo," "district 1") with the query. This index is built from a processed, accent-stripped data column to improve recall.

The two sets of results from FAISS and BM25 are combined (`np.union1d`) to create a
unique list of candidates.

### Stage 2: Re-ranking

The candidates from Stage 1 are passed through a more sophisticated ranking model to determine the final order:

1.  **Query Encoding:** The user's query is encoded *in real-time* using the fine-tuned Transformer model (SBERT/PhoBERT) to create a representative vector.
2.  **Score Calculation:**
    * **Rerank Score (Semantic):** Calculates the Cosine Similarity between the query vector and the candidate vectors.
    * **BM25 Score (Keyword):** Retrieves the BM25 score from Stage 1 and normalizes it.
3.  **Final Score:** The results are ranked based on a **weighted average score** that balances keyword relevance with semantic meaning:
    `final_score = (RETRIEVAL_WEIGHT * bm25_norm) + (RERANK_WEIGHT * rerank_score)`
4.  **Result:** The system returns the `TOP_K` restaurants with the highest final scores.

---

## Data Pipeline

To build the indices for the search system, the data undergoes a rigorous processing pipeline:

1.  **Data Crawling (`utils/tools_crawl/crawl_data.py`)**
    * Uses **Selenium** to automatically crawl restaurant data from Foody.vn (specifically the Can Tho region).
    * Collects essential information: Name, address, rating, opening hours, price range, images, and user reviews.

2.  **Data Cleaning & Enrichment (`utils/tools_crawl/clean_data.py`)**
    * This is the **most critical step** in preparing data for the hybrid search model.
    * **Semantic Generation:** Automatically generates semantic phrases based on raw data (e.g., `price_min` 50000 -> "Affordable price," `open_hour` 7 -> "Serves breakfast").
    * **Create `text_for_embedding` Column:**
        * **Purpose:** Used for the Semantic model (FAISS, SBERT).
        * **Features:** **ACCENTED** (Vietnamese) data, including name, tags, generated semantics, and truncated reviews.
    * **Create `text_for_bm25` Column:**
        * **Purpose:** Used for the Keyword model (BM25).
        * **Features:** **UN-ACCENTED** data, including all possible keywords (name, tags, original reviews, address) to maximize keyword matching.

3.  **Tokenize & Create Embeddings (`utils/tools_crawl/Tokenize.py`)**
    * Uses `AutoTokenizer` (from `vinai/phobert-base`) to tokenize the `text_for_embedding` column.
    * (Implied Process) These tokens are then fed through the fine-tuned Transformer model to create the vector embeddings, which are saved to `finetuned_item_embeddings.npy` for FAISS to use.

---

## Installation and Setup

### Requirements

* Python 3.x
* (Libraries from `requirements.txt`, e.g., Flask, pandas, numpy, faiss-cpu/gpu, rank-bm25, transformers, torch)

### Running the Application

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NguyenLeMinh-dev/Search_Engine_Easy.git
    cd SEG_project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Initialize Database (Important):**
    The system uses SQLite (`database.db`) to manage users and saved restaurants. You **must** run the `init_db.py` script once to create this database.
    ```bash
    python application/init_db.py
    ```
    If you skip this step, `app.py` will fail, reporting that the database file is not found.

4.  **Run the Flask Server:**
    ```bash
    python application/app.py
    ```
    The server will start and load the Search Engine model. This may take a few seconds.
    Once you see the message "✅ Search Engine đã sẵn sàng nhận yêu cầu." (Search Engine is ready to receive requests), the system is online.
    * The server runs at: `http://0.0.0.0:5000`

---

## API Endpoints

The API is served from `application/app.py`:

### Search System

* **`GET /search`**
    * **Description:** The main search endpoint.
    * **Query Params:**
        * `q` (string): The search query (e.g., "bún bò").
    * **Response (JSON):**
        ```json
        [
            {
                "id": "000123",
                "name": "Bún Bò Huế O Nở",
                "address": "123 ABC street, Ninh Kieu District...",
                "gps": "10.033,105.767",
                "image_src": "[http://127.0.0.1:5000/images/000123.jpg](http://127.0.0.1:5000/images/000123.jpg)",
                "score": 1.2345
                // ... and other columns
            }
        ]
        ```

### Authentication & User

* **`POST /register`**
    * **Body (JSON):** `{ "username": "...", "password": "..." }`
    * **Description:** Registers a new user. The password is hashed before being saved.

* **`POST /login`**
    * **Body (JSON):** `{ "username": "...", "password": "..." }`
    * **Description:** Logs in a user and returns user info upon success.

* **`GET /get_saved`**
    * **Query Params:** `user_id` (int)
    * **Description:** Gets a list of (names of) restaurants that the user has saved.

* **`POST /save`**
    * **Body (JSON):** `{ "user_id": ..., "restaurant_name": "..." }`
    * **Description:** Saves a restaurant to the user's favorites list.

* **`POST /unsave`**
    * **Body (JSON):** `{ "user_id": ..., "restaurant_name": "..." }`
    * **Description:** Removes a restaurant from the user's favorites list.

---

## System Evaluation

The project includes an evaluation script (`eval_system.py`) that uses standard Information Retrieval (IR) metrics.

* **Methodology:**
    1.  Run test queries (e.g., from the `main()` function in `system_search_engine.py`) to generate result files in the `RESULT_DIR`.
    2.  These result files are compared against a "Ground Truth" dataset (`GT_DIR`) that has been manually labeled (with relevance scores 0, 1, 2, 3...).
* **Metrics:**
    * **nDCG@k (Normalized Discounted Cumulative Gain):** Measures ranking quality (highly relevant results should be ranked higher).
    * **mAP@k (Mean Average Precision):** Measures the average precision across the set of queries.
* **How to run:**
    ```bash
    python eval_system.py
    ```
    The script will automatically find matching pairs of (Result - Ground Truth) files and print the average m-nDCG and mAP scores for k-values (10, 50, 100).
