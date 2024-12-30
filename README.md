# High-Value Link Scraper and Ranker

An intelligent system that scrapes web pages and ranks links based on relevance using multiple ranking methods, including semantic similarity, OpenAI's GPT, advanced NLP, and deep learning approaches.

## Features

- Web scraping with retry logic and error handling
- Multiple ranking systems working in ensemble
- SQLite database for storing links and scores
- FastAPI-based REST API
- Configurable relevance thresholds
- Keyword-based filtering and context

## Architecture

### Ranking Systems

1. **Semantic Ranker**

   - Uses sentence-transformers (all-MiniLM-L6-v2)
   - Computes semantic similarity between URL text and context
   - Fast and lightweight baseline ranking

2. **OpenAI Ranker**

   - Uses GPT-3.5-turbo for intelligent URL analysis
   - Considers multiple factors:
     - Content Match (40%)
     - Source Authority (30%)
     - URL Structure (20%)
     - General Quality (10%)
   - Provides detailed scoring with confidence levels

3. **Advanced NLP Ranker**

   - Uses NLTK and spaCy for linguistic analysis
   - Features:
     - Named Entity Recognition
     - Topic Modeling
     - Keyword Extraction
     - URL Structure Analysis

4. **Deep Learning Ranker**

   The deep learning ranker evolved through multiple iterations to improve URL relevance scoring. A key challenge was that our training data (`ranked_url_classes.csv`) contained rankings from 1-10 based on page popularity/importance rather than true topical relevance, which affected model performance.

   **Training Data Structure:**

   - 35,632 URLs with rankings
   - Rankings from 1-10 (higher = more important)
   - 12 unique topics
   - ~31,983 unique URLs
   - Mean rank: 5.49, Standard deviation: 2.87
   - Challenge: Rankings reflect page importance rather than topical relevance

   **V1 (Initial Version)**

   - Architecture:
     - Basic LSTM network for URL text processing
     - Embedding layer (100 dimensions)
     - Bidirectional LSTM (128 units)
     - Dense layers (64 → 32 → 1)
     - Binary classification (relevant/not relevant)
   - Training:
     - 5 epochs
     - Binary cross-entropy loss
     - Adam optimizer
   - Issues:
     - Binary output too simplistic
     - No context consideration
     - Poor URL text preprocessing
     - High variance in predictions

   **V2 (Improved)**

   - Architecture:
     - Added attention mechanism
     - Embedding layer (200 dimensions)
     - Bidirectional LSTM (256 units)
     - Self-attention layer
     - Dense layers (128 → 64 → 1)
     - Regression output (0-1 score)
   - Improvements:
     - Better URL preprocessing:
       - Path segmentation
       - Query parameter handling
       - Special token handling
     - Switched to regression for finer-grained scoring
     - Added dropout (0.3) for regularization
   - Training:
     - 8 epochs
     - MSE loss
     - Adam optimizer with learning rate scheduling
   - Issues:
     - Limited context understanding
     - Inconsistent scores across similar URLs
     - High computational overhead

   **V3 (Current)**

   - Architecture:
     - Base: MPNet transformer
     - Separate encoders:
       - URL encoder (512 hidden size)
       - Topic encoder (512 hidden size)
     - Multi-head attention (8 heads)
     - Feature fusion layer:
       - Concatenation
       - Projection (1024 → 512)
       - Layer normalization
     - Classification head:
       - Dense (512 → 256 → 64 → 1)
       - Dropout (0.2)
   - Advanced Preprocessing:
     - Domain extraction and cleaning
     - TLD handling
     - Date pattern recognition
     - Numeric token normalization
     - Special tokens for URL types
   - Training:
     - 10 epochs
     - Combined loss:
       - MSE loss
       - L1 regularization (λ=0.01)
     - Cosine learning rate schedule
     - Validation loss: 0.1015
   - Improvements:
     - Better context integration
     - More stable predictions
     - Faster inference
     - Improved generalization
   - Remaining Challenges:
     - Training data quality (page importance vs relevance)
     - Cold start for new topics
     - Computational requirements
     - Domain adaptation needs

The evolution of the deep learning ranker shows a progression from simple sequence processing to sophisticated context-aware ranking. However, the fundamental challenge remains that our training data is based on page importance rather than true topical relevance. This means the model might learn to identify "important" pages rather than truly relevant ones for a given topic.

### Technologies Used

- **Backend Framework**: FastAPI
- **Database**: SQLite with SQLAlchemy ORM
- **Machine Learning**:
  - PyTorch
  - Sentence-Transformers
  - NLTK
  - spaCy
  - OpenAI API
- **Web Scraping**:
  - aiohttp
  - BeautifulSoup4
- **Development Tools**:
  - Python 3.8+
  - uvicorn
  - pydantic

## Getting Started

1. **Installation**

   ```bash
   # Clone the repository
   git clone [repository-url]
   cd [repository-name]

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Environment Setup**

   ```bash
   # Create .env file
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Running the API**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Usage

Access the interactive API documentation at `http://localhost:8000/docs`

### Key Endpoints

1. **Scrape and Rank Links**

   ```json
   POST /scrape
   {
     "url": "https://docs.python.org/3/",
     "min_relevance": 0.5,
     "max_links": 50,
     "keywords": [
       "machine learning",
       "data science",
       "neural networks"
     ]
   }
   ```

2. **Retrieve Ranked Links**

   ```
   GET /links?base_url={url}&min_relevance=0.5&keywords=machine learning,neural networks&sort_by=ensemble
   ```

   Query Parameters:

   - `base_url`: Source URL to filter by
   - `min_relevance`: Minimum score threshold (0.0-1.0)
   - `content_type`: Filter by content type
   - `keywords`: Comma-separated keywords
   - `sort_by`: Sorting method (ensemble, semantic, openai, nlp, deep_learning, date)
   - `skip`: Pagination offset
   - `limit`: Results per page

3. **Delete Link**
   ```
   DELETE /links/{link_id}
   ```

## Ensemble Scoring

The final relevance score is calculated as an average of all four ranking systems:

```python
ensemble_score = (semantic_score + openai_score + nlp_score + deep_score) / 4
```

Each ranking system contributes equally to the final score, providing a balanced evaluation that combines:

- Semantic similarity (semantic_score)
- Contextual understanding (openai_score)
- Linguistic features (nlp_score)
- Learned patterns (deep_score)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]
