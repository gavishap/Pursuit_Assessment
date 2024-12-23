# High-Value Link Scraper with API

An intelligent system for scraping, ranking, and storing high-value links with a REST API interface.

## Features

- Web scraping with intelligent link prioritization
- Machine learning-based link ranking system
- RESTful API for data access
- Scalable database storage
- Comprehensive documentation

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Initialize the database:

```bash
python scripts/init_db.py
```

5. Start the API server:

```bash
uvicorn app.main:app --reload
```

## Project Structure

```
├── app/
│   ├── api/            # API endpoints
│   ├── core/           # Core application code
│   ├── models/         # Database models
│   └── services/       # Business logic
├── scripts/            # Utility scripts
├── tests/             # Test suite
└── config/            # Configuration files
```

## API Documentation

The API documentation is available at `/docs` when running the server.

### Key Endpoints

- `GET /links` - Retrieve scraped links
- `POST /links` - Add new links
- `DELETE /links/{id}` - Remove a link
- `GET /health` - API health check

## Configuration

Key configuration options in `.env`:

- `OPENAI_API_KEY` - OpenAI API key for link ranking
- `DATABASE_URL` - Database connection string
- `SCRAPER_KEYWORDS` - Priority keywords for scraping
- `SCRAPER_BATCH_SIZE` - Number of pages to scrape in parallel

## Testing

Run the test suite:

```bash
pytest
```

## License

MIT License
