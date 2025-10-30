# Transport Analytics

Comprehensive transport analytics and reporting tool for Perrys that generates interactive HTML dashboards with advanced visualizations.

## Features

- 📊 Executive summary with KPIs
- 💰 Cost flow analysis (Sankey diagrams)
- 🗺️ Delivery heatmap with geocoding
- 📈 Time-based trends and seasonal patterns
- 🚚 Provider performance metrics
- 👥 Internal driver shift analysis (by day of week)
- ⚡ Speed of movement tracking
- 📏 Distance band analysis
- 💵 Budget vs actual comparison
- ⚠️ Cost outlier detection

## Installation
```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/transport-analytics.git
cd transport-analytics

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python perrys_transport_analytics.py
```

The script will:
1. Fetch data from the configured Google Sheet
2. Geocode delivery postcodes (cached for performance)
3. Calculate comprehensive analytics
4. Generate `perrys_transport_infographic.html`

## Configuration

Edit these variables at the top of `perrys_transport_analytics.py`:
```python
# Date filtering (optional)
FILTER_START_DATE = None  # e.g., '2024-01-01'
FILTER_END_DATE = None     # e.g., '2024-12-31'

# Data source
GOOGLE_SHEET_CSV_URL = "your-google-sheet-url"
```

## Output

Generates an interactive HTML dashboard (`perrys_transport_infographic.html`) with:
- Interactive charts (Chart.js)
- Cost flow diagram (Plotly Sankey)
- Delivery heatmap (Leaflet + OpenStreetMap)
- On-time performance heatmap
- Comprehensive data tables

## Geocoding Cache

The script uses Nominatim (OpenStreetMap) for geocoding with a 1 req/sec rate limit. Results are cached in `geocode_cache.json` to speed up subsequent runs.

## Requirements

- Python 3.8+
- pandas
- requests
- numpy
- geopy

## License

MIT License
