# ETL Pipeline: Movie Data Processing and Aggregation

General purpose processing and aggregating of movie data into the shape and format needed for content-based recommendations.

## Project Overview

This ETL pipeline extracts movie data from multiple sources, transforms and cleans the data according to various requirements, and loads it into a SQLite database with optimized schema for efficient querying and content-based recommendations.

## Data Sources

**Extract data from multiple data sources:**
- [TU Graz Movie Dataset](https://github.com/tugraz-isds/datasets/tree/master/movies)
- [IMDb Dataset of Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)

## Curated Entity Relation Diagram
![hw4updated_ERD](https://github.com/user-attachments/assets/1c92c677-641d-4138-8c7b-01d8b112fa17)

## Pipeline Components

### Extract Phase
- **Multi-format data ingestion** from CSV files and external datasets
- **Data source validation** ensuring data integrity from extraction
- **Schema validation** with comprehensive data quality checks

### Transform Phase
- **Data cleaning and normalization** standardizing formats across sources
- **Data validation and rejection** filtering malformatted data points based on entity requirements
- **Join operations** combining data from multiple sources
- **Standardize value ranges** ensuring consistency across datasets
- **Database schema normalization** following relational database principles

### Load Phase
- **SQLite database population** using designed database schemas and scripts
- **Integrity constraints** review and implementation
- **Indexes and views** creation to support efficient content-based recommendations
- **Performance testing** through multiple query executions

## Database Design

### Logical Data Models and Schema
Created scripts implementing logical data models and database schemas based on the datasets:

- **Entity Relationship Design**: Comprehensive ERD diagram enumerating data entity relationships
- **Normalized Schema**: Following database normalization principles for optimal data storage
- **Data Integrity**: Appropriate integrity constraints ensuring data consistency
- **Performance Optimization**: Indexes and views designed for efficient content-based recommendations

### Data Validation and Quality Control
Given various requirements on data entity specifications, refined database schemas ensure the process rejects malformatted data points:

- **Schema Validation**: Strict validation against expected data formats
- **Data Type Enforcement**: Ensuring proper data types for all fields
- **Constraint Checking**: Business rule validation for data consistency
- **Quality Metrics**: Comprehensive reporting on data quality issues

## Technical Architecture

```
ETL Pipeline/
├── extract/                    # Data extraction from multiple sources
│   ├── data_sources/          # Data extraction scripts
│   │   └── extract.py        # Main extraction script
│   ├── config/               # Data source configurations
│   │   └── data_sources.yaml # Data source configuration
│   └── schemas/              # Database schemas
│       ├── movies.sql        # Movies table schema
│       └── movie_recommender.sql # Movie recommender schema
├── transform/                 # Data transformation and recommendation engine
│   ├── content-based-recommendation/ # Content-based recommendation system
│   │   ├── content_Rec.py    # Content-based recommendation logic
│   │   └── main.py          # Content-based recommendation main script
│   ├── db_setup.py          # Database setup and initialization
│   ├── db_schema.sql        # Database schema definitions
│   └── recommendation.py    # Collaborative filtering recommendation engine
├── load/                     # Movie recommendation interface
│   └── main.py              # Command-line interface for movie recommendations
├── config/                  # Pipeline configurations and settings
│   └── pipeline_config.yaml # Main pipeline configuration
└── logs/                    # Logging directory
    └── result.txt          # Recommendation results log
```

## Usage

### Movie Recommendation System
The main interface for generating movie recommendations:

```bash
# Generate 3 recommendations for user ID 1
cd load
python main.py 1

# Generate 5 recommendations for user ID 10
python main.py 10 --top_n 5

# Run with profiling enabled
python main.py 1 --profile
```

### ETL Pipeline Components

#### Data Extraction
```bash
cd extract/data_sources
python extract.py
```

#### Database Setup
```bash
cd transform
python db_setup.py
```

#### Content-Based Recommendations
```bash
cd transform/content-based-recommendation
python main.py
```

## Data Quality and Testing

The pipeline includes comprehensive data validation and testing:

- **Multiple query testing** to verify database design effectiveness
- **Data integrity verification** through constraint checking
- **Performance benchmarking** for content-based recommendation queries
- **Quality reporting** with detailed statistics on data processing
