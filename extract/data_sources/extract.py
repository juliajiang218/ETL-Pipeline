import sqlite3
import csv

# Function to create the SQLite tables
def create_tables(cursor):
    cursor.execute('DROP TABLE IF EXISTS "movie"')
    cursor.execute('DROP TABLE IF EXISTS "produces"')
    cursor.execute('DROP TABLE IF EXISTS "country"')
    cursor.execute('DROP TABLE IF EXISTS "categories"')
    cursor.execute('DROP TABLE IF EXISTS "genre"')
    cursor.execute('DROP TABLE IF EXISTS "speaks"')
    cursor.execute('DROP TABLE IF EXISTS "language"')
    cursor.execute('DROP TABLE IF EXISTS "actor"')
    cursor.execute('DROP TABLE IF EXISTS "role"')
    cursor.execute('DROP TABLE IF EXISTS "user_rating"')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "movie" (
            "id" INTEGER PRIMARY KEY,
            "orig_title" VARCHAR(200) NOT NULL,
            "eng_title" VARCHAR(200) NOT NULL,
            "releasedate" DATE NOT NULL,
            "runtime" INTEGER NOT NULL,
            "homepage" VARCHAR(300),
            "bugt_amt" INTEGER,
            "revenue" INTEGER
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "produces" (
            "id" INTEGER PRIMARY KEY,
            "movie_id" INTEGER,
            "country_code" VARCHAR(2),
            FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
            FOREIGN KEY("country_code") REFERENCES "country"("code")
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "country" (
            "code" VARCHAR(2) PRIMARY KEY,
            "name" VARCHAR(200)
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "categories" (
            "id" INTEGER,
            "movie_id" INTEGER,
            "genre_id" INTEGER,
            PRIMARY KEY("id"),
            FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
            FOREIGN KEY("genre_id") REFERENCES "genre"("id")
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "genre" (
            "id" INTEGER PRIMARY KEY,
            "name" VARCHAR(200)
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "speaks" (
            "id" INTEGER PRIMARY KEY,
            "movie_id" INTEGER,
            "language_code" VARCHAR(2),
            "spoken_len" VARCHAR(200),
            FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
            FOREIGN KEY("language_code") REFERENCES "language"("code")
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "language" (
            "code" VARCHAR(2) PRIMARY KEY,
            "endonym" VARCHAR(200)
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "actor" (
            "id" VARCHAR(200) PRIMARY KEY,
            "name" VARCHAR(200),
            "gender" INTEGER,
            "res_country" VARCHAR(200)
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "role" (
            "id" INTEGER PRIMARY KEY,
            "character" VARCHAR(200),
            "actor_id" VARCHAR(200),
            "movie_id" INTEGER,
            FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
            FOREIGN KEY("actor_id") REFERENCES "actor"("id")
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "user_rating" (
            "id" INTEGER PRIMARY KEY,
            "movie_id" INTEGER,
            "userID" INTEGER,
            "date" DATE,
            "score" FLOAT,
            FOREIGN KEY("movie_id") REFERENCES "movie"("id")
        );
    ''')

# Function to load data from CSV file into the movie, produces, country, categories, genre, and speaks tables
def load_movies_data(cursor, csv_file):
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        genre_id = 0  # For assigning IDs to the categories
        produces_id = 0  # For assigning IDs to the produces table
        speaks_id = 0  # For assigning IDs to the speaks table

        for row in reader:
            # Insert into the movie table
            cursor.execute('''
                INSERT INTO "movie" ("id", "orig_title", "eng_title", "bugt_amt", "revenue", "homepage", "runtime", "releasedate")
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            ''', (row['MovieID'], row['OriginalTitle'], row['EnglishTitle'], row['Budget'], row['Revenue'], row['Homepage'], row['Runtime'], row['ReleaseDate']))

             # Handle genres (many-to-many relationship)
            if row['Genres']:
                genres = row['Genres'].split('|')  # Split genres by '|'
                for genre in genres:
                    # Insert genre into genre table (ignore if exists)
                    cursor.execute('''
                        INSERT OR IGNORE INTO "genre" ("name") VALUES (?);
                    ''', (genre,))

                    # Get genre id
                    cursor.execute('SELECT id FROM "genre" WHERE name = ?', (genre,))
                    genre_id = cursor.fetchone()[0]

                    # Insert into categories table (junction table)
                    cursor.execute('''
                        INSERT INTO "categories" ("movie_id", "genre_id")
                        VALUES (?, ?);
                    ''', (row['MovieID'], genre_id))

            # Handle production countries
            if row['ProductionCountries']:
                countries = row['ProductionCountries'].split('|')  # Split countries by '|'
                for country in countries:
                    country_code, country_name = country.split('-')
                    # Insert country into the country table if it doesn't exist
                    cursor.execute('''
                        INSERT OR IGNORE INTO "country" ("code", "name")
                        VALUES (?, ?);
                    ''', (country_code, country_name))

                    # Insert into the produces table
                    cursor.execute('''
                        INSERT INTO "produces" ("id", "movie_id", "country_code")
                        VALUES (?, ?, ?);
                    ''', (produces_id, row['MovieID'], country_code))
                    produces_id += 1  # Increment the produces_id for uniqueness

            # Handle spoken languages
            if row['SpokenLanguages']:
                languages = row['SpokenLanguages'].split('|')  # Split languages by '|'
                for language in languages:
                    language_code, endonym = language.split('-')
                    # Insert language into the language table if it doesn't exist
                    cursor.execute('''
                        INSERT OR IGNORE INTO "language" ("code", "endonym")
                        VALUES (?, ?);
                    ''', (language_code, endonym))

                    # Insert into the speaks table
                    cursor.execute('''
                        INSERT INTO "speaks" ("id", "movie_id", "language_code", "spoken_len")
                        VALUES (?, ?, ?, NULL);
                    ''', (speaks_id, row['MovieID'], language_code))
                    speaks_id += 1  # Increment the speaks_id for uniqueness


# Function to load data from CSV file into the actor and role tables
def load_persons_data(cursor, csv_file):
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Insert actor
            cursor.execute('''
                INSERT OR IGNORE INTO "actor" ("id", "name", "gender", "res_country")
                VALUES (?, ?, ?, NULL);
            ''', (row['CastID'], row['Name'], row['Gender']))

            # Insert role
            cursor.execute('''
                INSERT INTO "role" ("id", "character", "actor_id", "movie_id")
                VALUES (NULL, ?, ?, ?);
            ''', (row['Character'], row['CastID'], row['MovieID']))

# Function to load data from CSV file into the user_rating table
def load_ratings_data(cursor, csv_file):
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cursor.execute('''
                INSERT INTO "user_rating" ("id", "movie_id", "userID", "date", "score")
                VALUES (NULL, ?, ?, ?, ?);
            ''', (row['MovieID'], row['UserID'], row['Date'], row['Rating']))

# Check if tables exist
def check_tables(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in the database:")
    for table in tables:
        print(table[0])

# Check row counts for each table
def check_row_counts(cursor):
    tables = ["movie", "produces", "country", "categories", "genre", "speaks", "language", "actor", "role", "user_rating"]
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM {table};')
        count = cursor.fetchone()[0]
        print(f"{table}: {count} rows inserted.")

# Sample query to fetch data from the movie table
def fetch_sample_data(cursor):
    cursor.execute('SELECT * FROM "categories" LIMIT 5;')
    movies = cursor.fetchall()
    print("Sample data from the categories table:")
    for movie in movies:
        print(movie)

# Main function to set up database, create tables, and load data
def main():
    # Connect to the SQLite database (or create it if it doesn't exist)
    con = sqlite3.connect("movies.db")
    cur = con.cursor()

    # Create tables based on the schema
    create_tables(cur)

    # Load data from CSV files
    load_movies_data(cur, "movies.csv")
    load_persons_data(cur, "persons.csv")
    load_ratings_data(cur, "ratings.csv")

    # Check if tables are created
    check_tables(cur)

    # Check row counts
    check_row_counts(cur)

    # Fetch sample data
    fetch_sample_data(cur)

    # Commit the transaction
    con.commit()

    # Close the database connection
    con.close()

if __name__ == "__main__":
    main()
