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
    cursor.execute('DROP TABLE IF EXISTS "director"')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "movie" (
            "id" INTEGER PRIMARY KEY,
            "orig_title" VARCHAR(200),
            "eng_title" VARCHAR(200),
            "releasedate" DATE,
            "runtime" INTEGER,
            "homepage" VARCHAR(300),
            "bugt_amt" DECIMAL(15, 2),
            "bugt_type" VARCHAR(6) AS (
                CASE
                    WHEN "bugt_amt" < 1000000 THEN 'low'
                    WHEN "bugt_amt" < 10000000 THEN 'medium'
                    ELSE 'high'
                END
            ) STORED,
            "revenue" DECIMAL(15, 2),
            "votes" INTEGER,
            "cert" VARCHAR(10),
            "img_url" VARCHAR(200),
            "overview" TEXT
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "produces" (
            "id" INTEGER PRIMARY KEY,
            "movie_id" INTEGER,
            "country_code" CHAR(2),
            FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
            FOREIGN KEY("country_code") REFERENCES "country"("code")
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "country" (
            "code" CHAR(2) PRIMARY KEY,
            "name" VARCHAR(200) UNIQUE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "categories" (
            "id" INTEGER PRIMARY KEY,
            "movie_id" INTEGER,
            "genre_id" INTEGER,
            FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
            FOREIGN KEY("genre_id") REFERENCES "genre"("id")
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "genre" (
            "id" INTEGER PRIMARY KEY,
            "name" VARCHAR(200) UNIQUE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "speaks" (
            "id" INTEGER PRIMARY KEY,
            "movie_id" INTEGER,
            "lan_code" CHAR(2),
            FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
            FOREIGN KEY("lan_code") REFERENCES "language"("code")
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "language" (
            "code" CHAR(2) PRIMARY KEY,
            "endonym" VARCHAR(200)
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "actor" (
            "id" CHAR(24) PRIMARY KEY,
            "name" VARCHAR(200),
            "gender" INTEGER CHECK ("gender" BETWEEN 1 AND 3),
            "res_country" VARCHAR(200) UNIQUE
        );
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS "role" (
            "id" INTEGER PRIMARY KEY,
            "character" VARCHAR(200),
            "actor_id" CHAR(24),
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
            "score" FLOAT CHECK ("score" BETWEEN 1.0 AND 10.0),
            FOREIGN KEY("movie_id") REFERENCES "movie"("id")
        );
    ''')

    cursor.execute('''
        CREATE TABLE "director" (
            "id" INTEGER PRIMARY KEY,
            "first_name" VARCHAR(200),
            "last_name" VARCHAR(200),
            "movie_id" INTEGER,
            FOREIGN KEY("movie_id") REFERENCES "movie"("id")
        );
    ''')

# Function to load data from CSV file into the movie, produces, country, categories, genre, and speaks tables
def load_movies_data(cursor, csv_file):
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        produces_id = 0
        speaks_id = 0

        for row in reader:
            # Insert into the movie table
            cursor.execute('''
                INSERT INTO "movie" ("id", "orig_title", "eng_title", "bugt_amt", "revenue", "homepage", "runtime", "releasedate")
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            ''', (row['MovieID'], row['OriginalTitle'], row['EnglishTitle'], row['Budget'], row['Revenue'], row['Homepage'], row['Runtime'], row['ReleaseDate']))

            # Handle genres (many-to-many relationship)
            if row['Genres']:
                genres = row['Genres'].split('|')
                for genre in genres:
                    cursor.execute('''
                        INSERT OR IGNORE INTO "genre" ("name") VALUES (?);
                    ''', (genre,))

                    cursor.execute('SELECT id FROM "genre" WHERE name = ?', (genre,))
                    genre_id = cursor.fetchone()[0]

                    cursor.execute('''
                        INSERT INTO "categories" ("movie_id", "genre_id")
                        VALUES (?, ?);
                    ''', (row['MovieID'], genre_id))

            # Handle production countries
            if row['ProductionCountries']:
                countries = row['ProductionCountries'].split('|')
                for country in countries:
                    country_code, country_name = country.split('-')
                    cursor.execute('''
                        INSERT OR IGNORE INTO "country" ("code", "name")
                        VALUES (?, ?);
                    ''', (country_code, country_name))

                    cursor.execute('''
                        INSERT INTO "produces" ("id", "movie_id", "country_code")
                        VALUES (?, ?, ?);
                    ''', (produces_id, row['MovieID'], country_code))
                    produces_id += 1

            # Handle spoken languages
            if row['SpokenLanguages']:
                languages = row['SpokenLanguages'].split('|')
                for language in languages:
                    language_code, endonym = language.split('-')
                    cursor.execute('''
                        INSERT OR IGNORE INTO "language" ("code", "endonym")
                        VALUES (?, ?);
                    ''', (language_code, endonym))

                    cursor.execute('''
                        INSERT INTO "speaks" ("id", "movie_id", "lan_code")
                        VALUES (?, ?, ?);
                    ''', (speaks_id, row['MovieID'], language_code))
                    speaks_id += 1

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
            try:
                # Check if the 'Rating' field is valid and convert it to a float
                if row['Rating'] and row['Rating'].strip().replace('.', '', 1).isdigit():
                    # Transform the score from 0.5-5.0 scale to 1.0-10.0 scale
                    transformed_score = float(row['Rating']) * 2

                    # Ensure the transformed score is within the valid range before inserting
                    if 1.0 <= transformed_score <= 10.0:
                        cursor.execute('''
                            INSERT INTO "user_rating" ("id", "movie_id", "userID", "date", "score")
                            VALUES (NULL, ?, ?, ?, ?);
                        ''', (row['MovieID'], row['UserID'], row['Date'], transformed_score))
                    else:
                        print(f"Transformed score out of range for row: {row}")
                else:
                    print(f"Invalid rating value for row: {row}")
            
            except ValueError as e:
                # Print an error message if there is an issue with data conversion
                print(f"Error converting rating for row: {row} - {e}")



# Function to load imdb data from csv file into movie, user_rating, director, actor tables
# Function to parse and load IMDb data from a CSV file
def load_imdb_data(cursor, csv_file):
    # Get the current highest movie ID from the database
    cursor.execute('SELECT MAX(id) FROM "movie";')
    max_existing_id = cursor.fetchone()[0] or 0  # If no rows, start from 0
    next_movie_id = max_existing_id + 1

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use next_movie_id for the new entry and increment for the next iteration
            current_movie_id = next_movie_id
            next_movie_id += 1
            # Insert movie data
            cursor.execute('''
                INSERT OR IGNORE INTO "movie" (
                    "id", "orig_title", "eng_title", "releasedate", "runtime", "homepage",
                    "bugt_amt", "revenue", "votes", "cert", "img_url", "overview"
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            ''', (
                current_movie_id,
                row['Series_Title'],  # Original title
                row['Series_Title'],  # English title (same as original)
                row['Released_Year'],  # Release year
                int(row['Runtime'].replace(' min', '')),  # Runtime in minutes (converted to int)
                row['Poster_Link'],  # Homepage URL (assuming poster link serves as this)
                None,  # Budget (not available in the CSV, set to None)
                row['Gross'].replace(',', '') if row['Gross'] else None,  # Revenue (remove commas)
                row['No_of_Votes'],  # Number of votes
                row['Certificate'],  # Certificate
                row['Poster_Link'],  # Image URL
                row['Overview']  # Overview text
            ))

            # Insert director data
            if row.get('Director'):
                director_name = row['Director'].split(' ', 1)
                first_name = director_name[0]
                last_name = director_name[1] if len(director_name) > 1 else ''
                cursor.execute('''
                    INSERT OR IGNORE INTO "director" ("first_name", "last_name", "movie_id")
                    VALUES (?, ?, ?);
                ''', (first_name, last_name, current_movie_id))

            # Insert actor data (assumes columns for starring actors: Star1, Star2, etc.)
            for i in range(1, 5):  # Adjust based on number of actor columns (Star1 to Star4)
                star_column = f'Star{i}'
                if row.get(star_column):
                    actor_name = row[star_column]
                    actor_id = f"{current_movie_id}_{i}"  # Create a unique ID based on movie ID and index
                    cursor.execute('''
                        INSERT OR IGNORE INTO "actor" ("id", "name", "gender", "res_country")
                        VALUES (?, ?, ?, NULL);
                    ''', (actor_id, actor_name, None))

                    # Insert roles (assign a generic character name if not provided)
                    cursor.execute('''
                        INSERT INTO "role" ("character", "actor_id", "movie_id")
                        VALUES (?, ?, ?);
                    ''', (f"Character {i}", actor_id, current_movie_id))

# Check if tables exist
def check_tables(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    names = [name[0] for name in tables]
    print("Tables in the database:", names)
        

# Check row counts for each table
def check_row_counts(cursor):
    tables = ["movie", "produces", "country", "categories", "genre", "speaks", "language", "actor", "role", "user_rating"]
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM {table};')
        count = cursor.fetchone()[0]
        print(f"{table}: {count} rows inserted.")

# Sample query to fetch data from the categories table
def fetch_sample_data(cursor):
    cursor.execute('SELECT * FROM "categories" LIMIT 5;')
    categories = cursor.fetchall()
    print("Sample data from the categories table:")
    for category in categories:
        print(category)

# Main function to set up database, create tables, and load data
def main():
    con = sqlite3.connect("movies.db")
    cur = con.cursor()

    # Create tables based on the schema
    create_tables(cur)

    # Load data from CSV files
    load_movies_data(cur, "Movies.csv")
    load_persons_data(cur, "Persons.csv")
    load_ratings_data(cur, "Ratings.csv")
    load_imdb_data(cur, "imdb_top_1000.csv")


    # Check if tables are created
    check_tables(cur)

    # Check row counts
    check_row_counts(cur)

    # # Fetch sample data
    # fetch_sample_data(cur)

    # Commit the transaction
    con.commit()

    # Close the database connection
    con.close()

if __name__ == "__main__":
    main()