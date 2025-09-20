-- Creating the "movie" table
CREATE TABLE "movie" (
    "id" INTEGER PRIMARY KEY,
    "orig_title" VARCHAR(200),
    "eng_title" VARCHAR(200),
    "releasedate" DATE,
    "runtime" INTEGER,
    "homepage" VARCHAR(300),
    "bugt_amt" DECIMAL(15, 2),
    "bugt_type" VARCHAR(6) AS (
        CASE
            WHEN "bugt_amt" < 1000000 THEN 'low'       -- Budget less than 1 million
            WHEN "bugt_amt" < 10000000 THEN 'medium'   -- Budget less than 10 million
            ELSE 'high'                                -- Budget 10 million or more
        END
    ) STORED,
    "revenue" DECIMAL(15, 2),
    "votes" INTEGER,
    "cert" VARCHAR(10),
    "img_url" VARCHAR(200),
    "overview" TEXT
);

-- Creating the "produces" table
CREATE TABLE "produces" (
    "id" INTEGER PRIMARY KEY,
    "movie_id" INTEGER,
    "country_code" CHAR(2),
    FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
    FOREIGN KEY("country_code") REFERENCES "country"("code")
);

-- Creating the "country" table
CREATE TABLE "country" (
    "code" CHAR(2) PRIMARY KEY,
    "name" VARCHAR(200) UNIQUE
);

-- Creating the "categories" table
CREATE TABLE "categories" (
    "id" INTEGER PRIMARY KEY,
    "movie_id" INTEGER,
    "genre_id" INTEGER,
    FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
    FOREIGN KEY("genre_id") REFERENCES "genre"("id")
);

-- Creating the "genre" table
CREATE TABLE "genre" (
    "id" INTEGER PRIMARY KEY,
    "name" VARCHAR(200) UNIQUE
);

-- Creating the "speaks" table
CREATE TABLE "speaks" (
    "id" INTEGER PRIMARY KEY,
    "movie_id" INTEGER,
    "lan_code" CHAR(2),
    FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
    FOREIGN KEY("lan_code") REFERENCES "language"("code")
);

-- Creating the "language" table
CREATE TABLE "language" (
    "code" CHAR(2) PRIMARY KEY,
    "endonym" VARCHAR(200)
);

-- Creating the "actor" table
CREATE TABLE "actor" (
    "id" CHAR(24) PRIMARY KEY,
    "name" VARCHAR(200),
    "gender" INTEGER CHECK ("gender" BETWEEN 1 AND 3),
    "res_country" VARCHAR(200) UNIQUE
);

-- Creating the "role" table
CREATE TABLE "role" (
    "id" INTEGER PRIMARY KEY,
    "character" VARCHAR(200),
    "actor_id" CHAR(24),
    "movie_id" INTEGER,
    FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
    FOREIGN KEY("actor_id") REFERENCES "actor"("id")
);

-- Creating the "user_rating" table
CREATE TABLE "user_rating" (
    "id" INTEGER PRIMARY KEY,
    "movie_id" INTEGER,
    "userID" INTEGER,
    "date" DATE,
    "score" FLOAT CHECK ("score" BETWEEN 1.0 AND 10.0),
    FOREIGN KEY("movie_id") REFERENCES "movie"("id")
);

-- Creating the "director" table
CREATE TABLE "director" (
    "id" INTEGER PRIMARY KEY,
    "first_name" VARCHAR(200),
    "last_name" VARCHAR(200),
    "movie_id" INTEGER,
    FOREIGN KEY("movie_id") REFERENCES "movie"("id")
);

-- Creating indexes
-- Ensures optimized search for user ratings (covering index)
create index "id_ur" on "user_rating"("userID", "movie_id");

-- Optimized search for movie genres
create index "movieid_c" on "categories"("movie_id");

--index for movie_id and english_title:
create index "movie_id_title" on "movie"("id", "eng_title");

--index:
create index "userID_movie_id_scor" on "user_rating"("userID", "movie_id", "score");

-- Creating a view for movie IDs and genre names
create view "movieid_name" as
select c.movie_id, g.name
from categories c
join genre g on c.genre_id = g.id;





