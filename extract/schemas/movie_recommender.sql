CREATE TABLE "movie" (
    "id" INTEGER,
    "orig_title" VARCHAR(200) NOT NULL,
    "eng_title" VARCHAR(200) NOT NULL,
    "releasedate" DATE NOT NULL,
    "runtime" INTEGER NOT NULL,
    "homepage" VARCHAR(300),
    "bugt_amt" INTEGER,
    -- "bugt_type" ENUM("low","medium","high"),
    "revenue" INTEGER,
    PRIMARY KEY("id")
);

CREATE TABLE "produces" (
    "id" INTEGER,
    "movie_id" INTEGER,
    "country_code" VARCHAR(2),
    PRIMARY KEY("id"),
    FOREIGN KEY("country_code") REFERENCES "country"("code"),
    FOREIGN KEY("movie_id") REFERENCES "movie"("id")
);

CREATE TABLE "country" (
    "code" VARCHAR(2),
    "name" VARCHAR(200),
    PRIMARY KEY("code")
);

CREATE TABLE "categories" (
    "id" INTEGER,
    "movie_id" INTEGER,
    "genre_id" INTEGER,
    PRIMARY KEY("id"),
    FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
    FOREIGN KEY("genre_id") REFERENCES "genre"("id")
);

CREATE TABLE "genre" (
    "id" INTEGER,
    "name" VARCHAR(200),
    PRIMARY KEY("id")
);

CREATE TABLE "speaks" (
    "id" INTEGER,
    "movie_id" INTEGER,
    "language_code" VARCHAR(2),
    "spoken_len" VARCHAR(200),
    PRIMARY KEY("id"),
    FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
    FOREIGN KEY("language_code") REFERENCES "language"("code")
);

CREATE TABLE "language" (
    "code" VARCHAR(2),
    "movie_id" INTEGER,
    "endonym" VARCHAR(200),
    PRIMARY KEY("code"),
    FOREIGN KEY("movie_id") REFERENCES "movie"("id")
);

CREATE TABLE "actor" (
    "id" VARCHAR(200),
    "name" VARCHAR(200),
    "gender" INTEGER,
    "res_country" VARCHAR(200) UNIQUE,
    PRIMARY KEY("id")
);

CREATE TABLE "role" (
    "id" INTEGER,
    "character" VARCHAR(200),
    "actor_id" VARCHAR(200),
    "movie_id" INTEGER,
    PRIMARY KEY("id"),
    FOREIGN KEY("movie_id") REFERENCES "movie"("id"),
    FOREIGN KEY("actor_id") REFERENCES "actor"("id")
);

CREATE TABLE "user_rating" (
    "id" INTEGER,
    "movie_id" INTEGER,
    "userID" INTEGER,
    "date" DATE,
    "score" FLOAT,
    PRIMARY KEY("id"),
    FOREIGN KEY("movie_id") REFERENCES "movie"("id")
);