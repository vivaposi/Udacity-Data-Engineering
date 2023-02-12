import os
import configparser
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, hour, weekofyear, date_format
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl
from pyspark.sql.types import StringType as Str, IntegerType as Int, DateType as Dat, TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get('AWS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS', 'AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    Creates a new Spark session with the specified configuration
    
    :return spark: Spark session
    """
    
    spark = SparkSession \
            .builder \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
            .getOrCreate()
    
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Loads the song_data from AWS S3 (input_data) and extracts the songs and artist tables
    and then loaded the processed data back to S3 (output_data)
    
    :param spark: Spark Session object
    :param input_data: Location (AWS S3 path) of songs metadata (song_data) JSON files
    :param output_data: Location (AWS S3 path) where dimensional tables will be stored in parquet format 
    """
    
    # Get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
            
    songSchema = R([
        Fld("artist_id", Str()),
        Fld("artist_latitude", Dbl()),
        Fld("artist_location", Str()),
        Fld("artist_longitude", Dbl()),
        Fld("artist_name", Str()),
        Fld("duration", Dbl()),
        Fld("num_songs", Int()),
        Fld("title", Str()),
        Fld("year", Int()),
    ])
    
    # Read song data file
    print("Reading song_data files from S3")
    df = spark.read.json(song_data, mode = 'PERMISSIVE', schema=songSchema)
    print("Read completed")
    
    # Extract columns to create songs table
    songs_table = df.select("song_id",
                            "title",
                            "artist_id",
                            "year",
                            "duration").dropDuplicates(["song_id"]) \
                    .withColumn("song_id", monotonically_increasing_id())

    print("Writing Songs table to S3 bucket")
    # Write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + "songs/", mode="overwrite", partitionBy=["year","artist_id"])
    print("Write Completed")
    
    # Extract columns to create artists table
    artists_table = df.select("artist_id",
                              "artist_name",
                              "artist_location",
                              "artist_latitude",
                              "artist_longitude").dropDuplicates(["artist_id"])

    print("Writing Artists table to S3 after processing")
    # Write artists table to parquet files
    artists_table.write.parquet(output_data + "artists/", mode="overwrite")
    print("Write Completed")
    

def process_log_data(spark, input_data, output_data):
    """
    Loads the log_data from AWS S3 (input_data) and extracts the songs and artist tables
    and then loaded the processed data back to S3 (output_data)
    
    :param spark: Spark Session object
    :param input_data: Location (AWS S3 path) of songs metadata (song_data) JSON files
    :param output_data: Location (AWS S3 path) where dimensional tables will be stored in parquet format             
    """
    
    # Get filepath to log data file
    log_data = input_data + 'log_data/*/*/*.json'

    # Read log data file
    print("Reading log_data files from S3")
    df = spark.read.json(log_data)
    print("Read completed")
    
    # Filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # Extract columns for users table  
    users_table = df.selectExpr("userId as user_id",
                                "firstName as first_name",
                                "lastName as last_name",
                                "gender",
                                "level").dropDuplicates(["user_id"]) 

    # Write users table to parquet files
    print("Writing Users table to S3 bucket")  
    users_table.write.parquet(os.path.join(output_data, "users/") , mode="overwrite")
    print("Write Completed")
    
    # Create timestamp column from original timestamp column
    get_timestamp = udf(lambda x : datetime.utcfromtimestamp(int(x) / 1000), TimestampType())
    df = df.withColumn("start_time", get_timestamp("ts"))
    
    # Extract columns to create time table
    time_table = df.withColumn("hour", hour("start_time")) \
                   .withColumn("day", dayofmonth("start_time")) \
                   .withColumn("week", weekofyear("start_time")) \
                   .withColumn("month", month("start_time")) \
                   .withColumn("year", year("start_time")) \
                   .withColumn("weekday", dayofweek("start_time")) \
                   .select("ts", "start_time", "hour", "day", "week", "month", "year", "weekday") \
                   .drop_duplicates(["start_time"])

    # Write time table to parquet files partitioned by year and month
    print("Writing Time table to S3 bucket")  
    time_table.write.parquet(os.path.join(output_data, "time_table/"), mode='overwrite', \
                             partitionBy=["year","month"])
    print("Write Completed")
    
    # Read song data for songplays table
    song_data = input_data + "song_data/*/*/*/*.json"
    song_df = spark.read.json(song_data, schema = get_song_schema())

    # extract columns from joined song and log datasets to create
    # songplays table
    song_df.createOrReplaceTempView("song_data")
    df.createOrReplaceTempView("log_data")
    
    songplays_table = spark.sql("""
                                SELECT monotonically_increasing_id() as songplay_id,
                                ld.timestamp as start_time,
                                year(ld.timestamp) as year,
                                month(ld.timestamp) as month,
                                ld.userId as user_id,
                                ld.level as level,
                                sd.song_id as song_id,
                                sd.artist_id as artist_id,
                                ld.sessionId as session_id,
                                ld.location as location,
                                ld.userAgent as user_agent
                                FROM log_data ld
                                JOIN song_data sd
                                ON (ld.song = sd.title
                                AND ld.length = sd.duration
                                AND ld.artist = sd.artist_name)
                                """)

    # Write songplays table to parquet files partitioned by year and month
    print("Writing Songplays table to S3 bucket")  
    songplays_table.write.parquet(os.path.join(output_data, "songplays/"), \
                                  mode="overwrite", partitionBy=["year","month"])
    print("Write Completed")

def test_parquet(spark, output_data):
    """
    Print first row of output data from S3 bucket
    and number of rows.
    
    :param spark: spark session object
    :param output_data: S3 bucket for output data
    """
    songplays_table = spark.read.parquet(output_data + "songplays_table.parquet")
    print("Reading output data and printing row...")
    print(songplays_table.head(1))
    print("Number of rows: {}".format(songplays_table.count()))

def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://sparkify-data-lake-s3-parquet"
    
    print("\n")
    
    print("Processing song_data files")
    process_song_data(spark, input_data, output_data)
    print("Processing completed\n")
    
    print("Processing log_data files")
    process_log_data(spark, input_data, output_data)
    print("Processing completed\n")


if __name__ == "__main__":
    main() os
import configparser
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, hour, weekofyear, date_format
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl
from pyspark.sql.types import StringType as Str, IntegerType as Int, DateType as Dat, TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get('AWS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS', 'AWS_SECRET_ACCESS_KEY')


def create_spark_session():
    """
    Creates a new Spark session with the specified configuration
    
    :return spark: Spark session
    """
    
    spark = SparkSession \
            .builder \
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
            .getOrCreate()
    
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Loads the song_data from AWS S3 (input_data) and extracts the songs and artist tables
    and then loaded the processed data back to S3 (output_data)
    
    :param spark: Spark Session object
    :param input_data: Location (AWS S3 path) of songs metadata (song_data) JSON files
    :param output_data: Location (AWS S3 path) where dimensional tables will be stored in parquet format 
    """
    
    # Get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
            
    songSchema = R([
        Fld("artist_id", Str()),
        Fld("artist_latitude", Dbl()),
        Fld("artist_location", Str()),
        Fld("artist_longitude", Dbl()),
        Fld("artist_name", Str()),
        Fld("duration", Dbl()),
        Fld("num_songs", Int()),
        Fld("title", Str()),
        Fld("year", Int()),
    ])
    
    # Read song data file
    print("Reading song_data files from S3")
    df = spark.read.json(song_data, mode='PERMISSIVE', schema=songSchema, \
                         columnNameOfCorruptRecord='corrupt_record').dropDuplicates()
    print("Read completed")
    
    # Extract columns to create songs table
    songs_table = df.select("song_id",
                            "title",
                            "artist_id",
                            "year",
                            "duration").dropDuplicates(["song_id"]) \
                    .withColumn("song_id", monotonically_increasing_id())

    print("Writing Songs table to S3 bucket")
    # Write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + "songs/", mode="overwrite", partitionBy=["year","artist_id"])
    print("Write Completed")
    
    # Extract columns to create artists table
    artists_table = df.select("artist_id",
                              "artist_name",
                              "artist_location",
                              "artist_latitude",
                              "artist_longitude").dropDuplicates(["artist_id"])

    print("Writing Artists table to S3 after processing")
    # Write artists table to parquet files
    artists_table.write.parquet(output_data + "artists/", mode="overwrite")
    print("Write Completed")
    

def process_log_data(spark, input_data, output_data):
    """
    Loads the log_data from AWS S3 (input_data) and extracts the songs and artist tables
    and then loaded the processed data back to S3 (output_data)
    
    :param spark: Spark Session object
    :param input_data: Location (AWS S3 path) of songs metadata (song_data) JSON files
    :param output_data: Location (AWS S3 path) where dimensional tables will be stored in parquet format             
    """
    
    # Get filepath to log data file
    log_data = input_data + 'log_data/*/*/*.json'

    # Read log data file
    print("Reading log_data files from S3")
    df = spark.read.json(log_data)
    print("Read completed")
    
    # Filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # Extract columns for users table  
    users_table = df.selectExpr("userId as user_id",
                                "firstName as first_name",
                                "lastName as last_name",
                                "gender",
                                "level").dropDuplicates(["user_id"]) 

    # Write users table to parquet files
    print("Writing Users table to S3 bucket")  
    users_table.write.parquet(os.path.join(output_data, "users/") , mode="overwrite")
    print("Write Completed")
    
    # Create timestamp column from original timestamp column
    get_timestamp = udf(lambda x : datetime.utcfromtimestamp(int(x) / 1000), TimestampType())
    df = df.withColumn("start_time", get_timestamp("ts"))
    
    # Extract columns to create time table
    time_table = df.withColumn("hour", hour("start_time")) \
                   .withColumn("day", dayofmonth("start_time")) \
                   .withColumn("week", weekofyear("start_time")) \
                   .withColumn("month", month("start_time")) \
                   .withColumn("year", year("start_time")) \
                   .withColumn("weekday", dayofweek("start_time")) \
                   .select("ts", "start_time", "hour", "day", "week", "month", "year", "weekday") \
                   .drop_duplicates(["start_time"])

    # Write time table to parquet files partitioned by year and month
    print("Writing Time table to S3 bucket")  
    time_table.write.parquet(os.path.join(output_data, "time_table/"), mode='overwrite', \
                             partitionBy=["year","month"])
    print("Write Completed")
    
    # Read song data for songplays table
    song_data = input_data + "song_data/*/*/*/*.json"
    song_df = spark.read.json(song_data, schema = get_song_schema())

    # extract columns from joined song and log datasets to create
    # songplays table
    song_df.createOrReplaceTempView("song_data")
    df.createOrReplaceTempView("log_data")
    
    songplays_table = spark.sql("""
                                SELECT monotonically_increasing_id() as songplay_id,
                                ld.timestamp as start_time,
                                year(ld.timestamp) as year,
                                month(ld.timestamp) as month,
                                ld.userId as user_id,
                                ld.level as level,
                                sd.song_id as song_id,
                                sd.artist_id as artist_id,
                                ld.sessionId as session_id,
                                ld.location as location,
                                ld.userAgent as user_agent
                                FROM log_data ld
                                JOIN song_data sd
                                ON (ld.song = sd.title
                                AND ld.length = sd.duration
                                AND ld.artist = sd.artist_name)
                                """)

    # Write songplays table to parquet files partitioned by year and month
    print("Writing Songplays table to S3 bucket")  
    songplays_table.write.parquet(os.path.join(output_data, "songplays/"), \
                                  mode="overwrite", partitionBy=["year","month"])
    print("Write Completed")

def test_parquet(spark, output_data):
    """
    Print first row of output data from S3 bucket
    and number of rows.
    
    :param spark: spark session object
    :param output_data: S3 bucket for output data
    """
    songplays_table = spark.read.parquet(output_data + "songplays_table.parquet")
    print("Reading output data and printing row...")
    print(songplays_table.head(1))
    print("Number of rows: {}".format(songplays_table.count()))

def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://sparkify-data-lake-s3-parquet"
    
    print("\n")
    
    print("Processing song_data files")
    process_song_data(spark, input_data, output_data)
    print("Processing completed\n")
    
    print("Processing log_data files")
    process_log_data(spark, input_data, output_data)
    print("Processing completed\n")


if __name__ == "__main__":
    main()
