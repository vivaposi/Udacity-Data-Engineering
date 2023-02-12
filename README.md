<!-- ABOUT THE PROJECT -->
#Sparkify Data Lake Project

## About The Project

A music streaming startup called Sparkify wants to analyze the data they've been collecting on songs and user activity on their new app. Sparkify has grown their user base and song database and need to move their data warehouse to a data lake on the cloud. Their data resides in S3 buckets as a compilation of JSON files.

Sparkify is looking for a data engineer to build an ETL pipeline that extracts their data from S3,  processes the data using Spark, and loads the data back into S3 as a set of fact and dimensional tables in Parquet format. This will allow their analytics team to continue finding insights into what songs their users are listening to, where their app is most used, and any other user or song analytics questions they need answered. The role of this project is to create a data lake on cloud (AWS S3) and build ETL pipeline for this process. 

### Project Description

In this project, we will build a data lake on AWS S3 and build an ETL pipeline for a data lake hosted on S3. The data is loaded from S3 and processed into analytics tables using Spark and the processed data is loaded back into S3 in the form of Parquet files.

### Dataset

#### Song Dataset

Songs dataset is a subset of [Million Song Dataset](http://millionsongdataset.com/). Each file in the dataset is in JSON format and contains meta-data about a song and the artist of that song. The dataset is hosted at S3 bucket `s3://udacity-dend/song_data`.

Sample of song meta-data :

```
{"num_songs": 1, "artist_id": "ARJIE2Y1187B994AB7", "artist_latitude": null, "artist_longitude": null, "artist_location": "", "artist_name": "Line Renaud", "song_id": "SOUPIRU12A6D4FA1E1", "title": "Der Kleine Dompfaff", "duration": 152.92036, "year": 0}
```

#### Log Dataset

Logs dataset is generated by [Event Simulator](https://github.com/Interana/eventsim). These log files in JSON format simulate activity logs from a music streaming application based on specified configurations. The dataset is hosted at S3 bucket `s3://udacity-dend/log_data`.

Sample of activity log :

```
{"artist": null, "auth": "Logged In", "firstName": "Walter", "gender": "M", "itemInSession": 0, "lastName": "Frye", "length": null, "level": "free", "location": "San Francisco-Oakland-Hayward, CA", "method": "GET","page": "Home", "registration": 1540919166796.0, "sessionId": 38, "song": null, "status": 200, "ts": 1541105830796, "userAgent": "\"Mozilla\/5.0 (Macintosh; Intel Mac OS X 10_9_4) AppleWebKit\/537.36 (KHTML, like Gecko) Chrome\/36.0.1985.143 Safari\/537.36\"", "userId": "39"}
```

### Prerequisites

These are the prerequisites to run the program.

* python 3.7 (or higher)
* AWS Account
* AWS EMR

### Running the ETL Pipeline

1. Edit the `dl.cfg` config file with corresponding Access Key and Secret Key information

2. Run ETL process 

   ```python
   python etl.py
   ```

   This Python file will execute the commands for loading data from S3, processing it with Spark, and storing the data back to S3 in parquet files.

3.  The stored files can be explored in your corresponding AWS S3 bucket.
