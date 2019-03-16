# Query Project
- In the Query Project, you will get practice with SQL while learning about Google Cloud Platform (GCP) and BiqQuery. You'll answer business-driven questions using public datasets housed in GCP. To give you experience with different ways to use those datasets, you will use the web UI (BiqQuery) and the command-line tools, and work with them in jupyter notebooks.
- We will be using the Bay Area Bike Share Trips Data (https://cloud.google.com/bigquery/public-data/bay-bike-share). 

#### Problem Statement
- You're a data scientist at Ford GoBike (https://www.fordgobike.com/), the company running Bay Area Bikeshare. You are trying to increase ridership, and you want to offer deals through the mobile app to do so. What deals do you offer though? Currently, your company has three options: a flat price for a single one-way trip, a day pass that allows unlimited 30-minute rides for 24 hours and an annual membership. 

- Through this project, you will answer these questions: 
  * What are the 5 most popular trips that you would call "commuter trips"?
  * What are your recommendations for offers (justify based on your findings)?


## Assignment 02: Querying Data with BigQuery

### What is Google Cloud?
- Read: https://cloud.google.com/docs/overview/

### Get Going

- Go to https://cloud.google.com/bigquery/
- Click on "Try it Free"
- It asks for credit card, but you get $300 free and it does not autorenew after the $300 credit is used, so go ahead (OR CHANGE THIS IF SOME SORT OF OTHER ACCESS INFO)
- Now you will see the console screen. This is where you can manage everything for GCP
- Go to the menus on the left and scroll down to BigQuery
- Now go to https://cloud.google.com/bigquery/public-data/bay-bike-share 
- Scroll down to "Go to Bay Area Bike Share Trips Dataset" (This will open a BQ working page.)


### Some initial queries
Paste your SQL query and answer the question in a sentence.

- What's the size of this dataset? (i.e., how many trips)

**_Query 1_** : 

```sql
	#standardSQL
	SELECT count(*)
	FROM `bigquery-public-data.san_francisco.bikeshare_trips`
```
There are **983,648** rows in the public data set for bay area bike trips.

- What is the earliest start time and latest end time for a trip?

**_Query 2_** : 

```sql 
	#standardSQL
	SELECT min(start_date), max(end_date)
	FROM `bigquery-public-data.san_francisco.bikeshare_trips`
```

The above query results in the earliest start time and latest end time for a bike trip in San Fransisco as shown in the following table: 

```
Earliest Start Time 		Latest End Time
2013-08-29 09:08:00 UTC		2016-08-31 23:48:00 UTC
```

However, since the data set if for bike trips in the bay area and the fact that pacific standard time is 8 hours behind UTC, the above results mean that earliest bike trip started around 1 AM in the morning. While this is certainly possible, it is unlikely given that there are many trips in the data set with start time of morning UTC. It is possible, thus, that there is an errors in the data set and the times are actually in PST (Pacific Standard Time)

- How many bikes are there?

**_Query 3_** : 

```sql 
	#standardSQL
        SELECT count(distinct bike_number)
        FROM `bigquery-public-data.san_francisco.bikeshare_trips`
```

There are **700** bikes deduced by querying distinct bike IDs from the data set. 

### Questions of your own
- Make up 3 questions and answer them using the Bay Area Bike Share Trips Data.
- Use the SQL tutorial (https://www.w3schools.com/sql/default.asp) to help you with mechanics.

**_Question 1:_ How many trips out of the total were taken by monthly or annual subscribers and how many by daily or 3 day members?**


  * Answer: Trips taken by annual or monthly subscribers were **846839** while those taken by daily or 3 day members were **136809**
  * SQL query:

```sql 
        #standardSQL
        SELECT count(subscriber_type)
        FROM `bigquery-public-data.san_francisco.bikeshare_trips`
        WHERE (subscriber_type LIKE "Subscriber"")
```

```sql
        #standardSQL
        SELECT COUNT(subscriber_type)
        FROM `bigquery-public-data.san_francisco.bikeshare_trips`
        WHERE (subscriber_type LIKE "Customer")
```

**_Question 2_: How many trips are taken by monthly or annual subscribers where the start and end stations are not the same?**


  * Answer: There is a significantly high number of **836945** trips taken by annual or monthly subscribers who  pick up the bike from one station and drop off at another location
  * SQL query:

```sql 
        #standardSQL
        SELECT  COUNT(distinct trip_id)
        FROM `bigquery-public-data.san_francisco.bikeshare_trips`
        WHERE start_station_id != end_station_id and subscriber_type = "Subscriber"
```

**_Question 3_: Which top 5 stations have the maximum number of monthly or annual subscribers taking bike tripes**


  * Answer: 
```
Row	TRIP_COUNT	start_station_name	
1	68384		San Francisco Caltrain (Townsend at 4th)
2	53694		San Francisco Caltrain 2 (330 Townsend)
3	37888		Temporary Transbay Terminal (Howard at Beale)
4	36621		Harry Bridges Plaza (Ferry Building)
5	35500		2nd at Townsend
```
  * SQL query:

```sql
        #standardSQL
        SELECT  COUNT(*) as TRIP_COUNT, start_station_name
        FROM `bigquery-public-data.san_francisco.bikeshare_trips`
        WHERE subscriber_type = "Subscriber"
        GROUP BY start_station_name
        ORDER BY COUNT(start_station_name) DESC
        LIMIT 5
```


