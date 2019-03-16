# Annotations For Assignment 08 MIDS W205 Spring 2019

## Author : Suhas Gupta
## Docker Containers Used: mids, zookeeper, kafka, spark and hdfs
### Objective: Publishing messages to a topic with kafkacat, consuming and analyzing messages with Spark, transforming messages in PySpark and publishing results to hdfs

## Assignment Steps: 

### Download the data 
  ```
cd ~/w205
curl -L -o assessment-attempts-20180128-121051-nested.json https://goo.gl/ME6hjp
cd spark-with-kafka
```
### Copy the docker-compose.yml file containg the cluster definitions from the week 07 course content area (cloned earlier).

```
cp ~/w205/course-content/08-Querying-Data/docker-compose.yml .
```

### Bring up headless containers in a cluster as defined in the docker-compose.yml file. We can also follow the kafka log file in a separate temrinal to watch the kafka container coming up. 

```
docker-compose up -d
docker-compose logs -f kafka
```
### Now we create a kafka topic called "assessments" for customers to publish their assessments to.

  ```
docker-compose exec kafka kafka-topics --create --topic assessments --partitions 1 --replication-factor 1 --if-not-exists --zookeeper zookeeper:32181
```

### We can check the topic just created using the following command:

```
docker-compose exec kafka kafka-topics --describe --topic assessments --zookeeper zookeeper:32181\n
```

### We can use jq to count the number of messages in our json file. The total number of messages were 3280.
```
jq length ../assessment-attempts-20180128-121051-nested.json
```
#### Output: 
```
3280
```

### Now we publish all the messages in the json file to the assessments topic using kafkacat
```
docker-compose exec mids bash -c "cat assessment-attempts-20180128-121051-nested.json | jq '.[]' -c | kafkacat -P -b kafka:29092 -t assessments && echo 'Produced 3280 messages.'"
```

### Now we will use Spark to consume the messages in our topic . We can start Spark using the spark container in our cluster that is running

```
docker-compose exec spark pyspark
```

### The above command starts a spark command line session where we read messages from kafka as follows

```python
messages = spark.read.format("kafka").option("kafka.bootstrap.servers", "kafka:29092").option("subscribe","assessments").option("startingOffsets", "earliest").option("endingOffsets", "latest").load() 
messages
```
#### Output:

```python
DataFrame[key: binary, value: binary, topic: string, partition: int, offset: bigint, timestamp: timestamp, timestampType: int]
```

### Lets cache this to avoid generating too many warnings in the later commands

```python
messages.cache()
```

### We can look at the schema of the data frame read above

```python
messages.printSchema()
```
#### Output: 
```python 
root
|-- key: binary (nullable = true)
|-- value: binary (nullable = true)
|-- topic: string (nullable = true)
|-- partition: integer (nullable = true)
|-- offset: long (nullable = true)
|-- timestamp: timestamp (nullable = true)
|-- timestampType: integer (nullable = true)
```

### We will cast the value field in the messages data frame to string type from the exisiting hexadecimal values

```python 
messages_as_strings=messages.selectExpr("CAST(value AS STRING)")
messages_as_strings.show()
```
#### Output:

```python
+--------------------+
|               value|
+--------------------+
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
|{"keen_timestamp"...|
+--------------------+

```

### Take a look at the schema of string converted messages: 
```python
messages_as_strings.printSchema()
```
#### Output: 
```python
root
 |-- value: string (nullable = true)
```

### Now we will transform the messages data frame using PySpark SQL as shown below. Here we load the json file and apply the Row function to unroll the messages data for the whole data frame. The resulting data structure is a map object. 
### Following the transformation we save the transformed messages to hdfs.

```python
extracted_messages= messages.rdd.map(lambda x: Row(**json.loads(x.value))).toDF()
extracted_messages.write.parquet("/tmp/extracted_messages")
```
### Lets us output the messages before and after transformation to visualize what we did during the extraction process.

```python 
messages.show()
```

#### Output: 

```python
+----+--------------------+-----------+---------+------+--------------------+-------------+
| key|               value|      topic|partition|offset|           timestamp|timestampType|
+----+--------------------+-----------+---------+------+--------------------+-------------+
|null|[7B 22 6B 65 65 6...|assessments|        0|     0|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|     1|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|     2|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|     3|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|     4|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|     5|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|     6|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|     7|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|     8|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|     9|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    10|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    11|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    12|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    13|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    14|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    15|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    16|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    17|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    18|1969-12-31 23:59:...|            0|
|null|[7B 22 6B 65 65 6...|assessments|        0|    19|1969-12-31 23:59:...|            0|
+----+--------------------+-----------+---------+------+--------------------+-------------+

```python
extracted_messages.show()
```

#### Output:
```python
+--------------------+-------------+--------------------+------------------+--------------------+------------------+------------+--------------------+--------------------+--------------------+
|        base_exam_id|certification|           exam_name|   keen_created_at|             keen_id|    keen_timestamp|max_attempts|           sequences|          started_at|        user_exam_id|
+--------------------+-------------+--------------------+------------------+--------------------+------------------+------------+--------------------+--------------------+--------------------+
|37f0a30a-7464-11e...|        false|Normal Forms and ...| 1516717442.735266|5a6745820eb8ab000...| 1516717442.735266|         1.0|Map(questions -> ...|2018-01-23T14:23:...|6d4089e4-bde5-4a2...|
|37f0a30a-7464-11e...|        false|Normal Forms and ...| 1516717377.639827|5a674541ab6b0a000...| 1516717377.639827|         1.0|Map(questions -> ...|2018-01-23T14:21:...|2fec1534-b41f-441...|
|4beeac16-bb83-4d5...|        false|The Principles of...| 1516738973.653394|5a67999d3ed3e3000...| 1516738973.653394|         1.0|Map(questions -> ...|2018-01-23T20:22:...|8edbc8a8-4d26-429...|
|4beeac16-bb83-4d5...|        false|The Principles of...|1516738921.1137421|5a6799694fc7c7000...|1516738921.1137421|         1.0|Map(questions -> ...|2018-01-23T20:21:...|c0ee680e-8892-4e6...|
|6442707e-7488-11e...|        false|Introduction to B...| 1516737000.212122|5a6791e824fccd000...| 1516737000.212122|         1.0|Map(questions -> ...|2018-01-23T19:48:...|e4525b79-7904-405...|
|8b4488de-43a5-4ff...|        false|        Learning Git| 1516740790.309757|5a67a0b6852c2a000...| 1516740790.309757|         1.0|Map(questions -> ...|2018-01-23T20:51:...|3186dafa-7acf-47e...|
|e1f07fac-5566-4fd...|        false|Git Fundamentals ...|1516746279.3801291|5a67b627cc80e6000...|1516746279.3801291|         1.0|Map(questions -> ...|2018-01-23T22:24:...|48d88326-36a3-4cb...|
|7e2e0b53-a7ba-458...|        false|Introduction to P...| 1516743820.305464|5a67ac8cb0a5f4000...| 1516743820.305464|         1.0|Map(questions -> ...|2018-01-23T21:43:...|bb152d6b-cada-41e...|
|1a233da8-e6e5-48a...|        false|Intermediate Pyth...|  1516743098.56811|5a67a9ba060087000...|  1516743098.56811|         1.0|Map(questions -> ...|2018-01-23T21:31:...|70073d6f-ced5-4d0...|
|7e2e0b53-a7ba-458...|        false|Introduction to P...| 1516743764.813107|5a67ac54411aed000...| 1516743764.813107|         1.0|Map(questions -> ...|2018-01-23T21:42:...|9eb6d4d6-fd1f-4f3...|
|4cdf9b5f-fdb7-4a4...|        false|A Practical Intro...|1516744091.3127241|5a67ad9b2ff312000...|1516744091.3127241|         1.0|Map(questions -> ...|2018-01-23T21:45:...|093f1337-7090-457...|
|e1f07fac-5566-4fd...|        false|Git Fundamentals ...|1516746256.5878439|5a67b610baff90000...|1516746256.5878439|         1.0|Map(questions -> ...|2018-01-23T22:24:...|0f576abb-958a-4c0...|
|87b4b3f9-3a86-435...|        false|Introduction to M...|  1516743832.99235|5a67ac9837b82b000...|  1516743832.99235|         1.0|Map(questions -> ...|2018-01-23T21:40:...|0c18f48c-0018-450...|
|a7a65ec6-77dc-480...|        false|   Python Epiphanies|1516743332.7596769|5a67aaa4f21cc2000...|1516743332.7596769|         1.0|Map(questions -> ...|2018-01-23T21:34:...|b38ac9d8-eef9-495...|
|7e2e0b53-a7ba-458...|        false|Introduction to P...| 1516743750.097306|5a67ac46f7bce8000...| 1516743750.097306|         1.0|Map(questions -> ...|2018-01-23T21:41:...|bbc9865f-88ef-42e...|
|e5602ceb-6f0d-11e...|        false|Python Data Struc...|1516744410.4791961|5a67aedaf34e85000...|1516744410.4791961|         1.0|Map(questions -> ...|2018-01-23T21:51:...|8a0266df-02d7-44e...|
|e5602ceb-6f0d-11e...|        false|Python Data Struc...|1516744446.3999851|5a67aefef5e149000...|1516744446.3999851|         1.0|Map(questions -> ...|2018-01-23T21:53:...|95d4edb1-533f-445...|
|f432e2e3-7e3a-4a7...|        false|Working with Algo...| 1516744255.840405|5a67ae3f0c5f48000...| 1516744255.840405|         1.0|Map(questions -> ...|2018-01-23T21:50:...|f9bc1eff-7e54-42a...|
|76a682de-6f0c-11e...|        false|Learning iPython ...| 1516744023.652257|5a67ad579d5057000...| 1516744023.652257|         1.0|Map(questions -> ...|2018-01-23T21:46:...|dc4b35a7-399a-4bd...|
|a7a65ec6-77dc-480...|        false|   Python Epiphanies|1516743398.6451161|5a67aae6753fd6000...|1516743398.6451161|         1.0|Map(questions -> ...|2018-01-23T21:35:...|d0f8249a-597e-4e1...|
+--------------------+-------------+--------------------+------------------+--------------------+------------------+------------+--------------------+--------------------+--------------------+
```

### Let us exit the spark session.
```python
exit()
```

### Let's check hadoop and see if our results are there:
```
docker-compose exec cloudera hadoop fs -ls /tmp/
```

#### Output:
```
Found 3 items
drwxr-xr-x   - root   supergroup          0 2019-03-09 04:44 /tmp/extracted_messages
drwxrwxrwt   - mapred mapred              0 2018-02-06 18:27 /tmp/hadoop-yarn
drwx-wx-wx   - root   supergroup          0 2019-03-09 03:47 /tmp/hive
```

```
docker-compose exec cloudera hadoop fs -ls /tmp/extracted_messages
```
#### Output:
```
Found 2 items
-rw-r--r--   1 root supergroup          0 2019-03-09 04:44 /tmp/extracted_messages/_SUCCESS
-rw-r--r--   1 root supergroup     345388 2019-03-09 04:44 /tmp/extracted_messages/part-00000-ecd10bf7-07eb-4268-a2ce-e0c3f78ef3f2-c000.snappy.parquet
```
### We were able to successfully create a kafka topic and publish messages to it using kafkacat. Then we read in the topic in Spark, transformed is using SparkSQL and finally wrote the results to hdfs. 

### Now we can bring our cluster down and check for any stray containers after the cluster is down. 

```
docker-compose down
```
#### Output:
```
Stopping assignment-08-suhasgupta791_spark_1     ... done
Stopping assignment-08-suhasgupta791_kafka_1     ... done
Stopping assignment-08-suhasgupta791_cloudera_1  ... done
Stopping assignment-08-suhasgupta791_mids_1      ... done
Stopping assignment-08-suhasgupta791_zookeeper_1 ... done
Removing assignment-08-suhasgupta791_spark_1     ... 
Removing assignment-08-suhasgupta791_spark_1     ... done
Removing assignment-08-suhasgupta791_kafka_1     ... done
Removing assignment-08-suhasgupta791_cloudera_1  ... done
Removing assignment-08-suhasgupta791_mids_1      ... done
Removing assignment-08-suhasgupta791_zookeeper_1 ... done
Removing network assignment-08-suhasgupta791_default
```

```
docker-compose ps 
```
#### Output:
```
Name   Command   State   Ports
------------------------------
```
```
docker ps -a
```

#### Output:
```
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```
