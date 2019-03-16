# Annotations For Assignment 07 MIDS W205 Spring 2019

## Author : Suhas Gupta
## Docker Containers Used: mids, zookeeper, kafka and spark
## Objective: Publishing messages to a topic with kafkacat and consuming and analyzing messages with Spark.

## Assingment Steps: 

### Download the data 
  ```
cd ~/w205
curl -L -o assessment-attempts-20180128-121051-nested.json https://goo.gl/ME6hjp
cd spark-with-kafka
```
### Copy the docker-compose.yml file containg the cluster definitions from the week 07 course content area (cloned earlier).

```
cp ~/w205/course-content/07-Sourcing-Data/docker-compose.yml .
vi docker-compose.yml
```

### Bring up headless containers in a cluster as defined in the docker-compose.yml file. We can also follow the kafka log file to watch the kafka container coming up. 

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

### The following three commands are used to check the messages in our json file. The utility jq is useful in formatting the message output. This second command formats the json file into a pretty format for human viewing. The third command puts all the messages in the json file on a single line for feeding into kafkacat.   

```
docker-compose exec mids bash -c "cat  assessment-attempts-20180128-121051-nested.json" 
docker-compose exec mids bash -c "cat  assessment-attempts-20180128-121051-nested.json  | jq '.'"
docker-compose exec mids bash -c "cat  assessment-attempts-20180128-121051-nested.json  | jq '.' -c "
```

### We can also use jq to count the number of messages in our json file. The total number of messages were 3280.
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


### Now we will use Spark to consume the messages in our topic 

### We can start Spark using the spark container in our cluster that is running

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
### Look at the messages:
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
only showing top 20 rows
```

### Note that in above result the values are in hexadecimal nibbles equivalents of their string ascii values. We can convert this hexadecimal list to strings for better readability as follows: 

```python 
messages_as_strings=messages.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
messages_as_strings.show()
```
#### Output:

```python
+----+--------------------+
| key|               value|
+----+--------------------+
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
|null|{"keen_timestamp"...|
```

### Take a look at the schema of string converted messages: 
```python
messages_as_strings.printSchema()
```
#### Output: 
```python
root
|-- key: string (nullable = true)
|-- value: string (nullable = true)
```

### Next we will unroll the json file in Spark to get meaningful information out of the assessments data frame.
    - We get the first message from the topic using the json.loads command on the messages (with string keys & values). Note the format that is required due to two levels of indirection in the data frame read into Spark.
    - We check the type of the data structure that 'first_message' is. It is a conventional python dictionary data strcuture. 
    - We can now simply use dictionary indexing to get the number of correct response for the first mesaage. 

```python
import json 
first_message=json.loads(messages_as_strings.select('value').take(1)[0].value)
type(first_message)
first_message
print(first_message['sequences']['counts']['correct'])
```

#### Output:

```python
<class 'dict'>

{'keen_timestamp': '1516717442.735266', 'max_attempts': '1.0', 'started_at': '2018-01-23T14:23:19.082Z', 'base_exam_id': '37f0a30a-7464-11e6-aa92-a8667f27e5dc', 'user_exam_id': '6d4089e4-bde5-4a22-b65f-18bce9ab79c8', 'sequences': {'questions': [{'user_incomplete': True, 'user_correct': False, 'options': [{'checked': True, 'at': '2018-01-23T14:23:24.670Z', 'id': '49c574b4-5c82-4ffd-9bd1-c3358faf850d', 'submitted': 1, 'correct': True}, {'checked': True, 'at': '2018-01-23T14:23:25.914Z', 'id': 'f2528210-35c3-4320-acf3-9056567ea19f', 'submitted': 1, 'correct': True}, {'checked': False, 'correct': True, 'id': 'd1bf026f-554f-4543-bdd2-54dcf105b826'}], 'user_submitted': True, 'id': '7a2ed6d3-f492-49b3-b8aa-d080a8aad986', 'user_result': 'missed_some'}, {'user_incomplete': False, 'user_correct': False, 'options': [{'checked': True, 'at': '2018-01-23T14:23:30.116Z', 'id': 'a35d0e80-8c49-415d-b8cb-c21a02627e2b', 'submitted': 1}, {'checked': False, 'correct': True, 'id': 'bccd6e2e-2cef-4c72-8bfa-317db0ac48bb'}, {'checked': True, 'at': '2018-01-23T14:23:41.791Z', 'id': '7e0b639a-2ef8-4604-b7eb-5018bd81a91b', 'submitted': 1, 'correct': True}], 'user_submitted': True, 'id': 'bbed4358-999d-4462-9596-bad5173a6ecb', 'user_result': 'incorrect'}, {'user_incomplete': False, 'user_correct': True, 'options': [{'checked': False, 'at': '2018-01-23T14:23:52.510Z', 'id': 'a9333679-de9d-41ff-bb3d-b239d6b95732'}, {'checked': False, 'id': '85795acc-b4b1-4510-bd6e-41648a3553c9'}, {'checked': True, 'at': '2018-01-23T14:23:54.223Z', 'id': 'c185ecdb-48fb-4edb-ae4e-0204ac7a0909', 'submitted': 1, 'correct': True}, {'checked': True, 'at': '2018-01-23T14:23:53.862Z', 'id': '77a66c83-d001-45cd-9a5a-6bba8eb7389e', 'submitted': 1, 'correct': True}], 'user_submitted': True, 'id': 'e6ad8644-96b1-4617-b37b-a263dded202c', 'user_result': 'correct'}, {'user_incomplete': False, 'user_correct': True, 'options': [{'checked': False, 'id': '59b9fc4b-f239-4850-b1f9-912d1fd3ca13'}, {'checked': False, 'id': '2c29e8e8-d4a8-406e-9cdf-de28ec5890fe'}, {'checked': False, 'id': '62feee6e-9b76-4123-bd9e-c0b35126b1f1'}, {'checked': True, 'at': '2018-01-23T14:24:00.807Z', 'id': '7f13df9c-fcbe-4424-914f-2206f106765c', 'submitted': 1, 'correct': True}], 'user_submitted': True, 'id': '95194331-ac43-454e-83de-ea8913067055', 'user_result': 'correct'}], 'attempt': 1, 'id': '5b28a462-7a3b-42e0-b508-09f3906d1703', 'counts': {'incomplete': 1, 'submitted': 4, 'incorrect': 1, 'all_correct': False, 'correct': 2, 'total': 4, 'unanswered': 0}}, 'keen_created_at': '1516717442.735266', 'certification': 'false', 'keen_id': '5a6745820eb8ab00016be1f1', 'exam_name': 'Normal Forms and All That Jazz Master Class'}

2
```

### We can exit the spark session now.
```python
exit()
```

### Now we can bring our cluster down and check for any stray containers after the cluster is down. 

```
docker-compose down 
docker-compose ps 
docker ps -a 
```

