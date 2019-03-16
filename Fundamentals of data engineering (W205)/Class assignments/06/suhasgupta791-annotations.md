# Annotations for assignment 06 MIDS W205

## Spin a cluster with mids, zookeeper and kafka
## Publishing and consuming messages with kafka

### Change directory to the W205 assignment 06 work area inside the home directory. 

### Download the data 
```
cd ~/w205/assignment-06-suhasgupta791/
curl -L -o assessment-attempts-20180128-121051-nested.json https://goo.gl/ME6hjp
```

### Copy the docker-compose.yml file containg the cluster definitions from the week 06 course content area (cloned earlier).
```
cd ~/w205/
cd assignment-06-suhasgupta791
cp ../course-content/06-Transforming-Data/docker-compose.yml .
```

### Bring up headless containers in a cluster as defined in the docker-compose.yml file and check that the clusters are up.

```
docker-compose up -d 
docker-compose ps 
```


### Kafka is a distributed streaming platform used for handling a very large number of events. It was initally conceived as a messaging queue and has evolved into a full streaming platform.

### Create a kafka topic called "assessments" for customers to publish their assessments on.

```
docker-compose exec kafka kafka-topics --create --topic assessments --partitions 1 --replication-factor 1 --if-not-exists --zookeeper zookeeper:32181
```

### Check the topic created above

```
docker-compose exec kafka kafka-topics --describe --topic assessments --zookeeper zookeeper:32181
```

### Now we can publish an example message to this topic

```
docker-compose exec mids bash -c "cat  assignment-06-suhasgupta791/assessment-attempts-20180128-121051-nested.json | jq '.[]' -c | kafkacat -P -b kafka:29092 -t assessments && echo 'Produced a lot of messages.'"
```

### To check all the messages in the topic we can do the following: 

```
docker-compose exec mids bash -c "kafkacat -C -b kafka:29092 -t assessments -o beginning -e"
```

### The following command will count the number of messages in the topic assessment that we created:
```
docker-compose exec mids bash -c "kafkacat -C -b kafka:29092 -t assessments -o beginning -e" | wc -l
```

### Now we can bring the cluster down
```
docker-compose down 
docker-compose ps 
docker ps -a 
```
