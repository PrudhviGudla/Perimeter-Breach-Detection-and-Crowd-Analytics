from kafka import KafkaConsumer
from kafka.structs import TopicPartition

topic = 'test_microservice'
bootstrap_servers = 'localhost:9092'
consumer = KafkaConsumer(
    topic, bootstrap_servers=bootstrap_servers, auto_offset_reset='earliest')

partitions = consumer.partitions_for_topic(topic)
for p in partitions:
    topic_partition = TopicPartition(topic, p)
    # Seek offset 0
    consumer.seek(partition=topic_partition, offset=0)
    for msg in consumer:
        print(msg.value.decode("utf-8"))