from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092')

for i in range(10):
    message = f'Message {i}'
    producer.send('test_microservice', message.encode('utf-8'))

producer.flush()