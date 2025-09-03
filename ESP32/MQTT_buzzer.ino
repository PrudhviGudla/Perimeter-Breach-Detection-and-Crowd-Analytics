#include <WiFi.h>
#include <PubSubClient.h>

// Replace with your network credentials
const char* ssid = "";
const char* password = "";

// MQTT Broker details
const char* mqtt_server = "broker.emqx.io";  // Replace with your EMQX broker IP or hostname
const int mqtt_port = 1883;
const char* mqtt_topic = "esp32/buzzer";

// Define the GPIO pin for the buzzer
const int buzzerPin = 4; // GPIO4 (D4)

// Initialize the WiFi and MQTT clients
WiFiClient espClient;
PubSubClient client(espClient);

// Function prototypes
void setup_wifi();
void callback(char* topic, byte* payload, unsigned int length);
void reconnect();

void setup() {
  // Initialize the buzzer pin
  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin, LOW); // Ensure buzzer is off

  // Start serial communication for debugging
  Serial.begin(115200);

  // Connect to Wi-Fi
  setup_wifi();

  // Set the MQTT server and callback
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to Wi-Fi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("Wi-Fi connected.");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("]: ");
  String message;

  for (unsigned int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
    message += (char)payload[i];
  }
  Serial.println();

  if (String(topic) == mqtt_topic) {
    if (message == "ON") {
      digitalWrite(buzzerPin, HIGH); // Activate buzzer
      Serial.println("Buzzer turned ON");
      // Keep the buzzer on for a duration or implement your own logic
      delay(5000);
      digitalWrite(buzzerPin, LOW); // Deactivate buzzer
      Serial.println("Buzzer turned OFF");
    }
  }
}

void reconnect() {
  // Loop until reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect (you can add MQTT username and password if needed)
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      // Subscribe to the topic
      client.subscribe(mqtt_topic);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" trying again in 5 seconds");
      // Wait before retrying
      delay(5000);
    }
  }
}
