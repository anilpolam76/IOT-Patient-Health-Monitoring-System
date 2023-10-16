#include <SoftwareSerial.h>
#define RX 2
#define TX 3
String AP = "giridhar";
String PASS = "12345678";
String API = "0HBLAQ6PY8X7LA1I";
String HOST = "api.thingspeak.com";
String PORT = "80";
String field1 = "field1";
String field2 = "field2";
int countTrueCommand;
int countTimeCommand; 
boolean found = false; 
SoftwareSerial esp8266(RX,TX); 
// Set up pins for each sensor and component
const int temperaturePin = A1;
const int pulsePin = A0;
const int ultraSonicTrigger = 4;
const int ultraSonicEcho = 5;
const int buzzerPin = 7;
const int buttonPin = 6;
float temperature = 0.0;
float distance = 0.0;
bool isButtonPressed = false;
String getData = "";
String getData2 = "";
 
void setup() {
 Serial.begin(9600);
 esp8266.begin(115200);
 sendCommand("AT",5,"OK");
 sendCommand("AT+CWMODE=1",5,"OK");
 sendCommand("AT+CWJAP=\""+ AP +"\",\""+ PASS +"\"",20,"OK");
 pinMode(ultraSonicTrigger, OUTPUT);
 pinMode(ultraSonicEcho, INPUT);
 pinMode(buzzerPin, OUTPUT);
 pinMode(buttonPin, INPUT_PULLUP);
}
void loop() {
 // Reading distance from the ultra sonic sensor
 digitalWrite(ultraSonicTrigger, LOW);
 delayMicroseconds(2);
 digitalWrite(ultraSonicTrigger, HIGH);
 delayMicroseconds(10);
 digitalWrite(ultraSonicTrigger, LOW);
 float duration = pulseIn(ultraSonicEcho, HIGH);
 float distance = duration * 0.034 / 2.0;
 
 // Checking if the push button is pressed
 bool isButtonPressed = !digitalRead(buttonPin);
 Serial.print(", Distance: ");
 Serial.print(distance);
 Serial.print("cm, Button: ");
 Serial.println(isButtonPressed);
 int pulsesensor = getpulseSensorData();
 String getData1 = "GET /update?api_key="+ API +"&"+ field1 +"="+ pulsesensor;
sendCommand("AT+CIPMUX=1",5,"OK");
 sendCommand("AT+CIPSTART=0,\"TCP\",\""+ HOST +"\","+ PORT,15,"OK");
 sendCommand("AT+CIPSEND=0," +String(getData1.length()+4),4,">");
 esp8266.println(getData1);delay(1500);countTrueCommand++;
 sendCommand("AT+CIPCLOSE=0",5,"OK");
int tempsensor = gettempSensorData();
 String getData2 = "GET /update?api_key="+ API +"&"+ field2 +"="+ tempsensor;
 sendCommand("AT+CIPMUX=1",5,"OK");
 sendCommand("AT+CIPSTART=0,\"TCP\",\""+ HOST +"\","+ PORT,15,"OK");
 sendCommand("AT+CIPSEND=0," +String(getData2.length()+4),4,">");
 esp8266.println(getData2);delay(1500);countTrueCommand++;
 sendCommand("AT+CIPCLOSE=0",5,"OK");
 // Checking if the ultrasonic sensor should be activated
 if (isButtonPressed && (tempsensor < 0 || tempsensor > 100 || pulsesensor < 0 || pulsesensor > 150 || 
distance < 0.1)) {
 digitalWrite(buzzerPin, HIGH);
 delay(1000);
 digitalWrite(buzzerPin, LOW);
 }
 
 delay(5000);
}
int getpulseSensorData(){
 //return random(1000); // Replace with your own sensor code
 int pulse=analogRead(pulsePin);
 Serial.print("pulse reading=");
 Serial.println(pulse);
 delay(1000);
}
int gettempSensorData(){
 //return random(1000); // Replace with your own sensor code
 int temp=analogRead(temperaturePin);
 Serial.print("temp reading=");
 Serial.println(temp);
 delay(1000);
}
void sendCommand(String command, int maxTime, char readReplay[]) {
 Serial.print(countTrueCommand);
 Serial.print(". at command => ");
 Serial.print(command);
 Serial.print(" ");
 while(countTimeCommand < (maxTime*1))
 {
 esp8266.println(command);//at+cipsend
 if(esp8266.find(readReplay))//ok
 {
 found = true;
 break;
 }
 
 countTimeCommand++;
 }
 
 if(found == true)
 {
 Serial.println("OYI");
 countTrueCommand++;
 countTimeCommand = 0;
 }
 
 if(found == false)
 {
 Serial.println("Fail");
 countTrueCommand = 0;
 countTimeCommand = 0;
 }
 
 found = false;
}
