// Test code for Ultimate GPS Using Hardware Serial (e.g. GPS Flora or FeatherWing)
//
// This code shows how to listen to the GPS module via polling. Best used with
// Feathers or Flora where you have hardware Serial and no interrupt
//
// Tested and works great with the Adafruit GPS FeatherWing
// ------> https://www.adafruit.com/products/3133
// or Flora GPS
// ------> https://www.adafruit.com/products/1059
// but also works with the shield, breakout
// ------> https://www.adafruit.com/products/1272
// ------> https://www.adafruit.com/products/746
//
// Pick one up today at the Adafruit electronics shop
// and help support open source hardware & software! -ada

#include <Adafruit_GPS.h>
#include <Adafruit_PMTK.h>
#include <NMEA_data.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <SD.h>


// what's the name of the hardware serial port?
#define GPSSerial Serial1
#define GPSECHO false
File bnoFILE;
File gpsFILE;

const int ledPin = 32;
#define UARTSERIAL Serial2

Adafruit_GPS GPS(&GPSSerial);
uint32_t timer = millis();


//Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29, &Wire);
Adafruit_BNO055 bno = Adafruit_BNO055(55, 0x29);

uint16_t BNO055_SAMPLERATE_DELAY_MS = 10000;

const int chipSelect = BUILTIN_SDCARD; 

void readGPS() // run over and over again
{
  //gpsFILE = SD.open("gps_demo.csv", FILE_WRITE);
  // read data from the GPS in the 'main loop'
  //gpsFILE = SD.open("gps_curr.csv", FILE_WRITE);
  char c = GPS.read();
  // if you want to debug, this is a good time to do it!
  if (GPSECHO)
    if (c) Serial.print(c);
  // if a sentence is received, we can check the checksum, parse it...
  if (GPS.newNMEAreceived()) {
    // a tricky thing here is if we print the NMEA sentence, or data
    // we end up not listening and catching other sentences!
    // so be very wary if using OUTPUT_ALLDATA and trying to print out data
    Serial.print(GPS.lastNMEA()); // this also sets the newNMEAreceived() flag to false
    if (!GPS.parse(GPS.lastNMEA())) // this also sets the newNMEAreceived() flag to false
      return; // we can fail to parse a sentence in which case we should just wait for another
  }

  // approximately every 2 seconds or so, print out the current stats
   if (millis() - timer > 200) {
    timer = millis(); // reset the timer
    Serial.print("\nTime: ");
    if (GPS.hour < 10) { Serial.print('0'); }
    Serial.print(GPS.hour, DEC); Serial.print(':');
    if (GPS.minute < 10) { Serial.print('0'); }
    Serial.print(GPS.minute, DEC); Serial.print(':');
    if (GPS.seconds < 10) { Serial.print('0'); }
    Serial.print(GPS.seconds, DEC); Serial.print('.');
    if (GPS.milliseconds < 10) {
      Serial.print("00");
    } else if (GPS.milliseconds > 9 && GPS.milliseconds < 100) {
      Serial.print("0");
    }
    Serial.println(GPS.milliseconds);
    Serial.print("Date: ");
    Serial.print(GPS.day, DEC); Serial.print('/');
    Serial.print(GPS.month, DEC); Serial.print("/20");
    Serial.println(GPS.year, DEC);
    Serial.print("Fix: "); Serial.print((int)GPS.fix);
    Serial.print(" quality: "); Serial.println((int)GPS.fixquality);
    if (GPS.fix) {
      Serial.print("Location: ");
      Serial.print(GPS.latitude, 4); Serial.print(GPS.lat);

      gpsFILE.print(GPS.latitude, 4);
      gpsFILE.print(",");
      gpsFILE.print(GPS.lat);
      gpsFILE.print(",");
      gpsFILE.println();
      
       Serial.print(", ");
       Serial.print(GPS.longitude, 4); Serial.println(GPS.lon);
      Serial.print("Speed (knots): "); Serial.println(GPS.speed);
      Serial.print("Angle: "); Serial.println(GPS.angle);
      Serial.print("Altitude: "); Serial.println(GPS.altitude);
      Serial.print("Satellites: "); Serial.println((int)GPS.satellites);
      Serial.print("Antenna status: "); Serial.println((int)GPS.antenna);
      digitalWrite(ledPin, LOW);
    }
    else {
      digitalWrite(ledPin, HIGH);
    }
  }
  //readBNO();
  //gpsFILE.close();
}

void setupLED() {
  pinMode(ledPin, OUTPUT);
}

void setupUART(void) {
  UARTSERIAL.begin(115200); //was 115200
  while (!UARTSERIAL) delay(10);  // wait for UARTSERIAL port to open!
}


void setupGPS()
{
  //while (!Serial);  // uncomment to have the sketch wait until Serial is ready

  // connect at 115200 so we can read the GPS fast enough and echo without dropping chars
  // also spit it out
  Serial.begin(115200);
  Serial.println("Adafruit GPS library basic parsing test!");

  // 9600 NMEA is the default baud rate for Adafruit MTK GPS's- some use 4800
  GPS.begin(9600);
  // uncomment this line to turn on RMC (recommended minimum) and GGA (fix data) including altitude
  GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCGGA);
  // uncomment this line to turn on only the "minimum recommended" data
  //GPS.sendCommand(PMTK_SET_NMEA_OUTPUT_RMCONLY);
  // For parsing data, we don't suggest using anything but either RMC only or RMC+GGA since
  // the parser doesn't care about other sentences at this time
  // Set the update rate
  GPS.sendCommand(PMTK_SET_NMEA_UPDATE_1HZ); // 1 Hz update rate
  // For the parsing code to work nicely and have time to sort thru the data, and
  // print it out we don't suggest using anything higher than 1 Hz

  // Request updates on antenna status, comment out to keep quiet
  GPS.sendCommand(PGCMD_ANTENNA);

  delay(1000);

  // Ask for firmware version
  GPSSerial.println(PMTK_Q_RELEASE);
  //for(int i = 0; i < 1000000; i++) {
  //  readGPS();
 // }
}

void setupBNO(void) {
  if (!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while (1);
  }
  //Wire0.begin().
  //Wire0.setClock(400000) ;
}

void setupSD(void) {
  if (!SD.begin(chipSelect)) {
    Serial.println("SD card initialization failed!");
    return;
  }
}

void setup() {
  setupGPS();
  setupBNO();
  setupSD();
  setupLED();
  setupUART();
  tempmon_init();
  tempmon_Start();

}

void readBNO()
{ 
  //bnoFILE = SD.open("bno_curr.csv", FILE_WRITE);
  sensors_event_t orientationData , angVelocityData , linearAccelData, magnetometerData, accelerometerData, gravityData;
  bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
  //bno.getEvent(&angVelocityData, Adafruit_BNO055::VECTOR_GYROSCOPE);
  bno.getEvent(&linearAccelData, Adafruit_BNO055::VECTOR_LINEARACCEL);
  //bno.getEvent(&magnetometerData, Adafruit_BNO055::VECTOR_MAGNETOMETER);
  //bno.getEvent(&accelerometerData, Adafruit_BNO055::VECTOR_ACCELEROMETER);
  //bno.getEvent(&gravityData, Adafruit_BNO055::VECTOR_GRAVITY);
  //Serial.println();
  //Serial.print("BNO DATA");
  //Serial.println();
  Serial.print(linearAccelData.acceleration.x);
  Serial.print(",");
  Serial.print(linearAccelData.acceleration.y);
  Serial.print(",");
  Serial.print(linearAccelData.acceleration.z);
  Serial.print(",");
  Serial.println();


  // UARTSERIAL.print("Orientation Data");
  // Serial.println();
  // Serial.print(orientationData.orientation.x);
  // Serial.print(",");
  // Serial.print(orientationData.orientation.y);
  // Serial.print(",");
  // Serial.print(orientationData.orientation.z);
  // Serial.print(",");
  // Serial.println();

  // UARTSERIAL.print("Linear Accel Data");
  // UARTSERIAL.println();
  // UARTSERIAL.print(linearAccelData.acceleration.x);
  // UARTSERIAL.print(",");
  // UARTSERIAL.print(linearAccelData.acceleration.y);
  // UARTSERIAL.print(",");
  // UARTSERIAL.print(linearAccelData.acceleration.z);
  // UARTSERIAL.print(",");
  // UARTSERIAL.println();

  //bnoFILE.close();

}

void loop() {
  
  //readGPS();
 
  //digitalWrite(ledPin, HIGH); // Turn the LED on
  //delay(1000);                // Wait for a second
  //digitalWrite(ledPin, LOW);  // Turn the LED off
 // delay(1000);                // Wait for a second
  int8_t boardTemp = bno.getTemp();
  while(boardTemp > 70) {
    Serial.print("TEENSY OVERHEATING");
  }
  
  
  //Serial.print("HELLO");
  //delay(1000);
  readBNO();
  //delay(1000);
}



void printEvent(sensors_event_t* event) {
  double x = -1000000, y = -1000000 , z = -1000000; //dumb values, easy to spot problem
  if (event->type == SENSOR_TYPE_ACCELEROMETER) {
    Serial.print("Accl:");
    x = event->acceleration.x;
    y = event->acceleration.y;
    z = event->acceleration.z;
  }
  else if (event->type == SENSOR_TYPE_ORIENTATION) {
    Serial.print("Orient:");
    x = event->orientation.x;
    y = event->orientation.y;
    z = event->orientation.z;
  }
  else if (event->type == SENSOR_TYPE_MAGNETIC_FIELD) {
    Serial.print("Mag:");
    x = event->magnetic.x;
    y = event->magnetic.y;
    z = event->magnetic.z;
  }
  else if (event->type == SENSOR_TYPE_GYROSCOPE) {
    Serial.print("Gyro:");
    x = event->gyro.x;
    y = event->gyro.y;
    z = event->gyro.z;
  }
  else if (event->type == SENSOR_TYPE_ROTATION_VECTOR) {
    Serial.print("Rot:");
    x = event->gyro.x;
    y = event->gyro.y;
    z = event->gyro.z;
  }
  else if (event->type == SENSOR_TYPE_LINEAR_ACCELERATION) {
    Serial.print("Linear:");
    x = event->acceleration.x;
    y = event->acceleration.y;
    z = event->acceleration.z;
  }
  else if (event->type == SENSOR_TYPE_GRAVITY) {
    Serial.print("Gravity:");
    x = event->acceleration.x;
    y = event->acceleration.y;
    z = event->acceleration.z;
  }
  else {
    Serial.print("Unk:");
  }

  Serial.print("\tx= ");
  Serial.print(x);
  Serial.print(" |\ty= ");
  Serial.print(y);
  Serial.print(" |\tz= ");
  Serial.println(z);
}

