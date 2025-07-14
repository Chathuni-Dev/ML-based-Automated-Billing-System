#include "HX711.h"
HX711 scale;

#define DOUT 2
#define CLK 3

void setup() {
  Serial.begin(9600);
  scale.begin(DOUT, CLK);
  scale.set_scale(); 
  scale.tare();      
}

void loop() {
  float weight = scale.get_units(5); 
  Serial.println(weight);
  delay(500);
}
