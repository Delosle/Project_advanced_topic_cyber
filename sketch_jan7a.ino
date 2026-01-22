// Acquisition avec référence interne 1.1V
// Résolution: 1.1V / 1023 = 1.075 mV (au lieu de 4.9 mV)
// Idéal pour mesurer ~0.52V

const int analogPin = A0;
const int bufferSize = 512;

volatile uint16_t buffer[bufferSize];
volatile uint16_t bufferIndex = 0;
volatile bool bufferReady = false;

void setup() {
  Serial.begin(2000000);
  
  // IMPORTANT: Référence interne 1.1V pour meilleure résolution
  analogReference(INTERNAL);  // 1.1V au lieu de 5V
  
  // Attendre stabilisation de la référence (critique !)
  delay(500);
  
  // Faire plusieurs lectures pour stabiliser
  for (int i = 0; i < 10; i++) {
    analogRead(analogPin);
    delay(10);
  }
  
  // Configuration ADC rapide
  ADCSRA &= ~0x07;
  ADCSRA |= 0x05;            // Prescaler = 32, accelerate the ADC
  ADCSRA |= (1 << ADATE);    // Auto-trigger => automatic conversions (analogic value -> numeric value)
  ADCSRB = 0x00;             // Free running => No trigger event needed to continuously get data
  ADCSRA |= (1 << ADIE);     // Interrupt at the end of each conversion
  ADCSRA |= (1 << ADEN);
  ADCSRA |= (1 << ADSC);     // Start first conversion
  
  delay(100);
  
  // En-tête For synchro with python code
  Serial.write(0xFF);
  Serial.write(0xAA);
  Serial.write(0x55);
  Serial.write(0xFF);
}


// After each conversion, read value in ADC register  and put it in buffer
ISR(ADC_vect) { 
  if (!bufferReady) {
    buffer[bufferIndex++] = ADC;
    if (bufferIndex >= bufferSize) {
      bufferIndex = 0;
      bufferReady = true;
    }
  }
}

void loop() {
  if (bufferReady) {
    // Transmission binaire
    for (int i = 0; i < bufferSize; i++) {
      Serial.write((uint8_t)(buffer[i] & 0xFF));
      Serial.write((uint8_t)((buffer[i] >> 8) & 0xFF));
    }
    bufferReady = false;
  }
}