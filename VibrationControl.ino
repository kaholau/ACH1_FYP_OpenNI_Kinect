int leftPin1 =    8;
int middlePin1 =  9;
int rightPin1 =   10;

void vibrateGroup(int group) {
  digitalWrite(leftPin1, HIGH);
  digitalWrite(middlePin1, HIGH);
  digitalWrite(rightPin1, HIGH);

  switch (group) {
    case 1: // left group
    digitalWrite(leftPin1, LOW);
    break;
    
    case 2: // middle group
    digitalWrite(middlePin1, LOW);
    break;
    
    case 3: // right group
    digitalWrite(rightPin1, LOW);
    break;
  }
}

void setup() {
  // initialize serial:
  Serial.begin(9600);

  // set pins to output
  pinMode(leftPin1, OUTPUT);
  pinMode(middlePin1, OUTPUT);
  pinMode(rightPin1, OUTPUT);
}

void loop() {
  if (Serial.available() < 1)
    return;

  int group = Serial.parseInt();

  vibrateGroup(group);
}