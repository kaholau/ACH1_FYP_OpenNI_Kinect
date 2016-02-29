int leftPin1 =    9;
int leftPin2 =    9;
int middlePin1 =  9;
int middlePin2 =  9;
int rightPin1 =   9;
int rightPin2 =   9;

void vibrateGroup(int group) {
  digitalWrite(leftPin1, LOW);
  digitalWrite(leftPin2, LOW);
  digitalWrite(middlePin1, LOW);
  digitalWrite(middlePin2, LOW);
  digitalWrite(rightPin1, LOW);
  digitalWrite(rightPin2, LOW);

  switch (group) {
    case 1: // left group
    digitalWrite(leftPin1, HIGH);
    digitalWrite(leftPin2, HIGH);
    break;
    
    case 2: // middle group
    digitalWrite(middlePin1, HIGH);
    digitalWrite(middlePin2, HIGH);
    break;
    
    case 3: // right group
    digitalWrite(rightPin1, HIGH);
    digitalWrite(rightPin2, HIGH);
    break;
  }
}

void setup() {
  // initialize serial:
  Serial.begin(9600);

  // set pins to output
  pinMode(leftPin1, OUTPUT);
  pinMode(leftPin2, OUTPUT);
  pinMode(middlePin1, OUTPUT);
  pinMode(middlePin2, OUTPUT);
  pinMode(rightPin1, OUTPUT);
  pinMode(rightPin2, OUTPUT);
}

void loop() {
  if (Serial.available() < 1)
    return;

  int group = Serial.parseInt();

  vibrateGroup(group);
}

