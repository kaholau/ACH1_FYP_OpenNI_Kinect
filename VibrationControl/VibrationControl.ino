int leftPin1 =    8;
int middlePin1 =  9;
int rightPin1 =   10;
int lastCmd = -1;
int group;

void vibrateGroup(int group) {

  switch (group) {
    case 1:
    case 49: // left group
      {
        digitalWrite(leftPin1, LOW);
        digitalWrite(middlePin1, HIGH);
        digitalWrite(rightPin1, HIGH);
        break;
      }

    case 2:
    case 50: // middle_left group
      {
        digitalWrite(leftPin1, LOW);
        digitalWrite(middlePin1, LOW);
        digitalWrite(rightPin1, HIGH);
        break;
      }
    case 3:
    case 51: // middle_middle group
      {
        digitalWrite(leftPin1, HIGH);
        digitalWrite(middlePin1, LOW);
        digitalWrite(rightPin1, HIGH);
        break;
      }
    case 4:
    case 52: // middle_right group
      {
        digitalWrite(leftPin1, HIGH);
        digitalWrite(middlePin1, LOW);
        digitalWrite(rightPin1, LOW);
        break;
      }
    case 5:
    case 53: // right group
      {
        digitalWrite(leftPin1, HIGH);
        digitalWrite(middlePin1, HIGH);
        digitalWrite(rightPin1, LOW);
        break;
      }

    default:
      {
        digitalWrite(leftPin1, HIGH);
        digitalWrite(middlePin1, HIGH);
        digitalWrite(rightPin1, HIGH);
        break;
      }
  }
}

void setup() {
  // initialize serial:
  Serial.begin(9600);

  // set pins to output
  pinMode(leftPin1, OUTPUT);
  pinMode(middlePin1, OUTPUT);
  pinMode(rightPin1, OUTPUT);

  digitalWrite(leftPin1, HIGH);
  digitalWrite(middlePin1, HIGH);
  digitalWrite(rightPin1, HIGH);
}

void loop() {
  if (Serial.available() < 1)
    return;

  group = Serial.read();
  Serial.println(group);
  if (group != lastCmd)
  {
    vibrateGroup(group);
    lastCmd = group;
  }
}
