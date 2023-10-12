import RPi.GPIO as GPIO          
from time import sleep


in1 = 24
in2 = 23
en = 25
temp1 = 1
hall1 = 6
hall2 = 5


GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en,GPIO.OUT)
GPIO.setup(hall1,GPIO.IN, pull_up_down=GPIO.PUD_UP) 
GPIO.setup(hall2,GPIO.IN, pull_up_down=GPIO.PUD_UP) 
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
p=GPIO.PWM(en,1000)
p.start(25)


#This function will be called if a change is detected
def change_detected(channel):
    if GPIO.input(hall1) == GPIO.LOW:
        print('Magnetic material detected')
        GPIO.output(in1,GPIO.HIGH)
        GPIO.output(in2,GPIO.LOW)
    else:
        print('No magnetic material')
        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)

# Register event-listener on falling and raising
# edge on HALL-sensor input. Call "change_detected" as
# callback
GPIO.add_event_detect(hall1, GPIO.BOTH, change_detected, bouncetime=25)

# The main-loop does nothing. All is done by the event-listener
try:
    while True:
        pass

# Quit on Ctrl-c
except KeyboardInterrupt:
    print("Ctrl-C - quit")

# Cleanup GPIO
finally:
    GPIO.cleanup() 