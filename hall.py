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
GPIO.setup(hall1,GPIO.IN) 
GPIO.setup(hall2,GPIO.IN) 
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
p=GPIO.PWM(en,1000)
p.start(25)


# The main-loop does nothing. All is done by the event-listener
try:
    while True:
        if(GPIO.input(hall1) == False):
            GPIO.output(in1,GPIO.HIGH)
            GPIO.output(in2,GPIO.LOW)
            print("magnet detected")
        else:
            GPIO.output(in1,GPIO.LOW)
            GPIO.output(in2,GPIO.LOW)
            print("magnetic field not detected")

# Quit on Ctrl-c
except KeyboardInterrupt:
    print("Ctrl-C - quit")

# Cleanup GPIO
finally:
    GPIO.cleanup() 