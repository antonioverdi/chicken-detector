# IMPORTS
import RPi.GPIO as GPIO          

topreed=5
bottomreed=6


GPIO.setmode(GPIO.BCM)
GPIO.setup(topreed, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
GPIO.setup(bottomreed, GPIO.IN, pull_up_down=GPIO.PUD_UP) 


if GPIO.input(5):
    print("topreed 1")
else:
    print("topreed 0")

if GPIO.input(6):
    print("bottomreed 1")
else:
    print("bottomread 0")
