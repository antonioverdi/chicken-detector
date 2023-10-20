# IMPORTS
import RPi.GPIO as GPIO          

topreed=5
bottomreed=6


GPIO.setmode(GPIO.BCM)
GPIO.setup(topreed, GPIO.IN) 
GPIO.setup(bottomreed, GPIO.IN) 


if topreed:
    print("topreed 1")
else:
    print("topreed 0")

if bottomreed:
    print("bottomreed 1")
else:
    print("bottomread 0")
