# IMPORTS
import RPi.GPIO as GPIO          

topreed=5
bottomreed=6


GPIO.setmode(GPIO.BCM)
GPIO.setup(topreed, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
GPIO.setup(bottomreed, GPIO.IN, pull_up_down=GPIO.PUD_UP) 


if topreed==1:
    print("topreed 1")

if topreed==0:
    print("topreed 0")

if bottomreed==1:
    print("bottomreed 1")

if bottomreed==0:
    print("bottomread 0")
