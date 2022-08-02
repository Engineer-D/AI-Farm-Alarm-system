import serial #for serial communication with GSM SIM800L
import time 

# _____________________________________________________________________________#
# Intro text

print("Setting up Raspberry No Reply bot")

#Speak with SIM800 -> gets AT command return as response
def SIM800(command):
    AT_command = command + "\r\n"
    ser.write(str(AT_command).encode('ascii'))
    time.sleep(1)
    if ser.inWaiting() > 0:
        echo = ser.readline() #waste the echo
        response_byte = ser.readline()
        response_str = response_byte.decode('ascii')
        return (response_str)
    else:
        return ("ERROR")

#Checks SIM800L status and connects with ShopifyAPI
def Init_GSM():
    if "OK" in SIM800("AT"):
        if ("OK" in (SIM800("AT+CMGF=1"))) and ("OK" in (SIM800("AT+CSMP=17,167,0,0"))):  # enble sms settings
            print("SIM800 Module -> Active and Ready")
    else:
        print("------->ERROR -> SIM800 Module not found")

#Receives the message and phone number and send that message to that phone number
def send_message(message, recipient):
    ser.write(b'AT+CMGS="' + recipient.encode() + b'"\r')
    time.sleep(0.5)
    ser.write(message.encode() + b"\r")
    time.sleep(0.5)
    ser.write(bytes([26]))
    time.sleep(0.5)
    print ("Message sent to customer")
    time.sleep(2)
    ser.flushInput()  # clear serial data in buffer if any

cus_name = "DeeKay"
cus_phone = "08166700905"

while (1): #Infinite loop
    # COM defanition for windows -> Should change for Pi
    ser = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=15)  # timeout affects call duration and waiting for response currently 30sec
    print("Established communication with", ser.name)
    Init_GSM() #check if GSM is connected and initialize it
    print("_____________________No Reply Bot START___________________")
    response = input("Input a response: ") #get response from customer
    print ("Response from customer => ", response.upper())
    if response.upper() == "CONFIRMED":
        text_message = "Hi " + cus_name + ". Your booking has been confirmed. Thank you!!. -Engineer-D Innovations"
        send_message(text_message, cus_phone)
    if response.upper() == "CANCELED":  # if the response was to cancel
        text_message = "Hi " + cus_name + ". Sorry that you have decided to cancel your booking. If you cancelled by mistake, kindly contact us through phone. -Engineer-D Innovations"
        send_message(text_message, cus_phone)
    if ((response.upper() == "CALL_REJECTED") or (response == "REJECTED_AFTER_ANSWERING")):  # if the response was rejected
        text_message = "Hi " + cus_name + ". We from Engineer-D Innovations have been trying to reach you, to confirm your booking. You will receive another call within few minutes, we kindly request you to answer it. Thank you"
        send_message(text_message, cus_phone)
    print("_____________________No Reply Bot END___________________")
    ser.close()
    time.sleep (5)