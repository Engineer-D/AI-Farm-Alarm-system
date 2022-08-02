

import serial #for serial communication with GSM SIM800L
import time 
import pygame #to play music


# _____________________________________________________________________________#
# Intro text
print("Setting up Raspberry PI IVR")




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

#checks if SIM800L is speaking and returns it as response
def wait_for_SIM800():
    echo = ser.readline()  # waste the echo
    response_byte = ser.readline()
    response_str = response_byte.decode('ascii')
    return (response_str)

#Checks SIM800L status and connects with ShopifyAPI
def Init_GSM():

    if "OK" in SIM800("AT"):
        if ("OK" in (SIM800("AT+DDET=1"))) and ("OK" in (SIM800("AT+CNMI =0,0,0,0,0"))) and ("OK" in (SIM800("AT+CMGF=1"))) and ("OK" in (SIM800("AT+CSMP=17,167,0,0"))):  # enble DTMF / disable notifications
            print("SIM800 Module -> Active and Ready")
    else:
        print("------->ERROR -> SIM800 Module not found")

#plays the given wav file #8000Mhz mono audio WAV works best on SIM800L
def play_wav(file_name):
    pygame.mixer.init(8000)
    pygame.mixer.music.load(file_name)
    pygame.mixer.music.play()
    #while pygame.mixer.music.get_busy() == True:
        #continue

# Makes a call to given number and returns NONE, NOT_REACHABLE, CALL_REJECTED, REJECTED_AFTER_ANSWERING,  REQ_CALLBACK,CANCELED, CONFIRMED
def Call_response_for (phone_number):
    AT_call = "ATD" + phone_number + ";"
    response = "NONE"

    time.sleep(1)
    ser.flushInput() #clear serial data in buffer if any

    if ("OK" in (SIM800(AT_call))) and (",2," in (wait_for_SIM800())) and (",3," in (wait_for_SIM800())):
        print("RINGING...->", phone_number)
        call_status = wait_for_SIM800()
        if "1,0,0,0,0" in call_status:
            print("**ANSWERED**")
            ser.flushInput()
            play_wav("intro.wav")
            time.sleep(0.5)
            dtmf_response = "start_over"
            while dtmf_response == "start_over":
                play_wav("press_request.wav")
                time.sleep(1)
                dtmf_response = wait_for_SIM800()
                if "+DTMF: 1" in dtmf_response:
                    play_wav("confirmed.wav")
                    response = "CONFIRMED"
                    hang = SIM800("ATH")
                    break
                if "+DTMF: 2" in dtmf_response:
                    play_wav("canceled.wav")
                    response = "CANCELED"
                    hang = SIM800("ATH")
                    break
                if "+DTMF: 9" in dtmf_response:
                    play_wav("callback_response.wav")
                    response = "REQ_CALLBACK"
                    hang = SIM800("ATH")
                    break
                if "+DTMF: 0" in dtmf_response:
                    dtmf_response = "start_over"
                    continue
                if "+DTMF: " in dtmf_response:
                    play_wav("invalid_input.wav")
                    dtmf_response = "start_over"
                    continue
                else:
                    response = "REJECTED_AFTER_ANSWERING"
                    break
        else:
            #print("REJECTED")
            response = "CALL_REJECTED"
            hang = SIM800("ATH")
            time.sleep(1)
            #ser.flushInput()
    else:
        #print("NOT_REACHABLE")
        response = "NOT_REACHABLE"
        hang = SIM800("ATH")
        time.sleep(1)
        #ser.flushInput()

    ser.flushInput()
    return (response)

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

def incoming_call():
    while ser.in_waiting: #if I have something in the serial monitor
        print ("%%Wait got something in the buffer")
        ser.flushInput()
        response = SIM800("ATH") #cut the incoming call
        if "+CLCC" in response:
            cus_phone = response[21:31]
            print("%%Incoming Phone call detect from ->", cus_phone)
            return (cus_phone)
        else:
            print("Nope its something else")
            return "0"
    return "0"

cus_name = "Aisha"
cus_phone = "96883XXXXX"

while (1): #Infinite loop

    # COM defanition for windows -> Should change for Pi
    ser = serial.Serial("/dev/ttyUSB0", baudrate=9600, timeout=15)  # timeout affects call duration and waiting for response currently 30sec
    print("Established communication with", ser.name)

    Init_GSM() #check if GSM is connected and initialize it

    print("_____________________IVR START___________________")


    response = Call_response_for(cus_phone) #place a call and get response from customer
    print ("Response from customer => ", response)

    if response == "CONFIRMED":
        text_message = "Hi " + cus_name + ". Your booking has been confirmed. Thank you!!. -Circuitdigest"
        send_message(text_message, cus_phone)

    if response == "CANCELED":  # if the response was to cancel
        text_message = "Hi " + cus_name + ". Sorry that you have decided to cancel your booking. If you cancled by mistake, kindly contact us through phone. -Circuitdigest"
        send_message(text_message, cus_phone)

    if ((response == "CALL_REJECTED") or (response == "REJECTED_AFTER_ANSWERING")):  # if the response was rejected
        text_message = "Hi " + cus_name + ". We from circuitdigest.com have been trying to reach you, to confirm your booking. You will receive another call within few minutes, we kindly request you to answer it. Thank you"
        send_message(text_message, cus_phone)

    print("_____________________IVR END___________________")

    ser.close()
    time.sleep (5)













