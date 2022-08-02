from w1thermsensor import W1Thermsensor
    # Read the data from the sensor
TempSensor =W1Thermsensor()
    # Print sensor info
print("The sensor info = ", TempSensor)
    # Read the temperature GPIO4 pin7
temperature = TempSensor.get_temperature()
    # Print the temperature in Celsius, Fahrenheit and Kelvin
print("Temperature in Celsius = ", temperature)
fahrenheit = (temperature * 9/5) + 32
print("Temperature in Fahrenheit = ", fahrenheit)
kelvin = temperature + 273.15
print("Temperature in Kelvin = ", kelvin)
