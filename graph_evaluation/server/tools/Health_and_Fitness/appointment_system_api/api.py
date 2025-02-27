import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_appointments(date: str, clinicname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "EndPoints returns all appointments in the date. 
		
		Parameter Description :
		date : the date the appointments is requested
		clinicName : Name of the clinic
		
		Return Value : 
		It returns json object. Json contains all the appointments in requested date.
		
		Json Format :
		
		hLabel:"07:00"
		id:5
		aDay:"2022-09-09T00:00:00"
		aHour:58
		aName:"Efe Kemal TASAR"
		aTel:"+905376853054"
		aType:"Appoint""
    
    """
    url = f"https://appointment-system-api.p.rapidapi.com/appointment/getAppointments"
    querystring = {'date': date, 'clinicName': clinicname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appointment-system-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def appointment_time_list(date: str, clinicname: str='DemoClinic', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "EndPoint gets Appointment Time List by Day. EndPoint gets date parameter and returns List of Appointment Times of that date. In the list there is available field, This field indicates whether there is an appointment at the specified time.
		Parameter Descriptions : 
		date --> The day that hours will be return
		clinicName --> Name of the clinic
		
		Return Value: 
		The EndPoint returns json object in format below.
		[
		{"id":1
		"hLabel":"07:00"
		"available":"true"
		"hour":7
		"minute":0
		"interval":15
		},
		......
		]
		
		Id --> is the key value of the record.
		hLabel --> is the string format of the time
		available --> if is true that means the time is suitable for new record. If it is false that means there is another appointment in this time interval. 
		hour --> Hour of the appointment time.
		minute --> Minute of the appointment time.
		interval  --> this field points out to the interval whether 15 minutes or 30 minutes"
    
    """
    url = f"https://appointment-system-api.p.rapidapi.com/appointment/getHoursByDayByClinic"
    querystring = {'date': date, }
    if clinicname:
        querystring['clinicName'] = clinicname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appointment-system-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def login(password: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint is use for login to the system and getting the Api Key for the clinic. It needs username and password as query parameters. 
		Parameter Descriptions:
		username --> parameter for username in the system. It is the same with username while creating the account.
		password --> password for the username. It is same with the password while creating the acoount.
		
		Return Value
		This Endpoint returns string value.
		If you get a sting with 6+ long this means that you have successfully passed username and password check and you are getting the Accesskey fort he system.
		If you get “W1” it means a warning that your username doesn’t exists
		If you get “W2” it means a warning that your password is not correct.
		It you get “E1” it means a programmatic internal error. If you get E1 in the next calls you can get help from us."
    
    """
    url = f"https://appointment-system-api.p.rapidapi.com/appointment/login"
    querystring = {'password': password, 'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "appointment-system-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

