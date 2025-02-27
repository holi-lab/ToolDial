import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def delivery_report(phonenumbers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives the Delivery report of SMS"
    phonenumbers: Multiple MessageID's seperated by Comma
        
    """
    url = f"https://gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com/report/{phonenumbers}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_sms_transactional(message: str, phonenumbers: str, unicode: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to send Direct Messages to the User. Pricing is based on the noof unit SMS consumed and no of phonenumbers used. 1 unit SMS = 140 characters. For each unit, 1 credit will be used. You will be billed only if the message is sent to the user"
    phonenumbers: Type Contact Numbers(10 digits only) seperated by Comma. Maximum 500 Numbers at a time
        unicode: Unicode messages are limited to 70 characters/unit SMS. Set value "true" to support all regional languages of india
        
    """
    url = f"https://gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com/sms/transactional/{phonenumbers}/{message}"
    querystring = {}
    if unicode:
        querystring['unicode'] = unicode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_sms_flash_transactional(message: str, phonenumbers: str, unicode: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send Flash SMS, You will be billed only if the message is sent to the user"
    phonenumbers: Type Contact Numbers(10 digits only) seperated by Comma. Maximum 500 Numbers at a time
        unicode: Unicode messages are limited to 70 characters/unit SMS. Set value "true" to support all regional languages of india
        
    """
    url = f"https://gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com/flashsms/transactional/{phonenumbers}/{message}"
    querystring = {}
    if unicode:
        querystring['unicode'] = unicode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_otp(duration: str, digits: str, otp: str, phonenumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to verify the OTP. the Phonenumber, duration and number of digitis should match with Send OTP endpoint. This Endpoint us free to Use"
    duration: The Period of valid OTP (in Seconds). (TTL). Should be same sa Send OTP
        digits: Number of Digits in OTP. Sould be same as Send OTP
        otp: THis the value recived from SMS sent to User. You can also get this by setting getOTP true in Send OTP
        
    """
    url = f"https://gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com/otp/verify/{phonenumber}/{digits}/{duration}/{otp}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_otp(digits: int, duration: int, message: str, phonenumber: str, messagetemplate: str, getotp: str='True', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate, Send and Get OTP. The best and easiest way of delivering OTPs is through the SMS channel. SMSly offers One-Time Passwords (OTPs) SMS delivery services which can be further used to authenticate a user or for account verification.  You will be billed only if message is sent to the user"
    digits: Number of Digits in OTP
        duration: The Period of valid OTP (in Seconds). (TTL)
        message: Write your own template containing OTP_VALUE. this OTP_VALUE will be replaced by the generated OTP. Example: OTP_VALUE is your App verification code.
        duration: The Period of valid OTP (in Seconds). (TTL)
        digits: Number of DIgits in OTP
        messagetemplate: Write your own template containing OTP_VALUE. this OTP_VALUE will be replaced by the generated OTP. Example: OTP_VALUE is your App verification code.
        getotp: Get generated OTP in response of the API
        
    """
    url = f"https://gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com/otp/generate/{phonenumber}"
    querystring = {'digits': digits, 'duration': duration, 'message': message, 'messagetemplate': messagetemplate, }
    if getotp:
        querystring['getOTP'] = getotp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gurubrahma-smsly-sms-to-india-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

