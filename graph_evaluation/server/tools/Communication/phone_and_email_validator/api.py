import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getemailvalidity(email: str, dns: bool=None, blacklist: bool=None, format: bool=None, smtp: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will check the validity of an email and return a Boolean. Specific aspects of the email can be used to check its validity"
    dns: Will compare the email to DNS mx-records. Default is true
        blacklist: Will verify if the email is in the blacklist of domains  from DadouData. Default is true
        format: Will check if the format of the email is correct. Defaults to true
        smtp: Will actually check if the email is true by establishing an  SMTP conversation. Default is true
        
    """
    url = f"https://phone-and-email-validator1.p.rapidapi.com/email"
    querystring = {'email': email, }
    if dns:
        querystring['dns'] = dns
    if blacklist:
        querystring['blacklist'] = blacklist
    if format:
        querystring['format'] = format
    if smtp:
        querystring['smtp'] = smtp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-and-email-validator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getphonedetails(number: str, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter phone number to check its validity and obtain details if its real. If the country code is not specified. The location of request's ip address is considered.  If the country prefix is specified directly in the phone number, country code is not required."
    
    """
    url = f"https://phone-and-email-validator1.p.rapidapi.com/phone"
    querystring = {'number': number, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "phone-and-email-validator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

