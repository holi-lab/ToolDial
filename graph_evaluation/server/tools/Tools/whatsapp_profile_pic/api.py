import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def picture_uri(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a whatsapp number profile picture as url encoded data uri"
    phone: The whatsapp number must be written as: countrycode and number; do **NOT** include any non-number character, spaces, or anything which is not a number. Otherwise, the request will not be processed.
Examples: of correct numbers are: 34123456789 (for spain) or 491234567890 (for Germany).
TIPS:
    Do NOT include '+' before your countrycode,
    Do NOT include a '-', or any other character or space (anything which is not a number) as part of your phone number.
    If you do not know which is your country code check this: https://countrycode.org/
        
    """
    url = f"https://whatsapp-profile-pic.p.rapidapi.com/wspic/uri"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-profile-pic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def picture_jpg(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the whatsapp's number profile picture as a jpg file.
		Learn how to use it on this [video](https://youtu.be/fJPgOvEMdOQ)."
    phone: The whatsapp number must be written as: countrycode and number; do **NOT** include any non-number character, spaces, or anything which is not a number. Otherwise, the request will not be processed.
Examples: of correct numbers are: 34123456789 (for spain) or 491234567890 (for Germany).
TIPS:
    Do NOT include '+' before your countrycode,
    Do NOT include a '-', or any other character or space (anything which is not a number) as part of your phone number.
    If you do not know which is your country code check this: https://countrycode.org/
        
    """
    url = f"https://whatsapp-profile-pic.p.rapidapi.com/wspic/png"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-profile-pic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def picture_url(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Url of a whatsapp number profile picture. 
		Learn how to use it on this [video]( https://youtu.be/jtjK6e7huQ0)."
    phone: The whatsapp number must be written as: countrycode and number; do **NOT** include any non-number character, spaces, or anything which is not a number. Otherwise, the request will not be processed.
Examples: of correct numbers are: 34123456789 (for spain) or 491234567890 (for Germany).
TIPS:
    Do NOT include '+' before your countrycode,
    Do NOT include a '-', or any other character or space (anything which is not a number) as part of your phone number.
    If you do not know which is your country code check this: https://countrycode.org/
        
    """
    url = f"https://whatsapp-profile-pic.p.rapidapi.com/wspic/url"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-profile-pic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_info(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetchs: `description`, `website`, `email`, `business hours`, `address` and `category`; if the number is a whatsapp for business account."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        
    """
    url = f"https://whatsapp-profile-pic.p.rapidapi.com/bizinfo"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-profile-pic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def is_a_business(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requests to this endpoint will return `true` if the number is a **Whatsapp for Business** account, or `false` if it is not."
    phone: The whatsapp number must be written as: `countrycode` and `number`; do NOT include any non-number character, spaces, or anything which is not a number.
Examples: of correct numbers are: `34123456789` (for spain) or `491234567890` (for Germany).
Country codes can be checked here: https://countrycode.org/
        
    """
    url = f"https://whatsapp-profile-pic.p.rapidapi.com/isbiz"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-profile-pic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def about(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gather user's own about description."
    phone: The whatsapp number must be written as: countrycode and number; do **NOT** include any non-number character, spaces, or anything which is not a number. Otherwise, the request will not be processed.
Examples: of correct numbers are: 34123456789 (for spain) or 491234567890 (for Germany).
TIPS:
    Do NOT include '+' before your countrycode,
    Do NOT include a '-', or any other character or space (anything which is not a number) as part of your phone number.
    If you do not know which is your country code check this: https://countrycode.org/
        
    """
    url = f"https://whatsapp-profile-pic.p.rapidapi.com/about"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-profile-pic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def whatsapp_number_checker(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the number you want to validate if it exists on the whatsapp network."
    phone: The whatsapp number must be written as: countrycode and number; do **NOT** include any non-number character, spaces, or anything which is not a number. Otherwise, the request will not be processed.
Examples: of correct numbers are: 34123456789 (for spain) or 491234567890 (for Germany).
TIPS:
    Do NOT include '+' before your countrycode,
    Do NOT include a '-', or any other character or space (anything which is not a number) as part of your phone number.
    If you do not know which is your country code check this: https://countrycode.org/
        
    """
    url = f"https://whatsapp-profile-pic.p.rapidapi.com/wchk"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-profile-pic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def picture_with_options(phone: int, quality: str=None, pictype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch the whatsapp number profile picture, you can select its resolution (`high` or `low`) and the response format: `png` file, `url` or `base64` encoded file."
    phone: The phone number must be written as a number (including countrycode); 
do **NOT** include: any non-number character, spaces, or anything which is not a number; do **NOT** add zeros (0) at the beginning.
        quality: Quality of the Picture: `High` or `Low` resolution. Defaults to `High`.
        pictype: The response file type: `url` , `jpg` or `base64`. Defaults to `url`.
        
    """
    url = f"https://whatsapp-profile-pic.p.rapidapi.com/wspicture"
    querystring = {'phone': phone, }
    if quality:
        querystring['quality'] = quality
    if pictype:
        querystring['pictype'] = pictype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-profile-pic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def picture_base64(phone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the base64 encoded file of a whatsapp number profile picture."
    phone: The whatsapp number must be written as: countrycode and number; do **NOT** include any non-number character, spaces, or anything which is not a number. Otherwise, the request will not be processed.
Examples: of correct numbers are: 34123456789 (for spain) or 491234567890 (for Germany).
TIPS:
    Do NOT include '+' before your countrycode,
    Do NOT include a '-', or any other character or space (anything which is not a number) as part of your phone number.
    If you do not know which is your country code check this: https://countrycode.org/
        
    """
    url = f"https://whatsapp-profile-pic.p.rapidapi.com/wspic/b64"
    querystring = {'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whatsapp-profile-pic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

