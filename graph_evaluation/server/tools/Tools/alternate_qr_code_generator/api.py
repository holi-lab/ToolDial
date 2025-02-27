import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text_qr_code(redirect_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to create a QR Code containing any text, in particular, an URL that may redirect the user to the website. After QR code is scanned, website will be displayed to the user."
    
    """
    url = f"https://alternate-qr-code-generator.p.rapidapi.com/qr"
    querystring = {'redirect_url': redirect_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alternate-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def email_qr_code(email_address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows the creation of a QR Code allowing the user to quickly send an email. The code contains an appropriately encoded message template. After scanning, the device starts the e-mail client with pre-filled specified fields."
    
    """
    url = f"https://alternate-qr-code-generator.p.rapidapi.com/qr"
    querystring = {'email_address': email_address, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alternate-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contact_qr_code(phone_number: str, country_code: str, name: str, website: str='www.asdf.com', title: str='self promoted', organization: str='I have no idea', photo_url: str='https://asdf.com/asdf.jpg', mobile_number: str='1234509876', email: str='asdf@asdf.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to create a QR Code that allows user to quickly add contact information to the phone book. The code contains an appropriately encoded electronic business card. After scanning, the device prompts to save the contact in the phone book."
    
    """
    url = f"https://alternate-qr-code-generator.p.rapidapi.com/qr"
    querystring = {'phone_number': phone_number, 'country_code': country_code, 'name': name, }
    if website:
        querystring['website'] = website
    if title:
        querystring['title'] = title
    if organization:
        querystring['organization'] = organization
    if photo_url:
        querystring['photo_url'] = photo_url
    if mobile_number:
        querystring['mobile_number'] = mobile_number
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alternate-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wifi_qr_code(wifi_encryption: str, wifi_ssid: str, wifi_password: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to create a QR Code that allows user to quickly connect to a WiFi network. The code contains properly encoded network credentials. After scanning, the device can automatically connect to the network without having to enter the password manually."
    
    """
    url = f"https://alternate-qr-code-generator.p.rapidapi.com/qr"
    querystring = {'wifi_encryption': wifi_encryption, 'wifi_ssid': wifi_ssid, 'wifi_password': wifi_password, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alternate-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

