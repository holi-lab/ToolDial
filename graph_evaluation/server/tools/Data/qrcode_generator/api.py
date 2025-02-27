import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def wifi(securitytype: str, ssid: str, style: str=None, type: str=None, color: str='000000', backgroundcolor: str='FFFFFF', border: int=1, password: str=None, width: int=512, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "creates a qr code for wifi data"
    securitytype: wifi security
        style: style of the qr-code
        type: image-type of the qr-code
default: PNG
        color: color of the qr-code
default: 000000 (black)
        backgroundcolor: background-color of the qr-code
default: FFFFFF (white)
        border: border of the qr-code
default: 1
        width: width (pixels) of the qr code
        
    """
    url = f"https://qrcode-generator8.p.rapidapi.com/qrcode/api/v1/wifi"
    querystring = {'securitytype': securitytype, 'ssid': ssid, }
    if style:
        querystring['style'] = style
    if type:
        querystring['type'] = type
    if color:
        querystring['color'] = color
    if backgroundcolor:
        querystring['backgroundColor'] = backgroundcolor
    if border:
        querystring['border'] = border
    if password:
        querystring['password'] = password
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-generator8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def email(to: str, border: int=1, color: str='000000', type: str=None, backgroundcolor: str='FFFFFF', width: int=512, style: str=None, body: str=None, subject: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "creates a qr code for sending an email"
    to: recipient of the email
        border: border of the qr-code
default: 1
        color: color of the qr-code
default: 000000 (black)
        type: image-type of the qr-code
default: PNG
        backgroundcolor: background-color of the qr-code
default: FFFFFF (white)
        width: width (pixels) of the qr-code
default: 512
        style: style of the qr-code
        body: body of the email
        subject: subject of the email
        
    """
    url = f"https://qrcode-generator8.p.rapidapi.com/qrcode/api/v1/email"
    querystring = {'to': to, }
    if border:
        querystring['border'] = border
    if color:
        querystring['color'] = color
    if type:
        querystring['type'] = type
    if backgroundcolor:
        querystring['backgroundColor'] = backgroundcolor
    if width:
        querystring['width'] = width
    if style:
        querystring['style'] = style
    if body:
        querystring['body'] = body
    if subject:
        querystring['subject'] = subject
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-generator8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def text(content: str, border: int=1, style: str=None, backgroundcolor: str='FFFFFF', color: str='000000', type: str=None, width: int=512, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "creates a qr code for raw text content"
    content: content of the qr code
        border: border of the qr-code
default: 1
        style: style of the qr-code
        backgroundcolor: background-color of the qr-code
default: FFFFFF (white)
        color: color of the qr-code
default: 000000 (black)
        type: image-type of the qr-code
default: PNG
        width: width (pixels) of the qr code
default: 512
        
    """
    url = f"https://qrcode-generator8.p.rapidapi.com/qrcode/api/v1/text"
    querystring = {'content': content, }
    if border:
        querystring['border'] = border
    if style:
        querystring['style'] = style
    if backgroundcolor:
        querystring['backgroundColor'] = backgroundcolor
    if color:
        querystring['color'] = color
    if type:
        querystring['type'] = type
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-generator8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def website(url: str, backgroundcolor: str='FFFFFF', color: str='000000', border: int=1, style: str=None, type: str=None, width: int=512, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "creates a qr code for a website"
    url: content of the qr code
        backgroundcolor: background-color of the qr-code
default: FFFFFF (white)
        color: color of the qr-code
default: 000000 (black)
        border: border of the qr-code
default: 1
        style: style of the qr-code
        type: image-type of the qr-code
default: PNG
        width: width (pixels) of the qr code
default: 512
        
    """
    url = f"https://qrcode-generator8.p.rapidapi.com/qrcode/api/v1/website"
    querystring = {'url': url, }
    if backgroundcolor:
        querystring['backgroundColor'] = backgroundcolor
    if color:
        querystring['color'] = color
    if border:
        querystring['border'] = border
    if style:
        querystring['style'] = style
    if type:
        querystring['type'] = type
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-generator8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

