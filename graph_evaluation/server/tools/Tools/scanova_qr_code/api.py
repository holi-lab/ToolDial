import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sms_qr_code(phone_no: str, message: str, data_gradient_start_color: str='#000000', eye_color_outer: str='#000000', data_gradient_style: str='None', size: str='m', background_color: str='#FFFFFF', data_gradient_end_color: str='#000000', eye_pattern: str='RECT_RECT', data_pattern: str='RECT', error_correction: str='M', eye_color_inner: str='#000000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request returns a Static QR Code (standard or custom-designed) encoded with the recipient phone number and pre-loaded message. When scanned, the QR Code will prompt the scanning device to 'send SMS' pre-loaded with the recipient number and message."
    
    """
    url = f"https://scanova-qr-code.p.rapidapi.com/v2/qrcode/SMS"
    querystring = {'phone_no': phone_no, 'message': message, }
    if data_gradient_start_color:
        querystring['data_gradient_start_color'] = data_gradient_start_color
    if eye_color_outer:
        querystring['eye_color_outer'] = eye_color_outer
    if data_gradient_style:
        querystring['data_gradient_style'] = data_gradient_style
    if size:
        querystring['size'] = size
    if background_color:
        querystring['background_color'] = background_color
    if data_gradient_end_color:
        querystring['data_gradient_end_color'] = data_gradient_end_color
    if eye_pattern:
        querystring['eye_pattern'] = eye_pattern
    if data_pattern:
        querystring['data_pattern'] = data_pattern
    if error_correction:
        querystring['error_correction'] = error_correction
    if eye_color_inner:
        querystring['eye_color_inner'] = eye_color_inner
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scanova-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def email_qr_code(address: str, background_color: str='#FFFFFF', eye_pattern: str='RECT_RECT', error_correction: str='M', size: str='m', data_gradient_start_color: str='#000000', data_gradient_style: str='None', data_pattern: str='RECT', eye_color_outer: str='#000000', data_gradient_end_color: str='#000000', eye_color_inner: str='#000000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request returns a Static QR Code (standard or custom-designed) encoded with the email address. When scanned, the QR Code will prompt the user to 'send an email' to the specified email address."
    
    """
    url = f"https://scanova-qr-code.p.rapidapi.com/v2/qrcode/email"
    querystring = {'address': address, }
    if background_color:
        querystring['background_color'] = background_color
    if eye_pattern:
        querystring['eye_pattern'] = eye_pattern
    if error_correction:
        querystring['error_correction'] = error_correction
    if size:
        querystring['size'] = size
    if data_gradient_start_color:
        querystring['data_gradient_start_color'] = data_gradient_start_color
    if data_gradient_style:
        querystring['data_gradient_style'] = data_gradient_style
    if data_pattern:
        querystring['data_pattern'] = data_pattern
    if eye_color_outer:
        querystring['eye_color_outer'] = eye_color_outer
    if data_gradient_end_color:
        querystring['data_gradient_end_color'] = data_gradient_end_color
    if eye_color_inner:
        querystring['eye_color_inner'] = eye_color_inner
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scanova-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phone_call_qr_code(number: str, data_gradient_style: str='None', eye_pattern: str='RECT_RECT', data_gradient_start_color: str='#000000', error_correction: str='M', eye_color_inner: str='#000000', data_pattern: str='RECT', background_color: str='#FFFFFF', data_gradient_end_color: str='#000000', eye_color_outer: str='#000000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request returns a Static QR Code (standard or custom-designed) encoded with the phone number. When scanned, the QR Code will prompt the scanning device to 'dial the number'."
    
    """
    url = f"https://scanova-qr-code.p.rapidapi.com/v2/qrcode/phonecall"
    querystring = {'number': number, }
    if data_gradient_style:
        querystring['data_gradient_style'] = data_gradient_style
    if eye_pattern:
        querystring['eye_pattern'] = eye_pattern
    if data_gradient_start_color:
        querystring['data_gradient_start_color'] = data_gradient_start_color
    if error_correction:
        querystring['error_correction'] = error_correction
    if eye_color_inner:
        querystring['eye_color_inner'] = eye_color_inner
    if data_pattern:
        querystring['data_pattern'] = data_pattern
    if background_color:
        querystring['background_color'] = background_color
    if data_gradient_end_color:
        querystring['data_gradient_end_color'] = data_gradient_end_color
    if eye_color_outer:
        querystring['eye_color_outer'] = eye_color_outer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scanova-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_map_qr_code(longitude: str, latitude: str, eye_color_inner: str='#000000', eye_pattern: str='RECT_RECT', data_gradient_end_color: str='#000000', data_gradient_style: str='None', data_gradient_start_color: str='#000000', size: str='m', data_pattern: str='RECT', eye_color_outer: str='#000000', error_correction: str='M', background_color: str='#FFFFFF', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request returns a Static QR Code (standard or custom-designed) encoded with a Google Maps URL linked to specified map coordinates. When scanned, the QR Code prompts the scanning device to open the URL in a mobile browser or Google Maps mobile application (if installed)"
    
    """
    url = f"https://scanova-qr-code.p.rapidapi.com/v2/qrcode/googlemaps"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if eye_color_inner:
        querystring['eye_color_inner'] = eye_color_inner
    if eye_pattern:
        querystring['eye_pattern'] = eye_pattern
    if data_gradient_end_color:
        querystring['data_gradient_end_color'] = data_gradient_end_color
    if data_gradient_style:
        querystring['data_gradient_style'] = data_gradient_style
    if data_gradient_start_color:
        querystring['data_gradient_start_color'] = data_gradient_start_color
    if size:
        querystring['size'] = size
    if data_pattern:
        querystring['data_pattern'] = data_pattern
    if eye_color_outer:
        querystring['eye_color_outer'] = eye_color_outer
    if error_correction:
        querystring['error_correction'] = error_correction
    if background_color:
        querystring['background_color'] = background_color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scanova-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wifi_qr_code(authentication: str, size: str, ssid: str, data_gradient_end_color: str='#000000', eye_pattern: str='RECT_RECT', error_correction: str='M', eye_color_inner: str='#000000', eye_color_outer: str='#000000', data_gradient_style: str='None', background_color: str='#FFFFFF', data_pattern: str='RECT', data_gradient_start_color: str='#000000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request returns a Static QR Code (standard or custom-designed) encoded with WiFi network details. When scanned, the QR Code prompts the scanning device to 'Join the Network'. Note that this QR Code works only for Android devices. iPhones do not allow automatic joining of WiFi networks and will display network details as text."
    
    """
    url = f"https://scanova-qr-code.p.rapidapi.com/v2/qrcode/wifi"
    querystring = {'authentication': authentication, 'size': size, 'ssid': ssid, }
    if data_gradient_end_color:
        querystring['data_gradient_end_color'] = data_gradient_end_color
    if eye_pattern:
        querystring['eye_pattern'] = eye_pattern
    if error_correction:
        querystring['error_correction'] = error_correction
    if eye_color_inner:
        querystring['eye_color_inner'] = eye_color_inner
    if eye_color_outer:
        querystring['eye_color_outer'] = eye_color_outer
    if data_gradient_style:
        querystring['data_gradient_style'] = data_gradient_style
    if background_color:
        querystring['background_color'] = background_color
    if data_pattern:
        querystring['data_pattern'] = data_pattern
    if data_gradient_start_color:
        querystring['data_gradient_start_color'] = data_gradient_start_color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scanova-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def text_qr_code(data: str, size: str, data_gradient_end_color: str='#000000', eye_pattern: str='RECT_RECT', data_pattern: str='RECT', data_gradient_start_color: str='#000000', error_correction: str='M', data_gradient_style: str='None', eye_color_inner: str='#000000', eye_color_outer: str='#000000', background_color: str='#FFFFFF', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request returns a Static QR Code (standard or custom-designed) encoded with specified text. When scanned, the QR Code prompts the scanning device to display the text."
    
    """
    url = f"https://scanova-qr-code.p.rapidapi.com/v2/qrcode/text"
    querystring = {'data': data, 'size': size, }
    if data_gradient_end_color:
        querystring['data_gradient_end_color'] = data_gradient_end_color
    if eye_pattern:
        querystring['eye_pattern'] = eye_pattern
    if data_pattern:
        querystring['data_pattern'] = data_pattern
    if data_gradient_start_color:
        querystring['data_gradient_start_color'] = data_gradient_start_color
    if error_correction:
        querystring['error_correction'] = error_correction
    if data_gradient_style:
        querystring['data_gradient_style'] = data_gradient_style
    if eye_color_inner:
        querystring['eye_color_inner'] = eye_color_inner
    if eye_color_outer:
        querystring['eye_color_outer'] = eye_color_outer
    if background_color:
        querystring['background_color'] = background_color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scanova-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v_card_qr_code(uname: str, error_correction: str='M', data_gradient_start_color: str='#000000', eye_color_inner: str='#000000', data_gradient_style: str='None', data_gradient_end_color: str='#000000', background_color: str='#FFFFFF', eye_color_outer: str='#000000', eye_pattern: str='RECT_RECT', size: str='m', data_pattern: str='RECT', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request returns a Static QR Code (standard or custom-designed) encoded with specified contact information. When scanned, the QR Code prompts the scanning device to 'Save as Contact'."
    
    """
    url = f"https://scanova-qr-code.p.rapidapi.com/v2/qrcode/vcard"
    querystring = {'uname': uname, }
    if error_correction:
        querystring['error_correction'] = error_correction
    if data_gradient_start_color:
        querystring['data_gradient_start_color'] = data_gradient_start_color
    if eye_color_inner:
        querystring['eye_color_inner'] = eye_color_inner
    if data_gradient_style:
        querystring['data_gradient_style'] = data_gradient_style
    if data_gradient_end_color:
        querystring['data_gradient_end_color'] = data_gradient_end_color
    if background_color:
        querystring['background_color'] = background_color
    if eye_color_outer:
        querystring['eye_color_outer'] = eye_color_outer
    if eye_pattern:
        querystring['eye_pattern'] = eye_pattern
    if size:
        querystring['size'] = size
    if data_pattern:
        querystring['data_pattern'] = data_pattern
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scanova-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def website_url_qr_code(error_correction: str='H', eye_pattern: str='RECT_RECT', data_gradient_style: str='None', format: str='png', size: str='m', eye_color_outer: str='#000000', data_gradient_end_color: str='#000000', data_gradient_start_color: str='#000000', eye_color_inner: str='#000000', url: str='https://scanova.io', data_pattern: str='RECT', background_color: str='#FFFFFF', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This request returns a Static QR Code (standard or custom-designed) encoded with a specified URL"
    
    """
    url = f"https://scanova-qr-code.p.rapidapi.com/v2/qrcode/url"
    querystring = {}
    if error_correction:
        querystring['error_correction'] = error_correction
    if eye_pattern:
        querystring['eye_pattern'] = eye_pattern
    if data_gradient_style:
        querystring['data_gradient_style'] = data_gradient_style
    if format:
        querystring['format'] = format
    if size:
        querystring['size'] = size
    if eye_color_outer:
        querystring['eye_color_outer'] = eye_color_outer
    if data_gradient_end_color:
        querystring['data_gradient_end_color'] = data_gradient_end_color
    if data_gradient_start_color:
        querystring['data_gradient_start_color'] = data_gradient_start_color
    if eye_color_inner:
        querystring['eye_color_inner'] = eye_color_inner
    if url:
        querystring['url'] = url
    if data_pattern:
        querystring['data_pattern'] = data_pattern
    if background_color:
        querystring['background_color'] = background_color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scanova-qr-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

