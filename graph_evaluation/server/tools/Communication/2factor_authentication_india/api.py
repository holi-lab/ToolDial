import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_balance_addon_services(service_name: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check Balance For Addon Services"
    service_name: Name of the addon service
        api_key: 2Factor account API Key
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/ADDON_SERVICES/BAL/{service_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sending_voice_otp_custom_otp(otp: int, phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to send VOICE OTP to India"
    otp: 4 Digit ( Numeric ) OTP code to be sent
        phone_number: 10 Digit Indian Phone Number
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/VOICE/{phone_number}/{otp}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sending_sms_otp_custom_otp(otp: str, phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to send Custom SMS OTP to India"
    otp: 4-6 Digit ( Numeric ) OTP code to be sent
        phone_number: 10 Digit Indian Phone Number
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/SMS/{phone_number}/{otp}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_sms_otp_input(otp_input: str, session_id: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is useful in verifying user entered OTP with sent OTP"
    otp_input: OTP Value input by end user
        session_id: Verification session id returned in send OTP step
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/SMS/VERIFY/{session_id}/{otp_input}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sending_sms_otp_auto_generated_otp(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to send Auto Generated SMS OTP to India"
    phone_number: 10 Digit Indian Phone Number
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/SMS/{phone_number}/AUTOGEN"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify_voice_otp_input(otp_input: str, session_id: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is useful in verifying user entered OTP with sent OTP"
    otp_input: OTP Value input by end user
        session_id: Verification session id returned in send OTP step
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/VOICE/VERIFY/{session_id}/{otp_input}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sending_sms_otp_auto_generated_otp_custom_template(phone_number: str, template_name: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to send Auto Generated SMS OTP to India"
    phone_number: 10 Digit Indian Phone Number
        template_name: Template name created using Custom Template Wizard
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/SMS/{phone_number}/AUTOGEN/{template_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sending_sms_otp_custom_otp_custom_template(otp: str, phone_number: str, template_name: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to send SMS OTP to India"
    otp: 4 Digit ( Numeric ) OTP code to be sent
        phone_number: 10 Digit Indian Phone Number
        template_name: Template name created using Custom Template Wizard
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/SMS/{phone_number}/{otp}/{template_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sending_voice_otp_auto_generated_otp(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to send Auto Generated VOICE OTP to India"
    phone_number: 10 Digit Indian Phone Number
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/VOICE/{phone_number}/AUTOGEN"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_number_sms_service(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to add number to SMS Blocklist"
    phone_number: 10 Digit Indian Phone Number
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/BLOCK/{phone_number}/SMS/ADD"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def un_block_number_sms_service(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to remove number from SMS Blocklist"
    phone_number: 10 Digit Indian Phone Number
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/BLOCK/{phone_number}/SMS/REMOVE"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def un_block_number_voice_service(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to remove number from VOICE Blocklist"
    phone_number: 10 Digit Indian Phone Number
        api_key: Get one from http://2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/BLOCK/{phone_number}/VOICE/REMOVE"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_number_voice_service(phone_number: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to add number to VOICE Blocklist"
    phone_number: 10 Digit Indian Phone Number
        api_key: Get one from http://2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/BLOCK/{phone_number}/VOICE/ADD"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pull_delivery_report(session_id: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pull Delivery Report - Transactional SMS"
    session_id: Session Id Returned By Send SMS Step
        api_key: API Obtained From 2Factor.in
        
    """
    url = f"https://2factor.p.rapidapi.com/API/V1/{api_key}/ADDON_SERVICES/RPT/TSMS/{session_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "2factor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

