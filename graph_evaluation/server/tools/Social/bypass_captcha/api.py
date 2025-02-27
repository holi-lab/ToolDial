import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def captchatypers_com_forms_requestbalance_htm(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request Balance"
    
    """
    url = f"https://imagetypers-bypass-captcha-v1.p.rapidapi.com//captchatypers.com/Forms/RequestBalance.htm"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagetypers-bypass-captcha-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def captchatypers_com_forms_fileuploadandgettextcaptchaurl_htm(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "File Upload And GetT ext Captcha URL"
    
    """
    url = f"https://imagetypers-bypass-captcha-v1.p.rapidapi.com//captchatypers.com/forms/FileUploadAndGetTextCaptchaURL.htm"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "imagetypers-bypass-captcha-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

