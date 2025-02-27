import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_for_figma(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to download a QR code for your Figma URL. If you have a password.
		
		Instructions:
		
		1. Copy and paste your Figma URL to the YourFigmaURL textbox. 
		Optionally if you have set a password for your Figma link, then add this bit to the end of the Figma URL you pasted  ***secret=yourpassword***"
    
    """
    url = f"https://qr-for-figma-links.p.rapidapi.com/qr"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-for-figma-links.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

