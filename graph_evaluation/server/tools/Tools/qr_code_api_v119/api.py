import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_api_web_url(url: str, rounded: bool=None, logo: str='https://www.freepnglogos.com/uploads/google-logo-png/google-logo-png-suite-everything-you-need-know-about-google-newest-0.png', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Call the API Endpoint /qr and pass in the url parameter along with your rapidApi key to obtain a QR Code."
    url: 170.64.137.43:8000/qr?url=https://www.example.com
        logo: [Optional] URL to logo which will be placed at the center of the QR Code.
Logo URL must be in .png
        
    """
    url = f"https://qr-code-api120.p.rapidapi.com/qr"
    querystring = {'url': url, }
    if rounded:
        querystring['rounded'] = rounded
    if logo:
        querystring['logo'] = logo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-api120.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

