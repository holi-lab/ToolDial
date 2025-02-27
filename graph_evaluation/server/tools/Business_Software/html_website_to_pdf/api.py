import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def instant_webpage_pdf_api(url: str, height: int=600, width: int=800, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Easily convert webpages to high-quality PDFs. By sending a GET request to the API endpoint with the URL of the webpage."
    
    """
    url = f"https://html-website-to-pdf.p.rapidapi.com/pdf"
    querystring = {'url': url, }
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "html-website-to-pdf.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

