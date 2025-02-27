import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_supported_languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lists all of the supported languages in [ISO_639-2/T](https://en.wikipedia.org/wiki/ISO_639-2) format.
		
		In the case of Chinese and Japanese, there are multiple versions suffixed with _tra (Traditional) and _sim (Simplified).
		For Chinese, Japanese and Korean there are versions suffixed with _vert for Vertical text. In some cases these are combined e.g chi_sim_vert
		
		The return also lists which language is available on which endpoint (currently 1 and 2)."
    
    """
    url = f"https://ocr-supreme.p.rapidapi.com/ocr/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ocr-supreme.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

