import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hint_font(is_id: str, type: str, adjustsubglyphs: bool=None, binaryresult: bool=None, hintcomposites: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hint the font and download"
    type: Hinting type(directwire/windows-gdi/grayscale/remove).
        adjustsubglyphs: Should adjust sub glyphs? Default enabled.
        binaryresult: Return converted file in binary format, the default is base64 json response.
        hintcomposites: Should hint composite glyphs? Default enabled.
        
    """
    url = f"https://fonthint.p.rapidapi.com/font/{is_id}/hint"
    querystring = {'type': type, }
    if adjustsubglyphs:
        querystring['adjustsubglyphs'] = adjustsubglyphs
    if binaryresult:
        querystring['binaryresult'] = binaryresult
    if hintcomposites:
        querystring['hintcomposites'] = hintcomposites
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fonthint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_fonts(start: str, public: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the fonts in the system"
    start: Starting index. This required since the result is paged.
        public: Should return public fonts in the result or not?
        
    """
    url = f"https://fonthint.p.rapidapi.com/font/list"
    querystring = {'start': start, }
    if public:
        querystring['public'] = public
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fonthint.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

