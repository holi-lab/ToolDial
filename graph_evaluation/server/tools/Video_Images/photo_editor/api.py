import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def https_photokit_com_integrations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "PhotoKit API allows you to have the editor at your own website (using the iframe), and configure it as you wish, for free. Easy to be integrated into a CMS, digital media, site builders or other content projects."
    
    """
    url = f"https://photo-editor.p.rapidapi.com/editor/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "photo-editor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

