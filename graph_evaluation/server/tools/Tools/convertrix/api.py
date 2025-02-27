import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_supported_conversion(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return a JSON Dictionary of all the supported conversions.
		
		In the example below `docx` is what you can convert FROM and all the children are what it can be converted to
		
		>     "docx": [
		      "pdf",
		      "html",
		      "jpeg",
		      "rtf"
		    ],"
    
    """
    url = f"https://convertrix.p.rapidapi.com/Api/FileManagement/GetAllConversionTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convertrix.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

