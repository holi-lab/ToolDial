import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_status_of_a_conversion(is_id: int, key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the status of an existing conversion and the URLs of the converted pictures and/or zip file."
    id: Conversion identifier
        key: Conversion passphrase
        
    """
    url = f"https://pdf2jpg-pdf2jpg.p.rapidapi.com/convert_pdf_to_jpg.php"
    querystring = {'id': is_id, 'key': key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pdf2jpg-pdf2jpg.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

