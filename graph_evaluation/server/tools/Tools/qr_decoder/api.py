import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qrdecoder(src: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the QRCode message from an image or PDF file with embeded QR codes.
		
		Files must be publicly accessible via the web and the src input parm expects a direct link to the file.  Html pages with embeded images or pdfs are not valid and will return empty."
    
    """
    url = f"https://qr-decoder.p.rapidapi.com/getQRCodesFromPDForImage"
    querystring = {'src': src, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

