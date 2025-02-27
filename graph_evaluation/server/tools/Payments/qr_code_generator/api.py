import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_code_image_base64(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return base64 image data as follows:
		` {
		        "statusCode": 200,
		        "headers": { "Content-Type": "image/png",
		            "Content-Disposition": "attachment; filename=\"qr-code.png\""
		        },
		        "body": img_str,
		        "isBase64Encoded": True
		    }`"
    
    """
    url = f"https://qr-code-generator40.p.rapidapi.com/qr-code-generator"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-code-generator40.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

