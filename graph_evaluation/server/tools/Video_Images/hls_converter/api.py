import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_your_hls_stream(uploadid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To get your stream, you can download your HLS stream files in a zipped folder.
		
		Before downloading your file, you may check the status of your encoding, to make sure the encoding is done."
    
    """
    url = f"https://hls-converter.p.rapidapi.com/download/{uploadid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hls-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_encoding_status(uploadid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "After upload the encoding process will start automatically.
		You can check its status on this route, using the `uploadId`.
		
		Five different return values statuses are possible:
		
		- `NOT_UPLOADED`: Your video has not yet been uploaded.
		- `PENDING`: Other videos are currently being processed by our system. Your video will be processed soon.
		- `ENCODING`: Your video is currently being encoded.
		- `DONE`: The encoding is done, you can download your HLS stream.
		- `ERROR`: An unexpected error happened, you should try the process again. If the problem persist, you may contact the administrator."
    
    """
    url = f"https://hls-converter.p.rapidapi.com/status/{uploadid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hls-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

