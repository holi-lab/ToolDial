import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_url(url: str, endpage: int=None, format: str=None, startpage: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts pdf files to html, docx, xlsx, json and txt files"
    url: URL for pdf file to convert. Make sure the file has a content type of "application/pdf". Max file size is 10 MB.
        endpage: Ending page of conversion. Use 1 for first page. Defaults to last page.
        format: Output format. Valid values are html, html2, docx, xlsx, txt and json. Defaults to html.
        startpage: Starting page of conversion. Use 1 for first page. Defaults to first page.
        
    """
    url = f"https://atticsoftware-pdf-converter-v1.p.rapidapi.com/convert"
    querystring = {'url': url, }
    if endpage:
        querystring['endPage'] = endpage
    if format:
        querystring['format'] = format
    if startpage:
        querystring['startpage'] = startpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "atticsoftware-pdf-converter-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

