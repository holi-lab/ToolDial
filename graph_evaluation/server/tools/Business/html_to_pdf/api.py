import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def html_to_pdf(url: str, viewport: str=None, margin_left: int=None, margin_bottom: int=None, margin_top: int=None, dpi: str=None, output: str='fileurl', grayscale: bool=None, format: str=None, margin: int=None, margin_right: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Submit the URL of the HTML you want to convert. You can also provide a list of space-separated URLs that will be merged into the same PDF-File."
    url: The URL of the HTML you want to convert into a PDF or a list of multiple URLs separated by space (%20).
        viewport: Size of the "browser window" that renders your page (default: 1366x768)
        margin_left: margin in millimeters on left border
        margin_bottom: margin in millimeters on bottom border
        margin_top: margin in millimeters on top border
        dpi: change the dpi of the document (default: 75)
        output: What to return: "data": binary data of pdf or "fileurl": url of generated pdf-file for download
        grayscale: Render the pdf in greyscale (default: false)
        format: Paper format: A4, letter, A3, ... (default: A4)
        margin: margin in millimeters on all 4 borders (default: 0)
        margin_right: margin in millimeters on right border
        
    """
    url = f"https://html-to-pdf.p.rapidapi.com/htmltopdf"
    querystring = {'url': url, }
    if viewport:
        querystring['viewport'] = viewport
    if margin_left:
        querystring['margin-left'] = margin_left
    if margin_bottom:
        querystring['margin-bottom'] = margin_bottom
    if margin_top:
        querystring['margin-top'] = margin_top
    if dpi:
        querystring['dpi'] = dpi
    if output:
        querystring['output'] = output
    if grayscale:
        querystring['grayscale'] = grayscale
    if format:
        querystring['format'] = format
    if margin:
        querystring['margin'] = margin
    if margin_right:
        querystring['margin-right'] = margin_right
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "html-to-pdf.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

