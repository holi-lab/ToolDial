import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_qr_code_with_external_logo_url(text: str, size: int=400, eye_style: str=None, block_style: str=None, bg_color: str='FFFFFF', validate: int=1, gradient_color_end: str='00FF00', gradient: int=1, gradient_color_start: str='FF0000', gradient_type: str=None, logo_size: int=0, fg_color: str='FF0000', format: str=None, logo_url: str='https://cdn.auth0.com/blog/symfony-blog/logo.png', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate QR code with logo url specified in GET param. Very convenient for quick testing.
		WARNING: RapidAPI preview is not working properly for binary API output! Use "Code snippets" to properly test the API."
    size: QR code size
        validate: Validate QR code after generation to avoid broken codes due to too large logo or low contrast.
        gradient: Set to 0 to disable gradient.
        fg_color: QR code foreground color in HEX. This is ignored if gradient mode is active.
        format: QR code output format.
        logo_url: URL to svg, png or jpg image, which will be used as logo in the QR code center.
        
    """
    url = f"https://qrcode-supercharged.p.rapidapi.com/"
    querystring = {'text': text, }
    if size:
        querystring['size'] = size
    if eye_style:
        querystring['eye_style'] = eye_style
    if block_style:
        querystring['block_style'] = block_style
    if bg_color:
        querystring['bg_color'] = bg_color
    if validate:
        querystring['validate'] = validate
    if gradient_color_end:
        querystring['gradient_color_end'] = gradient_color_end
    if gradient:
        querystring['gradient'] = gradient
    if gradient_color_start:
        querystring['gradient_color_start'] = gradient_color_start
    if gradient_type:
        querystring['gradient_type'] = gradient_type
    if logo_size:
        querystring['logo_size'] = logo_size
    if fg_color:
        querystring['fg_color'] = fg_color
    if format:
        querystring['format'] = format
    if logo_url:
        querystring['logo_url'] = logo_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcode-supercharged.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

