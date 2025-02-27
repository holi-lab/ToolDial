import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qr_qcode_pdf_generator(text_off: int=10, text_size: int=12, page_height: int=50, text_location: str='DOWN', fill_color: str=None, page_width: int=80, border: int=0, back_color: str=None, type: str='qr', data_url: str=None, size: int=120, data: str='Example QR Code;Example Code 2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "data=&data_url=http://localhost/asd.txt&type=CODE128&size=2&border=0&page_width=80&page_height=50&text_location=TOP&text_size=10&text_off=10
		
		&data = 
		&data_url=
		&type=qr/CODE128
		&size=1(barcode)/1-200
		&border=0
		&page_width=80
		&page_height=50
		&text_location='','TOP','DOWN','LEFT','RIGHT' (IF EMPTY NO TEXT)
		&text_size=10
		&text_off=10 (Text-offset)
		![](https://ibb.co/dB6Bqzg)
		![](https://ibb.co/GW5V7N4)
		
		For use URL_data: Use example: &data_url=http://localhost/example.txt
		1 line 1 qr code"
    
    """
    url = f"https://qrgenerator.p.rapidapi.com/barcode"
    querystring = {}
    if text_off:
        querystring['text_off'] = text_off
    if text_size:
        querystring['text_size'] = text_size
    if page_height:
        querystring['page_height'] = page_height
    if text_location:
        querystring['text_location'] = text_location
    if fill_color:
        querystring['fill_color'] = fill_color
    if page_width:
        querystring['page_width'] = page_width
    if border:
        querystring['border'] = border
    if back_color:
        querystring['back_color'] = back_color
    if type:
        querystring['type'] = type
    if data_url:
        querystring['data_url'] = data_url
    if size:
        querystring['size'] = size
    if data:
        querystring['data'] = data
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrgenerator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

