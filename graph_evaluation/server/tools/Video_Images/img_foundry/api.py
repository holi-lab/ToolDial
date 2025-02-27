import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_image(url: str, icon_image_url: str='https://img-foundry.com/static/image/img_foundry_square_logo_512.png', owner_h2: str='img-foundry.com', owner_h1: str='IMG Foundry', text: str='This is website description.', bg_color: str='#AAAAAA', owner_image_url: str='https://img-foundry.com/static/image/img_foundry_square_logo_512.png', title: str='IMG Foundry', width: int=1200, height: int=675, theme: str='light', store: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "generate image from url either retriving image directly or storing image for hotlink."
    icon_image_url: *[Pro Only]* override icon image.
        owner_h2: *[Pro Only]* override owner second lines text.
        owner_h1: *[Pro Only]* override owner first lines text.
        text: *[Pro Only]* override description text.
        bg_color: *[Pro Only]* override background color in format *#RRGGBB*
        owner_image_url: *[Pro Only]* override owner image.
        title: *[Pro Only]* override title text.
        width: override default width of image with the provided one.
        height: override default height of image with the provided one.
        theme: override default theme with *light* or *dark* options, default is *light*
        store: **true** : endpoint will response image id for later use
**false** : endpoint will response as image
Default : true
        
    """
    url = f"https://img-foundry.p.rapidapi.com/core/generate_image_from_url/"
    querystring = {'url': url, }
    if icon_image_url:
        querystring['icon_image_url'] = icon_image_url
    if owner_h2:
        querystring['owner_h2'] = owner_h2
    if owner_h1:
        querystring['owner_h1'] = owner_h1
    if text:
        querystring['text'] = text
    if bg_color:
        querystring['bg_color'] = bg_color
    if owner_image_url:
        querystring['owner_image_url'] = owner_image_url
    if title:
        querystring['title'] = title
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if theme:
        querystring['theme'] = theme
    if store:
        querystring['store'] = store
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "img-foundry.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

