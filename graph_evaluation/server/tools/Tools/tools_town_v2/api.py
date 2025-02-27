import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shayari_generator(category: str, language: str='Marathi', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate the Shayari in English, Hindi, Marathi, Telugu, Kannada, and other 
		 languages."
    
    """
    url = f"https://tools-town.p.rapidapi.com/shayari-generator/"
    querystring = {'category': category, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tools-town.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compress_image(url: str, height: int=1000, width: int=1000, format: str=' webp', quality: int=80, download: str='yes', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compress the image on fly.
		
		Use the quality, height, width, and format and get the optimized image as you want.
		
		- url - (required) The image URL.
		- quality - (optional) The image quality from 0 to 100. Default 80.
		- format - (optional) auto, jpg, jpeg, webp, avif, or png. Default auto.
		- width - (optional) The image max width. Default auto.
		- height - (optional) The image max  height. Default auto.
		- download - (optional) Download the image or just get the image data as response. Use download=yes to download the optimized image. Default ''"
    
    """
    url = f"https://tools-town.p.rapidapi.com/compress-image"
    querystring = {'url': url, }
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    if format:
        querystring['format'] = format
    if quality:
        querystring['quality'] = quality
    if download:
        querystring['download'] = download
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tools-town.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_images_from_website(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Do you want to extract the images from the websites?
		Then, Use our API which allow you to fetch the images from the 3rd party website."
    
    """
    url = f"https://tools-town.p.rapidapi.com/extract-images-from-website"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tools-town.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

