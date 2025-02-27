import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def contrast_correct_get(url: str, gamma: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "![](https://i.postimg.cc/zfcrjMz8/Contrast-Correct-House.jpg)
		
		For images with unruly dark or light patches, the Contrast Correct endpoint can help alleviate these contrast related issues.
		
		Note: 4mb image size limit. Supports JPG/PNG images.
		
		---------------------------------
		Parameters
		---------------------------------
		**gamma** :  controls the strength of the contrast correction. 
		&nbsp;&nbsp;&nbsp;&nbsp; range     = (0,10)
		&nbsp;&nbsp;&nbsp;&nbsp; default  = 1.1"
    
    """
    url = f"https://aisthetic.p.rapidapi.com/gamma-correct/"
    querystring = {'url': url, 'gamma': gamma, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aisthetic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dslr_enhance_get(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "![](https://i.postimg.cc/Mpz66HzL/DSLR-Sample-Image.jpg)
		
		Upgrade any image to look as if it came from a DSLR camera. Simply send this endpoint your image, and it will use AI to determine optimal adjustments to make in order for the input to resemble an image from a DSLR camera. 
		
		Note: 4mb image size limit. 
		
		---------------------------------
		Parameters
		---------------------------------
		None"
    
    """
    url = f"https://aisthetic.p.rapidapi.com/dslr-enhance/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aisthetic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def denoise_get(url: str, weight: str='7', n_iter: str='4', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "![](https://i.postimg.cc/FzR9ynBQ/Denoise-before-after.jpg)
		For images with various forms of noise (salt-pepper, gaussian, compression etc.) , the Denoise endpoint allows you to control what level of denoising you perform with the weight and n_iter parameters. 
		
		Note: 4mb image size limit. Supports JPG/PNG images.
		
		---------------------------------
		Parameters
		---------------------------------
		**weight**:  controls the tradeoff between denoising and faithfulness to the original image.
		&nbsp;&nbsp;&nbsp;&nbsp; range     = (0,10)
		&nbsp;&nbsp;&nbsp;&nbsp;  default   = 4
		
		**n_iter**: the number of denoising iterations to perform. 
		&nbsp;&nbsp;&nbsp;&nbsp; range     = (0,20)
		&nbsp;&nbsp;&nbsp;&nbsp; default   = 7"
    
    """
    url = f"https://aisthetic.p.rapidapi.com/denoise/"
    querystring = {'url': url, }
    if weight:
        querystring['weight'] = weight
    if n_iter:
        querystring['n_iter'] = n_iter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aisthetic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_quality_analysis_get(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "![](https://i.postimg.cc/zvzzFnkT/IQA-example.png)
		
		Get a single quality score for your image, based on sharpness, noise, compression and many other factors. 100 corresponds to the maximum quality score (a great image), while 0 is the minimum quality score possible (a not so great image). 
		
		Note: 4mb image size limit. Supports JPG/PNG images.
		
		---------------------------------------------
		Parameters
		---------------------------------------------
		
		None"
    
    """
    url = f"https://aisthetic.p.rapidapi.com/iqa/"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aisthetic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reduce_file_size_get(url: str, save_as: str, lossless: str='False', quality: str='75', progressive: str='False', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "![](https://i.postimg.cc/9076T06y/Capture.png)
		
		Optimize you image in JPG,PNG or WEBP format for reduced file size while maintaining image quality.
		
		Note: 4mb image size limit. Supports JPG/PNG images.
		
		------------------------------------------------
		Parameters
		------------------------------------------------
		###save_as = JPG
		**quality**: The image quality, on a scale from 0 (worst) to 95 (best). The default is 75. Values above 95 should be avoided; 100 disables portions of the JPEG compression algorithm, and results in large files with hardly any gain in image quality.
		&nbsp;&nbsp;&nbsp;&nbsp; range = (0,100)
		&nbsp;&nbsp;&nbsp;&nbsp; default =   75
		**progressive**: If present and true, indicates that this image should be stored as a progressive JPEG file.
		&nbsp;&nbsp;&nbsp;&nbsp; range = (True, False)
		&nbsp;&nbsp;&nbsp;&nbsp;default = False
		
		###save_as = PNG
		**n/a**: no parameters to specify for this save type.
		
		###save_as = WEBPG
		**lossless**: If present and true, instructs the WebP writer to use lossless compression.
		&nbsp;&nbsp;&nbsp;&nbsp; range = (True,False) 
		&nbsp;&nbsp;&nbsp;&nbsp; default = True
		**quality**: Integer, 1-100, Defaults to 80. For lossy, 0 gives the smallest size and 100 the largest. For lossless, this parameter is the amount of effort put into the compression: 0 is the fastest, but gives larger files compared to the slowest, but best, 100.
		&nbsp;&nbsp;&nbsp;&nbsp; range = (0,100) 
		&nbsp;&nbsp;&nbsp;&nbsp; default = 75"
    
    """
    url = f"https://aisthetic.p.rapidapi.com/optimize-image/"
    querystring = {'url': url, 'save_as': save_as, }
    if lossless:
        querystring['lossless'] = lossless
    if quality:
        querystring['quality'] = quality
    if progressive:
        querystring['progressive'] = progressive
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aisthetic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upsample_get(url: str, scale: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "![](https://i.postimg.cc/T3gPGYhL/upsample-before-after.png)
		Leverage AI to infer detail in your images and achieve 2,4,8x magnification.
		
		Note: 4mb image size limit. Supports JPG/PNG images.
		
		------------------------------------------------
		Parameters
		------------------------------------------------
		**scale**: The factor you wish to upsample by. 
		&nbsp;&nbsp;&nbsp;&nbsp; range = (2,4,8) 
		&nbsp;&nbsp;&nbsp;&nbsp; default = 2"
    
    """
    url = f"https://aisthetic.p.rapidapi.com/upsample/"
    querystring = {'url': url, }
    if scale:
        querystring['scale'] = scale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aisthetic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

