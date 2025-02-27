import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def rotate(url: str, degrees: int=90, compress: bool=None, format: str='auto', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    url: The URL of the source image.
        degrees: One of the following values: 90, 180 and 270. The amount of the degrees to rotate clock-wise. If omitted, then no rotation will be applied. If the value is not one the accepted values, then an error will be thrown.
        compress: A boolean representation of whether or not the resulting image should be compressed. Default is "false".
        format: Whether or not the file format of the resulting image should be converted to one that will produce a file with smaller file size. The returned image file format will depend on the support of the client software (e.g. based on the user agent header or a browser-based request). If omitted or set to "false", then the returned image file format will be the same as the source image. Available options are "auto" and "false". Default is "false".
        
    """
    url = f"https://sandcage-sandcage-v1.p.rapidapi.com/rotate"
    querystring = {'url': url, }
    if degrees:
        querystring['degrees'] = degrees
    if compress:
        querystring['compress'] = compress
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandcage-sandcage-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resize(url: str, format: str='auto', compress: bool=None, fit: bool=None, ratio: int=None, height: int=500, width: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    url: The URL of the source image.
        format: Whether or not the file format of the resulting image should be converted to one that will produce a file with smaller file size. The returned image file format will depend on the support of the client software (e.g. based on the user agent header or a browser-based request). If omitted or set to "false", then the returned image file format will be the same as the source image. Available options are "auto" and "false". Default is "false".
        compress: A boolean representation of whether or not the resulting image should be compressed. Default is "false".
        fit: A boolean representation of whether or not the resulting image should fit within the specified width and height. This flag will only take affect if both "width" and "height" have been specified. If the "width" and "height" are both specified without the "fit" parameter being set to "true" and the set values for "width" and "height" do not respect the aspect ratio of the image, then the produced image will be skewed. Default is "false".
        ratio: The percent-wise dimensions of the resulting resized image relative to the dimensions of the source image. If a value has been provided for either "width" or "height", the value provided for this parameter will not be taken into account. If the calculated width or height are less than 1 pixel, then that dimension will be capped to 1 pixel.
        height: The height of the resulting image.
        width: The width of the resulting image.
        
    """
    url = f"https://sandcage-sandcage-v1.p.rapidapi.com/resize"
    querystring = {'url': url, }
    if format:
        querystring['format'] = format
    if compress:
        querystring['compress'] = compress
    if fit:
        querystring['fit'] = fit
    if ratio:
        querystring['ratio'] = ratio
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandcage-sandcage-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cover(url: str='https://cdn.sandcage.com/p/a/img/others/before.jpg', width: int=200, height: int=200, compress: bool=None, format: str='auto', cover: str='x:left,y:middle', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    url: The URL of the source image.
        width: The width of the resulting image. If omitted it will default to the width of the source image.
        height: The height of the resulting image. If omitted it will default to the height of the source image.
        compress: A boolean representation of whether or not the resulting image should be compressed. Default is "false".
        format: Whether or not the file format of the resulting image should be converted to one that will produce a file with smaller file size. The returned image file format will depend on the support of the client software (e.g. based on the user agent header or a browser-based request). If omitted or set to "false", then the returned image file format will be the same as the source image. Available options are "auto" and "false". Default is "false".
        cover: The portion of the image to return. One or two, non-conflicting values for the "x" and "y" space. The "x" space can take one of the following values: left, right or center. The "y" space can take one of the following values: top, bottom or middle Values for both the "x" and "y" space can be provided by comma separating them. Example: y:top,x:right or x:left,y:middle. If either of the values is omitted or is not one of the accepted ones, then the default values will be "middle" for "y" and "center" for "x".
        
    """
    url = f"https://sandcage-sandcage-v1.p.rapidapi.com/cover"
    querystring = {}
    if url:
        querystring['url'] = url
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if compress:
        querystring['compress'] = compress
    if format:
        querystring['format'] = format
    if cover:
        querystring['cover'] = cover
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandcage-sandcage-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crop(url: str, compress: bool=None, format: str='auto', tl: str='x:50,y:150', br: str='x:449,y:349', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    url: The URL of the source image.
        compress: A boolean representation of whether or not the resulting image should be compressed. Default is "false".
        format: Whether or not the file format of the resulting image should be converted to one that will produce a file with smaller file size. The returned image file format will depend on the support of the client software (e.g. based on the user agent header or a browser-based request). If omitted or set to "false", then the returned image file format will be the same as the source image. Available options are "auto" and "false". Default is "false".
        tl: The upper-left corner coordinates (x,y) of the image to return. Accepts one or both of the values for the "x" and "y" coordinates, which must be equal to or larger than "0" and equal or smaller than the respective "x" and "y" coordinate values of "br". Example: x:10,y:20. If either of the values is omitted, then it will default to "0".
        br: The bottom-right corner coordinates (x,y) of the image to return. Accepts one or both of the values for the "x" and "y" coordinates, which must be equal to or smaller than dimensions of the source image, minus 1 pixel (E.g.: If a source image has width 768 pixel and height 512 pixel, the maximum allowed values for the "x" and "y" coordinates will be "767" and "511" respectively) and equal or larger than the respective "x" and "y" coordinate values of "tl". Example: x:350,y:400. If either of the values is omitted, then it will default to maximum dimensions available in the source image.
        
    """
    url = f"https://sandcage-sandcage-v1.p.rapidapi.com/crop"
    querystring = {'url': url, }
    if compress:
        querystring['compress'] = compress
    if format:
        querystring['format'] = format
    if tl:
        querystring['tl'] = tl
    if br:
        querystring['br'] = br
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandcage-sandcage-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

