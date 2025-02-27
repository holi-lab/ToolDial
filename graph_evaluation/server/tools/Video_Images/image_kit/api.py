import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_transformed_image_from_a_url(url: str, resize: str='auto,512', resizemode: str=None, flip: str=None, blur: int=8, format: str=None, flop: str=None, grayscale: str=None, quality: int=80, rotate: int=45, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the transformed image from a URL after applying format conversions, rotation effects, grayscale, blur, image inversion, quality and size adjustments."
    url: The URL of the image that will be transformed. The image must be smaller than 10 MB and have one of these MIME types: `image/png`, `image/jpeg` or `image/webp`.
        resize: The new size for the resulting image. The format of the parameter value is: `<width>,<height?>` with integer and `auto` allowed for the `width` and `height` attributes. For example:
- `1024,512`: will result in an image 1024px width and 512px height ;
- `auto,512`: will result in an image with width proportional to the original and height 512px ;
- `512,auto`: will result in an image 512px width and 512px height proportional to the original;
- `512`: will be converted to `512,auto` and result in an image with a width of 512px and a height proportional to the original.
By default, if the parameter `resize' is not provided, the size of the original image will be kept.
        resizemode: The desired resizing mode to apply to the resulting image. It can be any of these:

- `cover`: preserves aspect ratio, ensuring that the image covers both dimensions provided, cropping it if necessary;
- `contain`: preserves aspect ratio, ensuring that the image contain within both provided dimensions using margins if necessary;
- `fill`: ignores the aspect ratio of the input and stretch to both provided dimensions;
- `inside`: preserves aspect ratio, resizing the image to be as large as possible while ensuring its dimensions are less than or equal to both those specified;
- `outside`: preserves aspect ratio, resizing the image to be as small as possible while ensuring its dimensions are greater than or equal to both those specified.

By default, if the `resizeMode` parameter is not provided, the `cover` value will be used.
The resize mode will only take effect if used with the `resize` parameter.
        flip: Defines whether the resulting image should be flipped on the vertical Y-axis. It is a boolean value (`false`, `0`, `true` or `1`).
By default, if the `flip` parameter is not provided, the image will not be flipped on the vertical Y-axis.
        blur: The desired blur effect for the resulting image. It is an float value **between 0.3 and 8**.
By default, if the parameter `blur` is not provided, the image will not be blurred.
        format: The desired format for the resulting image. It can be one of these:

- `preserve`: keeps the same MIME type as the original image (`image/png`, `image/jpeg` or `image/webp`);
- `png`: converts the image to PNG (MIME type `image/png`);
- `jpg` or `jpeg`: converts the image to JPEG (MIME type `image/jpeg`);
- `webp`: converts the image to WEBP (MIME type `image/webp`).

By default, if the `format` parameter is not provided, the `preserve` value will be used.
        flop: Defines whether the resulting image should be flipped on the horizontal X-axis. It is a boolean value (`false`, `0`, `true` or `1`).
By default, if the `flop` parameter is not provided, the image will not be flipped on the horizontal X-axis.
        grayscale: Defines whether to apply the grayscale effect to the resulting image. It is a boolean value (`false`, `0`, `true` or `1`).
By default, if the parameter `grayscale` is not provided, the grayscale effect will not be applied to the image.
        quality: The desired value for the resulting image quality. It can be a value **between 1 and 100**.
By default, if the parameter `quality` is not provided, the value 100 will be used.
        rotate: The desired rotation angle for the resulting image. It is an **integer value**.
By default, if the parameter `rotate` is not provided, the image will not be rotated.
        
    """
    url = f"https://image-kit.p.rapidapi.com/imagekit"
    querystring = {'url': url, }
    if resize:
        querystring['resize'] = resize
    if resizemode:
        querystring['resizeMode'] = resizemode
    if flip:
        querystring['flip'] = flip
    if blur:
        querystring['blur'] = blur
    if format:
        querystring['format'] = format
    if flop:
        querystring['flop'] = flop
    if grayscale:
        querystring['grayscale'] = grayscale
    if quality:
        querystring['quality'] = quality
    if rotate:
        querystring['rotate'] = rotate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-kit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

