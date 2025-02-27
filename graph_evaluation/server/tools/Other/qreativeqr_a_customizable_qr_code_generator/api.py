import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_simple_qr_code(qr_data: str, spacing: int=2, download_file: int=0, qr_size: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The simple QR code generator endpoint allows users to easily create a QR code by passing in the data that they want to convert. With this endpoint, you can customize the size and border spacing of the generated QR code. Additionally, users can choose whether to force download the QR code file or not.
		
		Parameters:
		
		QR_data: (required) The data to be converted to QR code. This can be any text or URL that you want to encode into a QR code.
		
		QR_size: (optional) The size of the QR code. By default, the API will generate a QR code with a size of 10 (250 x 250). However, you can modify this parameter to set the size of your choice.
		
		spacing: (optional) The border spacing of the QR code. By default, the API will generate a QR code with a border spacing of 4 pixels. However, you can modify this parameter to adjust the border spacing according to your needs.
		
		download_file: (optional) A boolean parameter that determines whether the user wants to force download the QR code file or not. If set to 1, the API will return the QR code file as an attachment, prompting the user to download it. If set to 0, the QR code file will be returned as a response."
    qr_data: The data to be converted to QR code. This can be any text or URL that you want to encode into a QR code.
        spacing: The border spacing of the QR code. By default, the API will generate a QR code with a border spacing of 4 pixels. However, you can modify this parameter to adjust the border spacing according to your needs.
        download_file: A boolean parameter that determines whether the user wants to force download the QR code file or not. If set to true, the API will return the QR code file as an attachment, prompting the user to download it. If set to false, the QR code file will be returned as a response.
        qr_size: The size of the QR code. By default, the API will generate a QR code with a size of 250 x 250 pixels. However, you can modify this parameter to set the size of your choice.
        
    """
    url = f"https://qreativeqr-a-customizable-qr-code-generator.p.rapidapi.com/generate_simple_qr/"
    querystring = {'QR_data': qr_data, }
    if spacing:
        querystring['spacing'] = spacing
    if download_file:
        querystring['download_file'] = download_file
    if qr_size:
        querystring['QR_size'] = qr_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qreativeqr-a-customizable-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_colored_qr_code(qr_data: str, bgcolor: str='red', spacing: int=2, qrcolor: str='blue', download_file: int=0, qr_size: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint generates a colored QR code for the given data. The endpoint takes in the following parameters:
		
		QR_data: the data to be encoded into the QR code.
		qrcolor (optional): the color of the QR code. The default color is black.
		bgcolor (optional): the background color of the QR code. The default color is white.
		QR_size (optional): the size of the QR code.
		spacing (optional): the border space of the QR code.
		download_file (optional): determines whether the user wants to force download the QR code.
		
		Using this endpoint, developers can generate a colored QR code for their data with customizations such as color, size, and border spacing. The ability to modify the color and background color of the QR code provides developers with the flexibility to match the QR code to their branding."
    qr_data: The data to be converted to QR code. This can be any text or URL that you want to encode into a QR code.
        bgcolor: the background color of the QR code. The default color is white.
        spacing: The border spacing of the QR code. By default, the API will generate a QR code with a border spacing of 4 pixels. However, you can modify this parameter to adjust the border spacing according to your needs.
        qrcolor: the color of the QR code. The default color is black.
        download_file: A boolean parameter that determines whether the user wants to force download the QR code file or not. If set to true, the API will return the QR code file as an attachment, prompting the user to download it. If set to false, the QR code file will be returned as a response.
        qr_size: The size of the QR code. By default, the API will generate a QR code with a size of 250 x 250 pixels. However, you can modify this parameter to set the size of your choice.
        
    """
    url = f"https://qreativeqr-a-customizable-qr-code-generator.p.rapidapi.com/generate_colored_qr/"
    querystring = {'QR_data': qr_data, }
    if bgcolor:
        querystring['bgcolor'] = bgcolor
    if spacing:
        querystring['spacing'] = spacing
    if qrcolor:
        querystring['qrcolor'] = qrcolor
    if download_file:
        querystring['download_file'] = download_file
    if qr_size:
        querystring['QR_size'] = qr_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qreativeqr-a-customizable-qr-code-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

