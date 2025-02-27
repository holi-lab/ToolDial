import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_top_50_videos_of_douyin_account_home_page(homepage_url: str=' https://v.douyin.com/UudJDR4/', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top 50 videos of Douyin account home page"
    
    """
    url = f"https://douyin_video_remove-watermark.p.rapidapi.com/home/Get_homepage_Video"
    querystring = {}
    if homepage_url:
        querystring['homepage_url'] = homepage_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "douyin_video_remove-watermark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_douyin_watermark_free_short_video(videourl: str='https://v.douyin.com/UdQm48M/', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain short video addresses, images, titles, and other information without watermarks"
    
    """
    url = f"https://douyin_video_remove-watermark.p.rapidapi.com/home/Get_No_Watermark_Video"
    querystring = {}
    if videourl:
        querystring['VideoUrl'] = videourl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "douyin_video_remove-watermark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_sales_volume_data_of_douyin_products(product_id: str='3616147324520245955', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the sales volume data of douyin products"
    
    """
    url = f"https://douyin_video_remove-watermark.p.rapidapi.com/home/GetProductInfo"
    querystring = {}
    if product_id:
        querystring['product_id'] = product_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "douyin_video_remove-watermark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

