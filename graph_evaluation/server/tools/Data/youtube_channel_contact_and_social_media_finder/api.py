import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_channel_details(urlorusername: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download such information from the youtube channel as:
		- 🔥 **social media** profiles: facebook, **twitter**, instagram, **onlyfans**, discord, vine, **medium** and much more!
		- 🦔 channel name, location, internal youtube id
		- 📧 **emails** from description
		- 🐯 avatar
		- 🖇️ rss url
		- 😊 subscriber count
		- 🧮 overall channel view count"
    urlorusername: url or youtube username/channel
        
    """
    url = f"https://youtube-channel-contact-and-social-media-finder.p.rapidapi.com/GetChannelDetails"
    querystring = {'UrlOrUsername': urlorusername, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youtube-channel-contact-and-social-media-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

