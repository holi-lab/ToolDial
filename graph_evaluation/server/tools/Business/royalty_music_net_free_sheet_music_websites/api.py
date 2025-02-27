import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def royaltymusicnet(royaltymusicnet: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Royalty Music Net by Yong Min La - Address: 349 Đ. Âu Cơ, Nhật Tân, Tây Hồ, Hà Nội 100000. Mobile: 098 173 98 45. Email: royaltymusicnet@gmail.com Free Sheet Music Websites. Download and Print World Free Sheet Music (PDF, MIDI, MP3). Free Sheet Music Websites For All Instruments. Piano, Classical Guitar, Violin, Banjo - Royalty Music Net
		
		Royalty Music Net
		Free Sheet Music Websites (PDF, MIDI, MP3)
		Download and Print World Free Sheet Music (PDF, MIDI, MP3)
		Free Sheet Music Websites For All Instruments
		Find and Print Free Sheet Music
		Top 10 Free Sheet Music Websites
		Best Sheet Music Sites
		FREE Sheet Music PDF for Piano
		10 Websites for Free Sheet Music (2022)
		World Free Sheet Music (PDF, MIDI, MP3)
		5 Best Websites to get Free Piano Sheets
		
		[https://royaltymusic.net](url)
		[https://royaltymusic.net/music-type/piano ](url)
		[https://royaltymusic.net/music-type/guitar ](url)
		[https://royaltymusic.net/music-type/violin ](url)
		[https://royaltymusic.net/music-type/sad ](url)
		[https://royaltymusic.net/music-type/romantic](url)"
    
    """
    url = f"https://royalty-music-net-free-sheet-music-websites.p.rapidapi.com/royaltymusic.net"
    querystring = {}
    if royaltymusicnet:
        querystring['royaltymusicnet'] = royaltymusicnet
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "royalty-music-net-free-sheet-music-websites.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

