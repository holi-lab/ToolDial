import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lucky88(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lucky88 - https://lucky88.life/  - Link đăng ký chính thức của nhà cái lucky88. Lucky88 đã và đang mang đến một sân chơi đầy may mắn dành cho các cược thủ. Tại đây, người chơi sẽ được tận hưởng sân chơi cá cược bóng đá, casino trực tuyến hàng đầu Việt Nam
		#lucky88 #nhacailucky88 #casino  #xoso  #lode #cacuoc /m/033_l8  /m/033_l8 #Casino 
		Hotline (+63)99668852
		Gmail: lucky88dotlife@gmail.com
		Địa chỉ : 11-3 Phượng Bãi, Biên Giang, Hà Đông, Hà Nội, Việt Nam
		Website https://lucky88.life/the-thao/ 
		https://lucky88.life/casino/ 
		https://lucky88.life/tin-tuc/ 
		https://en.wikipedia.org/wiki/Online_casino  
		https://guides.co/p/lucky88life
		https://twitter.com/lucky88_life
		https://www.youtube.com/channel/UCxhXyjob65MbVWj20X9-T6w/about
		https://www.misterpoll.com/forums/1/topics/330959
		https://www.pinterest.com/lucky88life1/_saved/
		https://social.technet.microsoft.com/Profile/lucky88life
		https://social.msdn.microsoft.com/Profile/lucky88life
		https://lucky88life.blogspot.com/2022/06/lucky88life.html
		https://vi.gravatar.com/lucky88life1
		http://www.rohitab.com/discuss/user/541040-lucky88life/
		https://medium.com/@lucky88life
		https://www.behance.net/lucky88life
		https://www.openstreetmap.org/user/lucky88life
		https://issuu.com/lucky88life
		https://profile.hatena.ne.jp/lucky88life/profile
		https://www.beatstars.com/lucky88life/about 
		https://ok.ru/profile/600493288993/statuses/154036503980065
		https://www.liveinternet.ru/users/lucky88life/blog#post492851427
		https://bandcamp.com/lucky88life
		https://dribbble.com/lucky88life/about
		https://sites.google.com/view/lucky88life
		https://myspace.com/lucky88life
		https://www.goodreads.com/user/show/151926884-lucky88-life
		https://linktr.ee/lucky88life
		https://www.buzzfeed.com/lucky88life
		https://hub.docker.com/u/lucky88life"
    
    """
    url = f"https://lucky88.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lucky88.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

