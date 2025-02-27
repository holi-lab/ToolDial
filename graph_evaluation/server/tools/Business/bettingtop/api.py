import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bettingtopnet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bettingtop.net chuyên Review chi tiết các nhà cái game bài, slot game, game bài đổi thưởng cá cược trực tuyến mới nhất được cập nhật liên tục mỗi ngày.
		#bettingtopnet  #bettingtop #nhà_cái_uy_tín #nhà_cái_uy_tín_tại_Việt_Nam #nhà_cái_cá_cược_uy_tín  #cổng_game_uy_tín #cổng_game_đổi_thưởng #đổi_thưởng_uy tín  
		Hotline 0903214547
		Địa chỉ : 12A Floor, Siam Tower, 989 Rama I Road, Pathumwan,, Bangkok, 10330
		Gmail: bettingtopnet@gmail.com
		Website https://bettingtop.net/ 
		https://bettingtop.net/nha-cai-uy-tin 
		https://bettingtop.net/cong-game-uy-tin 
		https://bettingtop.net/top-game 
		https://guides.co/p/bettingtopnet
		https://guides.co/p/bettingtopnet
		https://twitter.com/bettingtopnet
		https://www.youtube.com/channel/UCWsjpzVX4-wAAXWKP7gWdYw
		https://www.misterpoll.com/forums/1/topics/330236
		https://www.pinterest.com/bettingtopnet
		https://connect.garmin.com/modern/profile/5f250f17-12cb-4baf-bf2d-b7d4e48543a9 
		https://social.technet.microsoft.com/Profile/bettingtopnet
		https://social.msdn.microsoft.com/profile/bettingtopnet/
		https://gallery.autodesk.com/users/9XG2YGA4PJDFWJYB 
		https://lightroom.adobe.com/u/bettingtopnet
		https://www.blogger.com/profile/13275243059561877400
		https://vi.gravatar.com/bettingtopnet
		https://staffmeup.com/profile/bettingtopnet
		http://www.rohitab.com/discuss/user/482255-bettingtopnet/
		https://discover.events.com/profile/bettingtopnet/3630877/savethedate/
		https://www.behance.net/bettingtopnet
		https://www.openstreetmap.org/user/bettingtopnet
		https://issuu.com/bettingtopnet
		https://profile.hatena.ne.jp/BettingtopN/
		https://ok.ru/bettingtopnet
		https://www.liveinternet.ru/users/bettingtopnet/blog#post492328466
		https://www.twitch.tv/bettingtopnet/about
		https://bettingtopnet.bandcamp.com/album/bettingtop
		https://dribbble.com/bettingtopnet/about
		https://groups.google.com/g/bettingtopnet 
		https://onlyfans.com/bettingtopnet 
		https://gitlab-test.eclipse.org/bettingtopnet 
		https://www.beatstars.com/bettingtopnet/about 
		https://stocktwits.com/bettingtopnet 
		https://ko-fi.com/bettingtopnet"
    
    """
    url = f"https://bettingtop.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bettingtop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

