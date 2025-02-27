import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dothuyhocdientu(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tôi Đỗ Thủy Học Điện Tử một người đam mê cung cấp kiến thức cuộc sống và kiến thức giáo dục văn học , vật lý , điện tử đến cho mọi người hãy cùng tôi tiếp nhận kiến thức mới nhé !  
		#Hocdientu #dothuyhocdientu #hocdientunet
		Website https://hocdientu.net/dothuyhocdientu/admin/ 
		Gmail: dothuy.alphavina@gmail.com
		Liên hệ  0389967362
		https://www.youtube.com/channel/UCWndtZoUs1VesSGCos080CQ/about
		https://www.instagram.com/dothuyhocdientu/
		https://twitter.com/dothuyhocdientu
		https://www.beatstars.com/dothuyhocdientu/about 
		https://gallery.autodesk.com/users/AX52M2PTRTYXKGUQ 
		https://www.pinterest.com/dothuyhocdientu/_saved/
		https://social.msdn.microsoft.com/Profile/dothuyhocdientu
		https://social.technet.microsoft.com/profile/dothuyhocdientu/
		https://www.flickr.com/people/195886372@N02/
		https://www.blogger.com/profile/18193046433910213136
		https://vi.gravatar.com/dothuyhocdientu
		https://www.reddit.com/user/dothuyhocdientu
		https://soundcloud.com/dothuyhocdientu
		https://www.behance.net/dothuyhocdientu
		https://www.twitch.tv/dothuyhocdientu/about
		https://issuu.com/dothuyhocdientu
		https://www.liveinternet.ru/users/dothuyhocdientu/profile"
    
    """
    url = f"https://dothuyhocdientu.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dothuyhocdientu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

