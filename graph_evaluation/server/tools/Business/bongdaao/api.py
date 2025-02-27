import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bongdaao(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bongdaao.online |Bóng đá ảo Online là trò chơi cá độ đổi thưởng thú vị và được ưa chuộng hiện nay. Các trận đấu bóng đá ảo được lập trình sẵn, diễn ra ngẫu nhiên giữa các đội bóng
		#bongdaao #bongdaaoonline #bongdaaouytin #cadobongdaao #bongdaao.online
		Địa chỉ 65 Đại Đồng, Thanh Trì, Hai Bà Trưng, Hà Nội
		Email: bongdaaoonline@gmail.com
		Website https://bongdaao.online/ 
		https://www.facebook.com/bongdaaoonline 
		https://sites.google.com/view/bongdaao 
		https://stocktwits.com/bongdaao 
		https://bongdaaoonline.blogspot.com/
		https://folkd.com/user/bongdaao 
		https://li.sten.to/bongdaao 
		https://www.blogger.com/profile/02620411345218405663 
		https://www.youtube.com/channel/UCvHkgQM0wA7q_THbGYMB6KA/about  
		https://vi.gravatar.com/bongdaaoonline 
		https://about.me/bongdaao 
		https://www.giantbomb.com/profile/bongdaaoonline/ 
		https://www.goodreads.com/user/show/150894110-bongdaao 
		https://linktr.ee/bongdaao 
		https://www.beatstars.com/bongdaao/about 
		https://connect.garmin.com/modern/profile/2165e6a5-3a78-41ff-9b30-c61717aac9a4
		https://gitlab-test.eclipse.org/bongdaao  
		https://sketchfab.com/bongdaao 
		https://onlyfans.com/bongdaao 
		https://www.thingiverse.com/bongdaao/designs 
		https://bongdaao.threadless.com/about
		https://telegra.ph/bongdaao-05-04
		https://jsfiddle.net/bongdaao/tf1q7uoj/
		https://career.habr.com/bongdaaoonline
		https://my.archdaily.com/us/@bongdaaoonline
		https://www.cplusplus.com/user/bongdaaoonline/ 
		https://repo.getmonero.org/bongdaao 
		 https://leetcode.com/bongdaao/ 
		https://camp-fire.jp/profile/bongdaao
		https://www.credly.com/users/bongda-ao/badges
		https://comicvine.gamespot.com/profile/bongdaao/
		https://qiita.com/bongdaao"
    
    """
    url = f"https://bongdaao.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bongdaao.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

