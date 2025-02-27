import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def uw88us(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Uw88 - Nhà cái uy tín số 1 Việt Nam, link vào uw88 - ucw88 mới nhất 2022 - cùng nhiều thể loại game: thể thao, đá gà, casino online, trực tiếp bóng đá
		#đăng ký Uw88 #nạp tiền Uw88 #rút tiền Uw88 #tải ứng dụng Uw88
		#/m/033_l8  /m/033_l8 #Casino #uw88us #link vào nhà cái uw88
		Địa chỉ : 159 Nguyễn Viết Xuân, Hà Cầu, Hà Đông, Hà Nội, Việt Nam
		Gmail: uw88us@gmail.com
		Hotline 0876.384.666
		Website https://uw88.us/
		https://uw88.us/dang-ky-uw88/ 
		https://uw88.us/nap-tien-uw88/ 
		https://uw88.us/rut-tien-uw88/ 
		https://uw88.us/tai-ung-dung-uw88/ 
		https://sketchfab.com/uw88us 
		https://gitlab.pasteur.fr/uw88us 
		https://bikepgh.org/message-board/users/uw88us/ 
		https://en.wikipedia.org/wiki/Online_casino 
		https://gitlab.manjaro.org/uw88us
		https://stocktwits.com/uw88us
		https://www.blogger.com/profile/12234232264250242551 
		https://experiment.com/users/uw88us
		https://sites.google.com/view/uw88us/link-vao-nha-cai-uw88 
		https://sites.google.com/view/uw88us 
		https://uw88us.blogspot.com/ 
		https://www.beatstars.com/uw88us/about 
		https://li.sten.to/uw88us
		https://uw88us.blogspot.com/2022/05/link-nha-cai-uw88.html 
		https://www.jigsawplanet.com/uw88us?viewas=10ed48d232dd
		http://www.wikidot.com/user:info/uw88us 
		https://www.giantbomb.com/profile/uw88us/ 
		https://ko-fi.com/uw88us 
		https://bikepgh.org/message-board/users/uw88us/ 
		https://www.divephotoguide.com/user/uw88us
		https://fr.ulule.com/uw88us/#/projects/followed
		https://my.archdaily.com/us/@uw88-us
		https://folkd.com/user/uw88us
		http://codepad.org/users/uw88us
		https://git.qoto.org/uw88us 
		https://www.twitch.tv/uw88us/about
		https://uw88us.blogspot.com/2022/05/uw88us.html
		https://community.tubebuddy.com/index.php?members/102520/#about
		https://gotartwork.com/Profile/uw88-us/143235/
		https://discover.events.com/profile/uw88us/3630233/savethedate/
		https://www.mixcloud.com/uw88us/
		https://lab.quickbox.io/uw88us 
		https://wakelet.com/@uw88us
		https://www.slideserve.com/uw88us
		https://www.blackhatworld.com/members/uw88us.1607180/#about
		https://twitter.com/uw88us
		https://www.youtube.com/channel/UC8pB5XVL6jrt0QSj4P9bkNg/about
		https://social.technet.microsoft.com/Profile/uw88us
		https://www.bigbasstabs.com/profile/37050.html
		https://social.msdn.microsoft.com/Profile/uw88us
		https://www.provenexpert.com/uw88us/
		https://soundcloud.com/uw88us
		https://issuu.com/uw88us
		https://profile.hatena.ne.jp/uw88us/profile
		https://www.liveinternet.ru/users/uw88us/profile
		https://www.openstreetmap.org/user/uw88us
		https://ok.ru/profile/581969595501
		https://linktr.ee/uw88us"
    
    """
    url = f"https://uw88us.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uw88us.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

