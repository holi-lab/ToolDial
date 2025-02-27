import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def debet_debet88(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Debet - debet88 ⭐️ Cá cược bóng đá uy tín online, cá cược thể thao trực tuyến nhà cái uy tín số 1 GERMANY, link vào debet chính thức, tỷ lệ cá cược tốt nhất.	
		#đăng ký debet #rút tiền debet #nạp tiền debet 
		#tải ứng dụng debet #/m/033_l8  /m/033_l8 #Casino #Ae888vip #Ae888
		Địa chỉ : 143 P. Nguyễn Chính, Tân Mai, Hai Bà Trưng, Hà Nội	
		Hotline 0888.794.300	
		Email: debet88us@gmail.com		
		Website https://debet88.us/ 	
		https://debet88.us/dang-ky-debet/ 
		https://debet88.us/nap-tien-debet/ 
		https://debet88.us/rut-tien-debet/ 
		https://debet88.us/tai-ung-dung-debet/ 	
		https://en.wikipedia.org/wiki/Online_casino   
		https://www.giantbomb.com/profile/debet88us/ 
		https://www.bigpictureclasses.com/users/debet88us
		https://li.sten.to/debet88us
		https://onlyfans.com/debet88us 
		https://www.blogger.com/profile/13591324188810697260 
		https://experiment.com/users/debet88us
		https://comicvine.gamespot.com/profile/debet88us/
		https://gifyu.com/debet88us
		https://folkd.com/user/debet88us
		https://sketchfab.com/debet88us 
		http://www.wikidot.com/user:info/debet88us
		https://bikepgh.org/message-board/users/debet88us
		https://gotartwork.com/Profile/debet88-us/141451/
		https://stocktwits.com/debet88us
		https://ko-fi.com/debet88us
		https://linktr.ee/debet88us
		https://about.me/debet88us
		http://uid.me/debet88_us 
		https://www.beatstars.com/debet88us/about 
		https://www.warriorforum.com/members/debet88us.html
		https://www.mountainproject.com/user/201354523/debet88-us
		https://www.trepup.com/debet-us
		https://sharree.com/User-debet88us
		https://app.bountysource.com/people/101061-debet88us
		https://www.goodreads.com/user/show/150657666-debet88-us
		https://www.metal-archives.com/users/debet88us
		https://gifyu.com/debet88us
		https://os.mbed.com/users/debet88us/
		http://photozou.jp/user/top/3314609
		https://gfycat.com/@debet88us
		https://zindi.africa/users/debet88us
		https://git.project-hobbit.eu/debet88us
		https://artistecard.com/debet88us
		https://notionpress.com/author/508064
		https://www.longisland.com/profile/debet88us"
    
    """
    url = f"https://debet.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "debet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

