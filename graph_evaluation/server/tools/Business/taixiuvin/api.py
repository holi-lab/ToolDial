import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def taixiu(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Taixiu.vin là trang web cập nhật top các trang tài xỉu online uy tín, app tài xỉu đổi thưởng trực tuyến ăn tiền thật đông người chơi nhất hiện nay. Tác giả Quang Cua là sẽ chia sẻ kinh nghiệm chơi game tài xỉu không bao giờ lỗ, giúp anh em kiếm tiền tươi thóc thật xanh chín nhất. #taixiu #taixiuonline #taixiudoithuong #quangcua #taixiuvin #taixiu3d  #/m/033_l8  /m/033_l8 #Casino 
		Địa chỉ : 608 Nguyễn Khoái, Thanh Long, Hai Bà Trưng, Hà Nội, Việt Nam
		Hotline +84944444444
		Gmail: taixiuvin@gmail.com
		Google map https://www.google.com/maps?cid=16315179566595217438 
		Website https://taixiu.vin/ 
		https://www.facebook.com/taixiuvin/about 
		https://sites.google.com/view/taixiuvin/home 
		https://taixiuvin.weebly.com/ 
		https://en.wikipedia.org/wiki/Online_casino 
		https://start.me/u/ydakaG/taixiuvinn
		https://tawk.to/taixiuvin
		https://www.doyoubuzz.com/taixiu-vin#/home
		https://www.reverbnation.com/artist/taixiuvin
		https://note.com/taixiuvin/
		https://www.discogs.com/fr/user/taixiuvin
		https://fliphtml5.com/homepage/wuqmz
		https://www.producthunt.com/@taixiuvin
		https://www.goodreads.com/user/show/151248035-taixiuvin 
		https://support.wedesignthemes.com/users/taixiuvin/
		http://wpc.hotlog.ru/profile.php?user_id=424278
		https://easypropertylistings.com.au/support/users/taixiuvin/
		https://thefeedfeed.com/taixiuvin
		https://www.zillow.com/profile/taixiuvinn
		https://padlet.com/taixiuvin
		https://www.mojomarketplace.com/user/taixiuvin-oDYc3XsyZr
		https://flipboard.com/@taixiuvin
		https://dev.to/taixiuvin
		https://www.walkscore.com/people/307616998804/taixiuvin
		https://pastebin.com/u/taixiuvin
		https://camp-fire.jp/profile/taixiuvin
		https://community.opengroup.org/taixiuvin
		https://www.360cities.net/profile/taixiuvin
		https://www.diggerslist.com/taixiuvin/about
		https://www.credly.com/users/taixiu-vin/badges
		https://marketplace.apartmenttherapy.com/store/612acf21-cdab-4978-b5e6-30b27b3ada19/about
		https://www.diigo.com/profile/taixiuvin
		https://www.scoop.it/u/taixiuvin
		https://qiita.com/taixiuvin
		https://band.us/band/87648134
		https://visual.ly/users/taixiuvin/portfolio
		https://ello.co/taixiuvin
		https://www.lifeofpix.com/photographers/taixiuvin/"
    
    """
    url = f"https://taixiuvin.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taixiuvin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

