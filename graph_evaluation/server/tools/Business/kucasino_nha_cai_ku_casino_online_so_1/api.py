import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def kucasino_nha_cai_ku_casino_online_so_1(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "KU Casino hay Kucasino là nhà cái cá cược trực tuyến uy tín. KU casino Pro luôn nằm trong top 5 nhà cái cá cược chất lượng năm 2021. Đăng nhập, đăng ký, tải app tại trang chủ https://kucasino.pro để tham gia thế giới casino online đẳng cấp thế giới, nhận nhiều ưu đãi hấp dẫn và tiền cược trải nghiệm miễn phí lên đến 128k
		/m/033_l8  /m/033_l8 #Casino #KU Casino #KUCasinoPro #TảiKucasino
		#TinTứcKu casino #đăngkýKucasino
		Địa chỉ 22 Nguyễn Huệ, Bến Nghé, Quận 1, Thành phố Hồ Chí Minh
		Hotline 84569912969
		Gmail: kucasino1.pro@gmail.com
		Google map https://www.google.com/maps?cid=9183105835723030276 
		Website https://kucasino.pro/  
		https://kucasino.pro/tai-ku-casino-app/ 
		https://kucasino.pro/tin-tuc/ 
		https://kucasino.pro/huong-dan-dang-ky-ku-casino-chi-tiet-nhat/ 
		https://www.facebook.com/kucasinopro1   
		https://www.youtube.com/channel/UCjyi5KYQxHrCN_6CopP_VCw/about
		https://www.pinterest.com/kucasino_pro
		https://kucasinopro3.simplesite.com
		https://www.beatstars.com/kucasinopro3/about 
		https://onlyfans.com/kucasino_pro
		https://padlet.com/Kucasinopro/3974qf57nwix43e9
		https://www.mojomarketplace.com/user/kucasinopro3-hFmfXI6Rmz
		https://flipboard.com/@Kucasino_pro
		https://www.thingiverse.com/kucasinopro3/designs
		https://jsfiddle.net/kucasinopro3/my8ucp7z/
		https://www.credly.com/users/kucasinopro3/badges
		https://camp-fire.jp/profile/kucasinopro3
		https://pastebin.com/u/Kucasino_pro
		https://my.archdaily.com/us/@kucasino-pro
		https://kucasinopro.threadless.com/about
		https://www.diigo.com/profile/kucasino_pro
		https://www.scoop.it/topic/kucasino-pro
		https://qiita.com/Kucasino_pro
		https://band.us/band/87990429
		https://wakelet.com/@Kucasino_pro
		https://comicvine.gamespot.com/profile/kucasino_pro/
		https://telegra.ph/Kucasino-pro-06-16
		https://visual.ly/users/kucasino_pro/portfolio
		https://fr.ulule.com/kucasinopro
		https://www.podomatic.com/podcasts/kucasinopro3
		https://community.opengroup.org/kucasinopro
		https://forum.cs-cart.com/user/218689-kucasino-pro/"
    
    """
    url = f"https://kucasino-nha-cai-ku-casino-online-so-1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kucasino-nha-cai-ku-casino-online-so-1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

