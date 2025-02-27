import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fun88(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fun88 là nhà cái hàng đầu Châu Á, cung cấp đa dạng thể loại cược, khuyến mãi cực chất cho người chơi mới, đăng ký tại Fun88link nhận KM khủng! 
		
		#Fun88 #NhacaiFun88#Cacuocfun88 #Cacuocthethaofun88 #Cacuocbongdafun88 #Cacuocthethaoaofun88 #linkvaonhacaifun88 #đăng_ký #nạp_rút_tiền #khuyến_mại_fun88 #/m/033_l8  /m/033_l8 #Casino  # Fun88link 
		Hotline 0973.727.479
		Email: fun88linkasia@gmail.com
		Địa chỉ : 5 Ng. 670 Đ. Nguyễn Khoái, Thanh Trì, Hai Bà Trưng, Hà Nội, Việt Nam
		Website https://fun88link.asia/ 
		https://fun88link.asia/dang-ky-fun88/ 
		https://fun88link.asia/link-vao-fun88/ 
		https://fun88link.asia/khuyen-mai-fun88/ 
		https://en.wikipedia.org/wiki/Online_casino
		https://sites.google.com/view/fun88linkasia/nha-cai-fun88 
		https://fun88linkasia.blogspot.com/2022/05/fun88-website-chinh-thuc-fun88.html 
		https://fun88linkasia.blogspot.com 
		https://www.facebook.com/fun88link    
		https://www.pinterest.com/fun88linkasia 
		https://experiment.com/users/fun88link
		https://comicvine.gamespot.com/profile/fun88linkasia/
		https://gifyu.com/fun88linkasia
		https://folkd.com/user/fun88linkasia
		https://sketchfab.com/fun88linkasia
		https://bikepgh.org/message-board/users/fun88link/
		https://pinshape.com/users/2345528-fun88link#designs-tab-open 
		https://stocktwits.com/Fun88linkasia
		https://onlyfans.com/fun88link 
		https://connect.garmin.com/modern/profile/d7abdabc-9646-492c-8119-0b54a4687e36
		https://ko-fi.com/fun88linkasia
		https://linktr.ee/fun88linkasia
		https://about.me/fun88_link
		https://gallery.autodesk.com/users/4ZQ2HP8LN9UNF8F4
		http://uid.me/fun88_link
		https://gitlab.manjaro.org/Fun88linkasia
		https://gitlab.pasteur.fr/Fun88linkasia
		https://repo.getmonero.org/Fun88linkasia
		https://gitlab.haskell.org/Fun88linkasia
		https://www.pinterest.com/funlinkasia/_saved/
		https://www.behance.net/fun88link2
		https://vi.gravatar.com/fun88linkasia
		https://www.jigsawplanet.com/Fun88linkasia?viewas=386e88739bee
		https://www.divephotoguide.com/user/fun88linkasia
		https://fr.ulule.com/fun88linkasia/#/
		https://my.archdaily.com/us/@fun88-link
		http://codepad.org/users/fun88linkasia
		https://www.twitch.tv/fun88linkasia"
    
    """
    url = f"https://nha-cai-fun88.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nha-cai-fun88.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

