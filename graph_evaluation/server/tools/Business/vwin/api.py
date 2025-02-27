import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def link_v_o_vwin_2022(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vwin là nhà cái cá cược trực tuyến hàng đầu thuộc tập đoàn BVI. Các sản phẩm đều được cấp phép hoạt động bởi Cục Quản lý Giải trí và Trò chơi Philippines.
		#vwinwin #nhà_cái_Vwin #nạp_tiền_Vwin #link_vào_Vwin #hướng_dẫn_chơi_Vwin /m/033_l8  /m/033_l8 #Casino #đăng ký vwin #khuyễn mãi vwin #lô_xiên_là_gì
		Địa chỉ 89 Xuân Phương, Hòe Thị, Từ Liêm, Hà Nội
		Hotline 0973 404 745
		Email:  vwin.win3@gmail.com
		Google map https://www.google.com/maps?cid=14698248580497652518 
		Website https://vwin.win/ 
		https://vwin.win/dang-ky/ 
		https://vwin.win/khuyen-mai/ 
		https://vwin.win/lo-xien/ 
		https://en.wikipedia.org/wiki/Online_casino  
		https://sites.google.com/view/vwinwin/ 
		https://sites.google.com/view/vwinwin/home 
		http://codepad.org/users/vwinwin
		https://descubre.beqbe.com/p/vwinwin
		https://li.sten.to/vwinwin
		https://ko-fi.com/vwinwin
		https://www.blogger.com/profile/13060404880878157843 
		https://experiment.com/users/vvwinwin
		https://comicvine.gamespot.com/profile/vwinwin/ 
		https://folkd.com/user/vwinwin 
		http://vwinwin.wikidot.com/system:welcome
		https://gotartwork.com/Profile/vwin-win/140261/ 
		https://stocktwits.com/vwinwin
		https://ko-fi.com/vwinwin
		https://linktr.ee/vwinwin
		http://wpc.hotlog.ru/profile.php?user_id=423494
		https://gallery.autodesk.com/users/9622XEQGL2LE2QWK?relProf=1 
		https://padlet.com/vwinwin
		https://www.mojomarketplace.com/user/vwinwin-WnVpG2ZrYl
		https://flipboard.com/@vwinwin
		https://dev.to/vwinwinn
		https://www.walkscore.com/people/500187087237/vwinwin 
		https://www.mobypicture.com/user/vwinwin
		https://pinshape.com/users/1985968-vwinwin#designs-tab-open
		https://kyteapp.mn.co/members/10979196
		https://www.gaiaonline.com/profiles/vwinwin/45835299/
		https://www.bhimchat.com/vwinwin
		https://pawoo.net/@vwinwin
		http://pixelhub.me/vwinwin 
		http://uid.me/vwinwin  
		https://www.beatstars.com/vwinwin/about 
		https://gitlab.pasteur.fr/vwinwinn"
    
    """
    url = f"https://vwin.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vwin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

