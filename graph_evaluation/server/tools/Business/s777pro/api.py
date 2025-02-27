import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def s777pro(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S777 - s777pro Game Bài Đổi Thưởng Uy Tín tặng ngay 30k trải nghiệm và 100% ngay lần nạp đầu tiên. Nhà Cái S777 Game Bài với nhiều thể loại game như xổ số, bắn cá,.. Dưới đây chúng tôi sẽ bật mí cho bạn các thông tin thú vị về sân chơi game bài đổi thưởng cực
		#s777 #s777pro #s777app #gamebais777 #conggames777 #s777fun #s777net #s777vn #Tải_app_S777 #Đăng_ký_s777 #Nạp_tiền_s777
		Địa chỉ : 10 Lê Văn Sỹ, Phường 11, Quận 3, Thành phố Hồ Chí Minh, Việt Nam
		Hotline 0528276899 – 0563791518
		Gmail: info.sodo66@gmail.com
		GG map : https://www.google.com/maps?cid=13451685837445220128 
		Website https://s777pro.com 
		https://s777pro.com/tai-s777-app/ 
		https://s777pro.com/dang-ky-s777/ 
		https://s777pro.com/nap-tien-s777/ 
		https://www.facebook.com/S777pro 
		https://twitter.com/cong_game_s777 
		https://www.youtube.com/channel/UCC9bmgQGlvd9Nfs6_YMG2ZA
		https://www.hulkshare.com/s777pro
		https://confengine.com/user/s777pro
		https://www.magcloud.com/user/s777pro
		https://forum.cs-cart.com/user/223536-s777pro/
		https://www.plurk.com/s777pro
		https://www.myminifactory.com/users/s777pro
		https://www.surfaceforums.net/members/s777pro.42609/
		https://forum.acronis.com/user/408202
		https://www.instapaper.com/p/10879464
		https://pubhtml5.com/homepage/xdlae
		https://gitlab.tails.boum.org/s777pro
		https://www.dualmonitorbackgrounds.com/s777pro
		https://sketchfab.com/s777pro
		https://muckrack.com/s777-pro
		https://roundme.com/@s777pro/about
		https://www.joindota.com/users/2043042-s777pro
		https://kienthucykhoa.edu.vn/members/s777pro.24313/
		https://community.aodyo.com/user/s777pro
		https://mmo4me.com/members/s777pro.197347/
		https://myanimelist.net/profile/s777pro
		https://communities.bentley.com/members/4b2b8dae_2d00_8904_2d00_41c6_2d00_a083_2d00_4c00969d3bd3
		https://pxhere.com/en/photographer/3855852
		https://www.cplusplus.com/user/s777pro
		https://www.picfair.com/users/s777pro
		https://replit.com/@s777pro
		https://leetcode.com/s777pro/
		https://www.intensedebate.com/profiles/s777procom
		http://cloudsdeal.xobor.de/u7240_s-pro.html
		https://lnsw-rec.portail.gsitlab.com/web/s777procom/home
		https://platform.xr4all.eu/s777procom
		https://justpaste.it/u/s777pro
		https://pantip.com/profile/7113927#topics
		https://www.emoneyspace.com/s777pro
		https://www.veoh.com/users/s777pro
		https://app.glosbe.com/profile/6950291835147259102
		https://hdvietnam.org/members/s777pro.2076664/
		https://seedandspark.com/user/s777pro
		https://meadd.com/s777pro"
    
    """
    url = f"https://s777pro.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "s777pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

