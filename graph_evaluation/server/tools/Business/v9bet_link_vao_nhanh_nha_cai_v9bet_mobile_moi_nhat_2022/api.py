import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v9bet_link_vao_nhanh_nha_cai_v9bet_mobile_moi_nhat_2022(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "V9bet là tên một nhà cái cá cược trực tuyến, xuất hiện lần đầu tiên tại thị trường Việt Nam vào khoảng tháng 10 năm 2012. Nhà cái V9bet cung cấp các dịch cá cược như thể thao, Casino Online (Sòng bạc trực tuyến), Slots game, Xổ số, Game 3D,… Với giao diện đẹp mắt, trò chơi phòng phú và đa dạng, dễ dàng nạp rút nhanh với việc hỗ trợ nhiều hình thức thanh toán các dịch vụ của công ty đã thu hút rất đông người đam mê giải trí đỏ đen tìm đến.
		#v9bet #V9BET88 #v9betvi # /m/033_l8  /m/033_l8 #Casino 
		Địa chỉ: 388 Nguyễn Thị Thập, Tân Quy, Quận 7, Thành phố Hồ Chí Minh 700000
		Hotline 0582 516 325
		Website https://v9betvi.net/ 
		https://www.facebook.com/v9betvinet 
		https://twitter.com/v9betvinet 
		https://www.pinterest.com/v9betvinet 
		https://sites.google.com/view/v9betvi-net 
		https://englishbaby.com/findfriends/gallery/detail/2404943
		https://storium.com/user/v9betvi
		https://www.drupalgovcon.org/user/204736
		https://www.mpug.com/members/v9betvi/profile/
		https://gettogether.community/profile/36666/
		https://chillspot1.com/user/v9betvinet
		https://www.elephantjournal.com/profile/v9betvinet/
		https://www.pokecommunity.com/member.php?u=1075309
		http://www.travelful.net/location/4956106/vietnam/v9betvi
		https://www.ohay.tv/profile/v9betvi
		https://www.bigpictureclasses.com/users/v9betvi
		http://www.talkstats.com/members/v9betvinet.143600/#about
		https://v9betvinet1.wixsite.com/v9betvi
		https://lazi.vn/user/vi.vbet
		https://bsaber.com/members/v9betvi/info/
		https://openprocessing.org/user/334654?view=activity#topPanel
		https://www.adaxes.com/questions/user/v9betvi
		https://fyi.org.nz/user/v9betvi/profile
		https://www.castingcall.club/v9betvi
		http://uid.me/v9betvi_net
		https://www.scribblemaps.com/maps/view/v9betvi/lvExUXDG80
		https://career.habr.com/v9betvinet
		http://lightroom.adobe.com/u/v9betvi"
    
    """
    url = f"https://v9bet-link-vao-nhanh-nha-cai-v9bet-mobile-moi-nhat-2022.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "v9bet-link-vao-nhanh-nha-cai-v9bet-mobile-moi-nhat-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

