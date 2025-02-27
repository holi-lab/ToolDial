import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nh_p_lux(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Nhà Đẹp Lux là đơn vị chuyên tư vấn thiết kế thi công trọn gói biệt thự, nhà ống , nội thất chung cư, xây nhà chìa khóa trao tay, thiết kế thi công lâu đài trọn gói giá rẻ trên toàn quốc. Với nhiều năm kinh nghiệm trong lĩnh vực xây nhà, nhadeplux cam kết sẽ mang lại cho khách hàng những sản phẩm tốt nhất, chất lượng nhất với giá cả hợp lí nhất.
		Hastag : #Nhà_Đẹp_Lux  #nhadeplux #thietkethicongnhadep #thicongnoithattrongoi #noithatchungcu #noithatnhaong #xaybietthu234tang
		Hotline: 0978988780 | 0246.260.2342
		E-mail: nhadep1549@gmail.com  
		MST: 0106478371, ngày cấp 10/3/2014
		Địa chỉ Số 15 – Ngõ 49 – Huỳnh Thúc Kháng – Đống Đa – Hà Nội
		Xưởng sản xuất: 107 – Thôn Thượng – Cự Khê – Thanh Oai – Hà Nội
		Website: https://nhadeplux.com/ 
		 Xem thêm https://sites.google.com/view/nhadepluxcom/ 
		https://vi-vn.facebook.com/XuongSanXuatBanGheSofa/
		https://nhadepluxcom.tumblr.com/
		https://www.walkscore.com/people/114477771174/nhadepluxcomhttps://pxhere.com/en/photographer-me/384481
		8
		https://www.beatstars.com/nhadepluxcom 
		https://guides.co/p/nhadeplux-com 
		https://kuwestions.248am.com/user/nhadepluxcom
		https://www.ourboox.com/i-am/nhadeplux-com/
		http://kiredu.ru/UserProfile/tabid/182/userId/87090/Default.aspx
		https://www.exchangle.com/nhadepluxcom
		https://storium.com/user/nhadepluxcom
		https://www.bitrated.com/nhadepluxcom
		https://www.youtube.com/channel/UCGO8l2FiAU6XDCJBJK8S-mQ/about
		https://twitter.com/nhadepluxcom
		https://www.pinterest.com/nhadepluxcom/_saved/
		https://vi.gravatar.com/nhadepluxcom
		https://nhadepluxcom.blogspot.com/2022/06/nhadepluxcom_8.html
		https://medium.com/@nhadepluxcom/about
		https://www.openstreetmap.org/user/nhadepluxcom
		https://www.behance.net/nhadepluxcom
		https://bandcamp.com/nhadepluxcom
		https://www.twitch.tv/nhadepluxcom
		https://www.liveinternet.ru/users/nhadepluxcom/profile
		https://profile.hatena.ne.jp/nhadepluxcom/profile
		https://issuu.com/nhadepluxcom
		https://ok.ru/profile/587391085006
		https://dribbble.com/nhadepluxcom"
    
    """
    url = f"https://nhadepluxcom.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nhadepluxcom.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

