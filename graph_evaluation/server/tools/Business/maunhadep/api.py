import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def m_u_nh_p(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mẫu nhà đẹp là nhóm của Công ty TNHH Thiết kế Thi công Nhà đẹp. Với mục đích muốn tạo ra một sân chơi chung, tham khảo những mẫu nhà đẹp và chia sẻ kinh nghiệm trong lĩnh vực thiết kế, thi công nhà. Chính vì thế nếu các bạn đang có nhu cầu xây nhà có thể vào tham khảo và thảo luận cùng các thành viên có nhiều kinh nghiệm trong lĩnh vực.
		Với 10 năm kinh nghiệm thiết kế Nhà đẹp sẽ có những chuyên gia trong lĩnh vực này có thể hỗ trợ và tư vấn thêm cho các bạn trong quá trình xây dựng nhà. Ngoài ra chúng tôi đã có hàng nghìn mẫu thiết kế trong những năm vừa qua
		Group https://www.facebook.com/groups/maunhadep1549/about 
		Số 15 – Ngõ 49 – Huỳnh Thúc Kháng – Đống Đa – Hà Nội
		Xưởng sản xuất: 107 – Thôn Thượng – Cự Khê – Thanh Oai – Hà Nội
		MST: 0106478371, ngày cấp 10/3/2014
		Hotline: 0976012358
		E-mail: nhadep1549@gmail.com
		https://forum.acronis.com/user/403798
		https://www.leetchi.com/c/maunhadep
		https://www.magcloud.com/user/maunhadep
		https://kuula.co/profile/maunhadep
		https://kyteapp.mn.co/members/11603638
		https://paper.li/~/publisher/7c2098f3-56ff-48ac-b111-5549d664be29
		https://australian-school-holidays.mn.co/members/11603650
		https://hashnode.com/@maunhadep
		https://pubhtml5.com/homepage/ytpe
		https://www.plurk.com/maunhadep
		https://www.bitchute.com/channel/89i9oyyFpIBZ/
		https://www.beatstars.com/maunhadep
		http://wiki.cs.hse.ru/%D0%A3%D1%87%D0%B0%D1%81%D1%82%D0%BD%D0%B8%D0%BA:Maunhadep
		https://hearthis.at/maunhadep/
		https://www.question2answer.org/qa/user/maunhadep
		https://muckrack.com/mau-nhadep/bio
		https://reedsy.com/discovery/user/maunhadep
		https://myanimelist.net/profile/maunhadep2022
		https://beacons.ai/maunhadep
		https://www.beatstars.com/maunhadep/about 
		https://leetcode.com/maunhadep/
		https://cplusplus.com/user/maunhadep/
		https://mootools.net/forge/profile/maunhadep
		https://community.dynamics.com/members/maunhadep
		https://themehunt.com/profile/nhadep"
    
    """
    url = f"https://maunhadep.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maunhadep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

