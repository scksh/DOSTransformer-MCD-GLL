from pymatgen.ext.matproj import MPRester
import pandas as pd
import sys

# 본인의 API 키로 교체하세요
USER_API_KEY = "v2m4uHWMix8TGzyIboGTUYJs9B7gioNv"

if __name__ == "__main__":
    icsd_pd = pd.read_csv('raw/mpdata/mp_data.csv', header=None)
    
    with MPRester(USER_API_KEY) as m:
        # 자료들의 ID 리스트를 가져옵니다.
        material_ids = icsd_pd.iloc[:, 0].to_list()
        results = m.summary.search(material_ids=material_ids, fields=["material_id", "cif"])
        
        # CIF 파일 저장
        for result in results:
            if "cif" in result and result["cif"]:  # CIF 데이터가 있는지 확인
                with open(f"raw/{result['material_id']}.cif", "w") as f:
                    f.write(result["cif"])
    
    print("Download complete!")