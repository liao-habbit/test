import os
import xml.etree.ElementTree as ET

def from_xml_to_image_metadata(xml_dir):
    metadata_list = []
    for file in os.listdir(xml_dir):
        if not file.endswith(".xml"):
            continue
        xml_path = os.path.join(xml_dir, file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 影像層級資訊
        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        depth = int(size.find("depth").text)
        objects = []
        for idx, obj in enumerate(root.findall("object"), start=1):
            obj_dict = {
                "object_id": idx,
                "name": obj.find("name").text,
                "pose": obj.find("pose").text,
                "truncated": int(obj.find("truncated").text),
                "difficult": int(obj.find("difficult").text)
            }
            # Basic_Info_1~9
            for i in range(1, 10):
                tag = f"Basic_Info_{i}"
                elem = obj.find(tag)
                obj_dict[tag] = int(elem.text) if elem is not None else None
            
            # Bounding box
            bndbox = obj.find("bndbox")
            obj_dict.update({
                "xmin": int(bndbox.find("xmin").text),
                "ymin": int(bndbox.find("ymin").text),
                "xmax": int(bndbox.find("xmax").text),
                "ymax": int(bndbox.find("ymax").text)
            })
            objects.append(obj_dict)

        # 每張影像只生成一個 metadata
        image_meta = {
            "filename": filename,
            "width": width,
            "height": height,
            "depth": depth,
            "objects": objects
        }

        metadata_list.append(image_meta)
        print(f"metadata ✅")
    return metadata_list

# 測試用
# xml_dir = r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害徵狀影像資料集\標註檔"
# from_xml_to_image_metadata(xml_dir)