from transformers import AutoImageProcessor, AutoModel
from create_metadata import from_xml_to_image_metadata
from Qdrant_operation.operations import to_create_collection, to_delete_collection, upload
import os
from PIL import Image
import torch 

def find_image_file(filename, image_dir):
    filename_lower = filename + ".JPG"
    file_map = {f:f for f in os.listdir(image_dir)}
    if filename_lower in file_map:
        return os.path.join(image_dir, file_map[filename_lower])
    else:
        return None
# -------- 核心函式：依 metadata 順序生成 embedding --------
def generate_embeddings_by_metadata(metadata_list, img_dir, processor, model, device):
    all_embeddings = []
    found_filenames = []
    for meta in metadata_list:
        filename = meta["filename"]
        image_path = find_image_file(filename, img_dir)
        if image_path is None:
            print(f"找不到影像: {filename}")
            continue

        # 讀取影像
        image = Image.open(image_path).convert("RGB")

        # preprocessing
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # forward
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        all_embeddings.append(embedding)
        found_filenames.append(filename)
    print("embedding ✅")
    return found_filenames, all_embeddings

# DINOv2 模型能夠對於目前所有照片資料產生足夠好的嵌入空間
# 載入 DINOv2-base pretrain model
# 根據現有資料產生 meta-data 並上傳到 Qdrant 資料庫 
# 進行下游任務，給予 embedding 加上任意的 head (classification, segmentation)
# 檢索相關資料，透過 Qdrant 設定篩選條件調出 embedding 就好 


def uploads(xml_dirs, img_dirs, collection_names):
    for xml_dir, image_dir, collection_name in zip(xml_dirs,img_dirs,collection_names):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "facebook/dinov2-base"
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device) 
        model.eval() 
        metadata = from_xml_to_image_metadata(xml_dir)
        _, embedding_list = generate_embeddings_by_metadata(metadata, image_dir, processor, model, device)
        db_url = "http://192.168.1.30:6333"
        to_create_collection(url=db_url, name = collection_name, vector_size = 768)
        upload(url = db_url, embedding_list = embedding_list, metadata = metadata, collection = collection_name)

def main():
    xml_dirs = [r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害徵狀影像資料集\標註檔",
                r"C:\Users\user\Downloads\茶病害徵狀影像資料集\標註檔"] 
    img_dirs = [r"C:\Users\user\Downloads\水稻病害徵狀影像資料集\水稻病害徵狀影像資料集\影像檔",
                r"C:\Users\user\Downloads\茶病害徵狀影像資料集\影像集"] 
    collection_names = ["rice disease DINOv2 pretrained", "tea disease DINOv2 pretrained"]
    uploads(xml_dirs,img_dirs,collection_names)

if __name__ == "__main__":
    main()


