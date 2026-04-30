import requests
import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def get_dataset_and_queries(images_path, queries_path, qrels_path, output_dir, total_docs=200, num_queries=5):
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    
    queries_df = pd.read_csv(queries_path, sep="\t", header=None, names=['qid', 'query_text'])
    qrels_df = pd.read_csv(qrels_path, sep=" ", header=None, names=['qid', 'q0', 'img_id', 'rel'])
    
    if len(existing_files) >= total_docs:
        existing_ids = [f.split('.')[0] for f in existing_files]
        
        valid_qrels = qrels_df[(qrels_df['rel'] == 1) & (qrels_df['img_id'].isin(existing_ids))]
        valid_qids = valid_qrels['qid'].unique()
        
        if len(valid_qids) >= num_queries:
            selected_qids = np.random.choice(valid_qids, num_queries, replace=False)
            query_mapping = []
            target_images = []
            
            for qid in selected_qids:
                query_text = queries_df[queries_df['qid'] == qid]['query_text'].values[0]
                all_rel_imgs = valid_qrels[valid_qrels['qid'] == qid]['img_id'].tolist()
                base_img = np.random.choice(all_rel_imgs)
                
                target_images.extend(all_rel_imgs)
                query_mapping.append({
                    'qid': qid,
                    'query_text': query_text,
                    'target_img_id': base_img,
                    'all_relevant_ids': all_rel_imgs
                })
                
            target_images = list(set(target_images))
            available_distractors = [img for img in existing_ids if img not in target_images]
            distractors_needed = max(0, total_docs - len(target_images))
            chosen_distractors = list(np.random.choice(available_distractors, distractors_needed, replace=False))
            
            final_ids = target_images + chosen_distractors
            downloaded_data = [{'id': img, 'path': os.path.join(output_dir, f"{img}.jpg")} for img in final_ids]
            
            return downloaded_data, query_mapping
        else:
            print("Imagens nao acharam targets suficientes")

    print("Baixando")
    images_df = pd.read_csv(images_path, sep="\t")
    relevant_qrels = qrels_df[qrels_df['rel'] == 1]
    valid_qids = relevant_qrels['qid'].unique()
    
    selected_qids = np.random.choice(valid_qids, num_queries, replace=False)
    
    target_images = []
    query_mapping = []
    
    for qid in selected_qids:
        query_text = queries_df[queries_df['qid'] == qid]['query_text'].values[0]
        all_rel_imgs = relevant_qrels[relevant_qrels['qid'] == qid]['img_id'].tolist()
        base_img = np.random.choice(all_rel_imgs)
        
        target_images.extend(all_rel_imgs)
        query_mapping.append({
            'qid': qid,
            'query_text': query_text,
            'target_img_id': base_img,
            'all_relevant_ids': all_rel_imgs
        })
        
    target_images = list(set(target_images))
    if len(target_images) > total_docs: total_docs = len(target_images)
        
    all_img_ids = images_df['id'].unique()
    available_distractors = [img for img in all_img_ids if img not in target_images]
    remaining_needed = total_docs - len(target_images)
    distractor_images = list(np.random.choice(available_distractors, remaining_needed, replace=False))
    
    final_images_df = images_df[images_df['id'].isin(target_images + distractor_images)]
    
    downloaded_data = []
    for _, row in tqdm(final_images_df.iterrows(), total=len(final_images_df), desc=f"Baixando {len(final_images_df)} Imagens"):
        img_id = row["id"]
        url = row.get("url", row.iloc[-1]) 
        save_path = os.path.join(output_dir, f"{img_id}.jpg")

        if os.path.exists(save_path):
            downloaded_data.append({"id": img_id, "path": save_path})
            continue

        try:
            resp = requests.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192): f.write(chunk)
            downloaded_data.append({"id": img_id, "path": save_path})
        except Exception:
            pass 

    return downloaded_data, query_mapping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO('yolov8n.pt')
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)
vit.heads = torch.nn.Identity()
vit.eval()

vit_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_vit_features(image_crop):
    if image_crop.size == 0: return torch.zeros(768).numpy()
    
    image_crop_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_crop_rgb)
    input_tensor = vit_preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = vit(input_tensor)
    
    return features.squeeze().cpu().numpy()

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return []
        
    results = yolo_model(img, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    regions = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if w == 0 or h == 0: continue
        
        crop = img[y1:y2, x1:x2]
        features = extract_vit_features(crop)
        regions.append({'box': [x1, y1, w, h], 'features': features})
        
    return regions

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

def rank_results(query_features, query_box, database, alpha=0.7):
    ranked = []
    for item in database:
        visual_sim = cosine_similarity([query_features], [item['features']])[0][0]
        spatial_sim = calculate_iou(query_box, item['box'])
        final_score = (alpha * visual_sim) + ((1 - alpha) * spatial_sim)
        
        ranked.append({
            'image_id': item['image_id'], 'box': item['box'],
            'score': final_score, 'visual_sim': visual_sim, 'spatial_sim': spatial_sim
        })
        
    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked

if __name__ == "__main__":
    images_tsv_path = "images.tsv" 
    queries_tsv_path = "queries.tsv"
    qrels_path = "qrels.txt"
    output_dir = "downloaded_images"
    pdf_output_filename = "Resultados.pdf"

    dataset, query_mapping = get_dataset_and_queries(
        images_tsv_path, queries_tsv_path, qrels_path, output_dir, total_docs=200, num_queries=5
    )
    
    database_index = []
    for img_data in tqdm(dataset, desc="Indexando"):
        regions = process_image(img_data['path'])
        for r in regions:
            database_index.append({
                'image_id': img_data['id'], 'box': r['box'], 'features': r['features']
            })

    print(f"\nGerando: {pdf_output_filename}")
    
    with PdfPages(pdf_output_filename) as pdf:
        for q in query_mapping:
            print(f"Processando Query {q['qid']}...")
            
            target_regions = [r for r in database_index if r['image_id'] == q['target_img_id']]
            if not target_regions:
                print(f"  [!] Pulando {q['qid']}")
                continue
                
            main_query_region = max(target_regions, key=lambda r: r['box'][2] * r['box'][3])
            q_features, q_box = main_query_region['features'], main_query_region['box']
            
            results = rank_results(q_features, q_box, database_index)
            filtered_results = [res for res in results if res['image_id'] != q['target_img_id']]
            
            top_3 = filtered_results[:3]
            correct_in_top_3 = sum(1 for res in top_3 if res['image_id'] in q['all_relevant_ids'])
            precision_at_3 = correct_in_top_3 / 3.0
            
            fig = plt.figure(figsize=(16, 6))
            fig.suptitle(f"Query {q['qid']}: {q['query_text']} (Precisao: {precision_at_3:.2f})", fontsize=18, fontweight='bold')
            
            target_img_path = os.path.join(output_dir, f"{q['target_img_id']}.jpg")
            img_target = cv2.cvtColor(cv2.imread(target_img_path), cv2.COLOR_BGR2RGB)
            x, y, w, h = q_box
            cv2.rectangle(img_target, (x, y), (x+w, y+h), (0, 255, 0), 4) 
            
            ax1 = plt.subplot(1, 4, 1)
            ax1.imshow(img_target)
            ax1.set_title(f"Referencia\n (ID: {q['target_img_id']})")
            ax1.axis('off')
            
            for i, res in enumerate(top_3):
                res_img_path = os.path.join(output_dir, f"{res['image_id']}.jpg")
                res_img = cv2.cvtColor(cv2.imread(res_img_path), cv2.COLOR_BGR2RGB)
                
                rx, ry, rw, rh = res['box']
                color = (0, 255, 0) if res['image_id'] in q['all_relevant_ids'] else (255, 0, 0)
                cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), color, 4) 
                
                ax = plt.subplot(1, 4, i+2)
                ax.imshow(res_img)
                ax.set_title(f"Rank {i+1} (ID: {res['image_id']})\nPontuacao: {res['score']:.3f}")
                ax.axis('off')
                
            plt.tight_layout()
            pdf.savefig(fig) 
            plt.close() 