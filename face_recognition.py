# importing the required libraries
import numpy as np
from face_embeddings import Face_emb

# recognition using cos metric 
def cos_metric(fase_emb_img, fase_emb_list, threshold=0.8 ):
    
    distance = []
    for emb in fase_emb_list:
        distance.append(np.dot(fase_emb_img, emb)/(np.linalg.norm(fase_emb_img) * np.linalg.norm(emb)))
    if max(distance) >  threshold:
        return 'Yes, the person is in the database - cos metric'
    return 'No, the person is in the database - cos metric'       

# recognition using euclidean metric 
def euclid_metric(fase_emb_img, fase_emb_list, threshold = 12):
    
    distance = []
    for emb in fase_emb_list:
        distance.append(np.linalg.norm(np.array(fase_emb_img) - np.array(emb)))
    if min(distance) < threshold:
        return 'Yes, the person is in the database - euclidean metric'
    return 'No, the person is in the database - euclidean metric'
    
def recognition_face(face_path):
    
    face_emb = Face_emb()
    fase_emb_list = face_emb.get_fase_emb_list()
    fase_emb_img = face_emb.embeddings_img(face_path)
    
    print(cos_metric(fase_emb_img, fase_emb_list))
    print(euclid_metric(fase_emb_img, fase_emb_list))

if __name__ == '__main__':
    face_path = r'img_new.jpg'
    recognition_face(face_path)