# importing the required libraries
from imgbeddings import imgbeddings
from PIL import Image

class Face_emb:
    fase_emb_list = []
    
    def __init__(self):
        
        # loading the `imgbeddings`
        self.ibed = imgbeddings()        
        self.img_list = self.load_img()
        self.embeddings_face_bd(self.img_list)         

    # loading the face image 
    def load_img(self):
        img_list = []
        
        file_name_1 = r'img_1.jpg'
        file_name_2 = r'img_2.jpg'
        file_name_3 = r'img_3.jpg'
        file_name_4 = r'img_4.jpg'
    
        img_list.append(Image.open(file_name_1))
        img_list.append(Image.open(file_name_2))
        img_list.append(Image.open(file_name_3))
        img_list.append(Image.open(file_name_4))
    
        return img_list

    # calculating the embeddings img bd
    def embeddings_face_bd(self, img_list):
        
        emb_list = []       
        for img in img_list:
            Face_emb.fase_emb_list.append(self.ibed.to_embeddings(img)[0])
        
    def get_fase_emb_list(self):
        return Face_emb.fase_emb_list         
            
    # calculating the embedding img
    def embeddings_img(self, face_path):       
        file_name = Image.open(face_path)        
        return self.ibed.to_embeddings(file_name)[0]   
     