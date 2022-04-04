import pickle

with open('flickr30k_class_name.txt','r') as f:
  lines=f.readlines()

tokens_list =[e.strip() for e in lines]
token_2_image={}
for token in tokens_list:
  token_2_image[token]=[]

for (i,cap_instance) in enumerate(cap_flickr30k):
  tokens_checked=[]
  for cap in cap_instance:
    
    bboxes = cap['bbox']
    tokens =cap['clss']
    
    images=[dic_flickr30k['images'][i]['file_path']]*len(tokens)

    

    for token,image,bbox in zip( tokens,images,bboxes):      
      if token not in tokens_checked:
        # print(token,image,bbox)
        tokens_checked.append(token)
        # print(tokens_checked)
        # print(token,image,bbox)
        token_2_image[token].append((image,bbox))
        

pickle.dump( token_2_image , open( "token_2_image_flick30k.p", "wb" ) )
#token_2_image_flickr30k = pickle.load( open( "token_2_image_flick30k.p", "rb" ) )
