
from PIL import Image 

print("loading image")
image = Image.open("testimages/ropher.jpg")

from fashion_clip.fashion_clip import FashionCLIP
print("loading fashion clip")
fclip = FashionCLIP('fashion-clip')
print("fashion clip loaded")
images = [image]

# we create image embeddings and text embeddings
image_embeddings = fclip.encode_images(images, batch_size=1)

print("image embeddings ", image_embeddings)
print("image embeddings shape ", image_embeddings.shape)
#text_embeddings = fclip.encode_text(texts, batch_size=32)
print("image embeddings created")

# we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
#image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
#text_embeddings = text_embeddings/np.linalg.norm(text_embeddings, ord=2, axis=-1, keepdims=True)