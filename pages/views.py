from django.shortcuts import render
from django.db import connection
from django.conf import settings
from django.core.files.storage import default_storage
from PIL import Image
import numpy as np
import io
import os

# Create your views here.

def home(request):
    """Home page view"""
    return render(request, 'pages/home.html')

def about(request):
    """About page view"""
    return render(request, 'pages/about.html')

def features(request):
    """Features page view"""
    return render(request, 'pages/features.html')

def contact(request):
    """Contact page view"""
    if request.method == 'POST':
        # Handle form submission here
        name = request.POST.get('name')
        email = request.POST.get('email')
        message = request.POST.get('message')
        # Process the contact form (save to database, send email, etc.)
        return render(request, 'pages/contact.html', {'success': True})
    
    return render(request, 'pages/contact.html')

def upload(request):
    """Upload wardrobe item and generate FashionCLIP embeddings"""
    if request.method == 'POST':
        try:
            # Get form data
            title = request.POST.get('title')
            uploaded_file = request.FILES.get('image')
            
            if not uploaded_file:
                return render(request, 'pages/upload.html', {
                    'error': 'Please select an image file'
                })
            
            # Validate file size (10MB max)
            if uploaded_file.size > 10 * 1024 * 1024:
                return render(request, 'pages/upload.html', {
                    'error': 'File size must be less than 10MB'
                })
            
            # Load image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Initialize FashionCLIP
            from fashion_clip.fashion_clip import FashionCLIP
            fclip = FashionCLIP('fashion-clip')
            
            # Generate embeddings
            images = [image]
            image_embeddings = fclip.encode_images(images, batch_size=1)
            
            # Normalize embeddings
            image_embeddings = image_embeddings / np.linalg.norm(
                image_embeddings, ord=2, axis=-1, keepdims=True
            )
            
            # Extract single embedding vector (remove batch dimension)
            embedding_vector = image_embeddings[0].tolist()
            
            # Save uploaded file to media directory
            # Create a unique filename to avoid conflicts
            import uuid
            file_extension = os.path.splitext(uploaded_file.name)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = os.path.join('wardrobe_items', unique_filename)
            
            # Save the file
            saved_path = default_storage.save(file_path, uploaded_file)
            
            # Save to database
            with connection.cursor() as cursor:
                # Insert into database with the saved file path
                cursor.execute("""
                    INSERT INTO wardrobe_items (title, path, embedding)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, [title, saved_path, embedding_vector])
                
                item_id = cursor.fetchone()[0]
            
            # Return success page
            return render(request, 'pages/upload.html', {
                'success': True,
                'item': {
                    'id': item_id,
                    'title': title,
                    'path': saved_path
                },
                'embedding_shape': f"({len(embedding_vector)},)"
            })
            
        except Exception as e:
            return render(request, 'pages/upload.html', {
                'error': f'Error processing image: {str(e)}'
            })
    
    return render(request, 'pages/upload.html')

def search(request):
    """Search for similar items using FashionCLIP embeddings and Python-based cosine similarity"""
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('image')
            
            if not uploaded_file:
                return render(request, 'pages/search.html', {
                    'error': 'Please select an image file'
                })
            
            # Validate file size (10MB max)
            if uploaded_file.size > 10 * 1024 * 1024:
                return render(request, 'pages/search.html', {
                    'error': 'File size must be less than 10MB'
                })
            
            # Load image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Initialize FashionCLIP
            from fashion_clip.fashion_clip import FashionCLIP
            fclip = FashionCLIP('fashion-clip')
            
            # Generate embeddings for query image
            images = [image]
            query_embeddings = fclip.encode_images(images, batch_size=1)
            
            # Normalize embeddings
            query_embeddings = query_embeddings / np.linalg.norm(
                query_embeddings, ord=2, axis=-1, keepdims=True
            )
            
            # Extract single embedding vector (keep as numpy array)
            query_vector = query_embeddings[0]
            
            # Perform similarity search in Python
            with connection.cursor() as cursor:
                # Fetch all embeddings from database
                cursor.execute("""
                    SELECT id, title, path, embedding
                    FROM wardrobe_items
                """)
                
                rows = cursor.fetchall()
                
                # Compute similarities in Python
                similarities = []
                for item_id, title, path, embedding in rows:
                    # Parse embedding: PostgreSQL returns VECTOR as string like '[0.1,0.2,...]'
                    if isinstance(embedding, str):
                        # Remove brackets and split by comma
                        embedding_str = embedding.strip('[]')
                        embedding_list = [float(x) for x in embedding_str.split(',')]
                    else:
                        embedding_list = embedding
                    
                    # Convert to numpy array
                    stored_vector = np.array(embedding_list, dtype=np.float32)
                    
                    # Compute cosine similarity (dot product since vectors are normalized)
                    similarity = np.dot(query_vector, stored_vector)
                    
                    similarities.append({
                        'id': item_id,
                        'title': title,
                        'path': path,
                        'similarity': float(similarity),
                        'similarity_percent': float(similarity) * 100
                    })
                
                # Sort by similarity (highest first) and take top 5
                results = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:5]
            
            return render(request, 'pages/search.html', {
                'results': results
            })
            
        except Exception as e:
            return render(request, 'pages/search.html', {
                'error': f'Error processing search: {str(e)}'
            })
    
    return render(request, 'pages/search.html')
