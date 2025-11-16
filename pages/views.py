from django.shortcuts import render
from django.db import connection
from PIL import Image
import numpy as np
import io

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
            embedding_vector = image_embeddings[0]
            
            # Convert to list for database storage
            embedding_list = embedding_vector.tolist()
            
            # Save to database
            with connection.cursor() as cursor:
                # Store image path (for now, just use the filename)
                # In production, you'd save the file to media storage
                path = uploaded_file.name
                
                # Insert into database
                cursor.execute("""
                    INSERT INTO wardrobe_items (title, path, embedding)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, [title, path, embedding_list])
                
                item_id = cursor.fetchone()[0]
            
            # Return success page
            return render(request, 'pages/upload.html', {
                'success': True,
                'item': {
                    'id': item_id,
                    'title': title,
                    'path': path
                },
                'embedding_shape': f"({len(embedding_list)},)"
            })
            
        except Exception as e:
            return render(request, 'pages/upload.html', {
                'error': f'Error processing image: {str(e)}'
            })
    
    return render(request, 'pages/upload.html')
