from django.shortcuts import render

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
