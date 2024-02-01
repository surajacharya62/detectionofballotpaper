import requests
from bs4 import BeautifulSoup
import os

# Define the URL of the website
url = 'https://www.pexels.com/search/dhaka%20topi/'

# Use requests to retrieve data from the URL
response = requests.get(url)

# Parse the content of the response with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all image tags
image_tags = soup.find_all('img')

# Directory to save the downloaded images
os.makedirs('downloaded_images', exist_ok=True)

# Loop over each image tag
for i, img in enumerate(image_tags):
    # Get the image source URL
    img_url = img.get('src')
    # Get the content of the image
    img_data = requests.get(img_url).content
    
    # Write the image data to a file
    with open(f'./downloaded_images/image_{i}.png', 'wb') as handler:
        
        handler.write(img_data)
    # Stop after downloading 100 images
    if i >= 99:
        break
