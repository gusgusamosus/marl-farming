import os
import shutil

# Folder containing your images
image_folder = r"C:\Users\srich\OneDrive\Desktop\Projects\PJT-1\MARL\MARL\disease_classification_mappo\EvaluationData"

# Get all image files
image_files = [f for f in os.listdir(image_folder) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

print(f"Found {len(image_files)} images")

# Create HTML with relative paths
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Random Image Slideshow</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background-color: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            overflow: hidden;
        }
        #slideshow {
            max-width: 90vw;
            max-height: 90vh;
            object-fit: contain;
        }
        .controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255, 255, 255, 0.2);
            padding: 10px 20px;
            border-radius: 25px;
            display: flex;
            gap: 10px;
        }
        button {
            background: rgba(255, 255, 255, 0.8);
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: rgba(255, 255, 255, 1);
        }
        #loading {
            color: white;
            font-family: Arial, sans-serif;
            font-size: 20px;
        }
    </style>
</head>
<body>
    <div id="loading">Loading images...</div>
    <img id="slideshow" alt="Slideshow" style="display:none;">
    <div class="controls">
        <button onclick="togglePause()">Pause</button>
        <button onclick="showRandomImage()">Next</button>
    </div>
    <script>
        const imageFiles = [
"""

# Add image filenames
for img_file in image_files:
    html_content += f'            "images/{img_file}",\n'

html_content += """        ];

        let currentIndex = 0;
        let intervalId;
        let isPaused = false;
        const img = document.getElementById('slideshow');
        const loading = document.getElementById('loading');

        function showRandomImage() {
            currentIndex = Math.floor(Math.random() * imageFiles.length);
            img.src = imageFiles[currentIndex];
        }

        function togglePause() {
            if (isPaused) {
                intervalId = setInterval(showRandomImage, 5000);
                event.target.textContent = 'Pause';
            } else {
                clearInterval(intervalId);
                event.target.textContent = 'Resume';
            }
            isPaused = !isPaused;
        }

        // Show slideshow once first image loads
        img.onload = function() {
            loading.style.display = 'none';
            img.style.display = 'block';
        };

        // Handle image loading errors
        img.onerror = function() {
            console.error('Failed to load:', imageFiles[currentIndex]);
            showRandomImage(); // Try next image
        };

        // Start slideshow
        showRandomImage();
        intervalId = setInterval(showRandomImage, 5000);
    </script>
</body>
</html>
"""

# Create output directory structure
os.makedirs("slideshow_package", exist_ok=True)
os.makedirs("slideshow_package/images", exist_ok=True)

# Copy all images to the package
print("Copying images...")
for idx, img_file in enumerate(image_files):
    src = os.path.join(image_folder, img_file)
    dst = os.path.join("slideshow_package/images", img_file)
    shutil.copy2(src, dst)
    if (idx + 1) % 100 == 0:
        print(f"Copied {idx + 1}/{len(image_files)} images...")

# Save HTML file
with open("slideshow_package/slideshow.html", 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\nSlideshow created successfully in 'slideshow_package' folder!")
print("Open slideshow.html in a web browser to view your slideshow.")
print("The entire 'slideshow_package' folder is portable - copy it anywhere!")
