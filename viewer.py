import numpy as np
import matplotlib.pyplot as plt

# 1. Load your file
data = np.load('/Users/debasmibasu/Documents/dse-6_braincognition/dermamnist.npz')

# 2. See what arrays are inside (e.g., 'images', 'arr_0')
print("Available arrays:", data.files)

# 3. Pull the image array (replace 'arr_0' with your actual array name)
images = data['val_images']

# 4. Check the shape of your data
print("Data shape:", images.shape)

# 5. Display the first image in the array
# (If your array holds multiple images, use images[0]. If it's just one image, use images)
plt.imshow(images[10])  # Remove cmap='gray' if it's a color image
plt.axis('off')                      # Hides the axis grid lines
plt.show()                           # Opens a window showing your image
