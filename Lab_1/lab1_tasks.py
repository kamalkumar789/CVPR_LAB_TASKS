
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np;


def display_image(img):
    plt.imshow(img);
    plt.show();

# change directory according to your directory 
image = img.imread("/home/kamal/CVPR_Labs/Lab_1/sphinx.jpg");
# plt.show();

## Dimensions / size of an image
print(image.shape);

## printing a pixel of each color. 
print(image[230,5]);

## printing a pixel of red image.
print(image[230,5,0]);


copied_image = np.copy(image);
copied_image[200:301, 100:151] = 255; 
# Display the modified image
display_image(copied_image);


copied_image = np.copy(image);
cropped_image = copied_image[200:301, 100:151];
# Display the cropped part only.
display_image(cropped_image);

# save this cropped image in separate file
plt.imsave("/home/kamal/CVPR_Labs/Lab_1/cropped_image.png", cropped_image);
normalized_image = image / 255;
display_image(normalized_image);


greyimg = normalized_image[:,:,0]*0.30 + normalized_image[:,:,1]*0.59 + normalized_image[:,:,2]*0.11;
display_image(greyimg);