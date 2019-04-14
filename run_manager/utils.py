import matplotlib.pyplot as plt


def visualize_util(recon_images, recon_images2, images):
    plt.subplot(5, 2, 1)
    plt.imshow(recon_images[0])
    plt.subplot(5, 2, 2)
    plt.imshow(recon_images[1])

    plt.subplot(5, 2, 3)
    plt.imshow(images[0])
    plt.subplot(5, 2, 4)
    plt.imshow(recon_images2[0])

    plt.subplot(5, 2, 5)
    plt.imshow(images[1])
    plt.subplot(5, 2, 6)
    plt.imshow(recon_images2[1])

    plt.subplot(5, 2, 7)
    plt.imshow(images[2])
    plt.subplot(5, 2, 8)
    plt.imshow(recon_images2[2])

    plt.subplot(5, 2, 9)
    plt.imshow(images[3])
    plt.subplot(5, 2, 10)
    plt.imshow(recon_images2[3])

    plt.show()
