# CV_PROJECT
IMAGE MATCHING + IMAGE COMPRESSION

IMAGE COMPRESSION
Introduction:
Image compression is a fundamental technique used in digital image processing to reduce the size of an image file while preserving its visual quality to a reasonable extent. The primary goal of image compression is to minimize the amount of data required to represent an image, thereby saving storage space and reducing transmission time over networks. It plays a crucial role in various applications such as digital photography, video streaming, medical imaging, satellite imagery, and more.
Types of Image Compression
Image compression techniques can be broadly categorized into two main types:
1.	Lossless Compression: Lossless compression algorithms preserve all original image data when compressing and decompressing images. They ensure exact reconstruction of the original image without any loss of information. Lossless compression is suitable for scenarios where image fidelity is critical, such as medical imaging or text documents.
2.	Lossy Compression: Lossy compression algorithms achieve higher compression ratios by selectively discarding less important image information during compression. While this results in some loss of image quality, the human visual system may tolerate such losses to a certain extent without significant perceptual degradation. Lossy compression is widely used in applications like digital photography, web images, and video compression.
Methods:
1) JPEG compression
2) Image compression using BTC
3) Colour Quantization


IMAGE MATCHING

Introduction:

Image matching is a fundamental task in computer vision that involves finding correspondences between different images or parts of the same image. It plays a crucial role in various applications such as object recognition, image retrieval, 3D reconstruction, and augmented reality. The goal of image matching is to establish relationships between images based on their visual content, enabling machines to understand and interpret visual information.

Types of Image Matching:
1)Feature-Based Matching:
•	Keypoint Detection: Feature-based matching algorithms detect distinctive points or keypoints in images, such as corners or blobs, that are invariant to transformations like rotation, scaling, and illumination changes.
•	Descriptor Extraction: Once keypoints are detected, descriptors are computed to represent the local appearance around each keypoint. Common descriptors include SIFT (Scale-Invariant Feature Transform), SURF (Speeded-Up Robust Features), and ORB (Oriented FAST and Rotated BRIEF).
•	Matching: Matching involves finding correspondences between keypoints in different images based on their descriptors. This is typically done using techniques like nearest neighbor search and geometric verification.
2)Geometric Matching:
•	RANSAC (Random Sample Consensus): Feature-based matching algorithms detect distinctive points or keypoints in images, such as corners or blobs, that are invariant to transformations like rotation, scaling, and illumination changes.
3)Deep Learning-Based Matching:
•	Siamese Networks: Siamese networks are a type of neural network architecture used for learning similarity between two inputs. In image matching, Siamese networks can learn to compare pairs of images and determine their similarity or dissimilarity.
•	Convolutional Neural Network(CNNs): NNs have shown remarkable success in various computer vision tasks, including image matching. They can learn hierarchical representations of images, capturing both low-level and high-level features for matching.
4)Graph-Based Matching:
•	Graph-Based Matching: Graph-based matching approaches represent images as graphs, with nodes representing keypoints or image patches and edges representing relationships between them. Matching involves finding correspondences between nodes in different graphs.

Methods to find Descriptors:
1.	Sift 
The SIFT (Scale-Invariant Feature Transform) algorithm is a computer vision technique used for feature detection and description. It detects distinctive key points or features in an image that are robust to changes in scale, rotation, and affine transformations. SIFT(scale invariant feature transform) works by identifying key points based on their local intensity extrema and computing descriptors that capture the local image information around those key points.

2.	N-Sift
NG-SIFT (Normalized -gardient  SIFT) is a variant of the Scale-Invariant Feature Transform (SIFT) algorithm used for extracting keypoints and descriptors from images. Unlike traditional SIFT, which relies on gradient information, NG-SIFT utilizes non-gravitational gradients, computed from magnitude and angle information obtained through methods like Sobel operators. This approach allows NG-SIFT to be robust to image rotations and translations, making it suitable for various computer vision tasks such as object recognition, image stitching, and feature matching.

3.	PCA-Sift
PCA-SIFT is an adaptation of the SIFT algorithm, aiming to overcome computational challenges while maintaining feature quality. It utilizes Principal Component Analysis (PCA) to reduce feature dimensionality without compromising discriminative power. This enhances computational efficiency without sacrificing accuracy, making it suitable for large-scale and real-time applications in computer vision and image processing.


4.	A-Sift
Affine SIFT (A-SIFT) extends the traditional SIFT algorithm to handle affine transformations like shearing, scaling, and skewing, in addition to rotation and translation invariance. These transformations commonly occur in real-world scenarios, especially in object recognition and scene understanding tasks. By detecting affine-covariant keypoints and computing affine-invariant descriptors, A-SIFT enables more accurate feature matching across a wider range of geometric transformations. This enhancement is crucial for robust feature-based image processing, particularly in scenarios where objects undergo non-rigid deformations or perspective changes.

5.	Mops
MOPS (Multi-Scale Oriented Patches) is a feature extraction technique used in computer vision for describing local image patches. It divides image patches into sub-regions and computes histograms of gradient orientations within each sub-region. These histograms capture both the spatial information and the orientation of gradients, allowing MOPS to represent each patch effectively. By concatenating histograms from multiple scales, MOPS creates feature descriptors that are robust to scale variations and preserve both fine and coarse details in the image. This technique is particularly useful for feature-based matching tasks, where it provides efficient and accurate representation of local image structures.

![image](https://github.com/anurag161/CV_PROJECT/assets/102911572/92339202-9e7f-4f59-bf43-0c9e9bcb9232)

IMAGE STITCHING

The Image_Stitching class provides functionality for stitching images together to create panoramic views. Here's a summary of its key features and methods: 
1)Feature Extraction and Matching: Utilizes the SIFT algorithm to extract and match keypoints between two images. 
2)Homography Estimation: Estimates the homography matrix using the RANSAC algorithm to align the images properly. 
3)Mask Generation: Generates masks to blend the images seamlessly, ensuring gradual transitions between them. 
4)Blending: Blends the images together using the estimated homography and masks to create a smooth final panorama. 
5)Parameter Tuning: Allows for the adjustment of parameters like matching ratio, minimum matches, and smoothing window size to optimize stitching results.
![image](https://github.com/anurag161/CV_PROJECT/assets/102911572/07baaadf-d321-43d6-9236-7f796a02ba86)




