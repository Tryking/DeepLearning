"""
http://cs231n.github.io/classification/
"""
"""
Challenges. 
Since this task of recognizing a visual concept (e.g. cat) is relatively trivial for a human to perform, 
it is worth considering the challenges involved from the perspective of a Computer Vision algorithm. 
As we present (an inexhaustive) list of challenges below, keep in mind the raw representation of images as a 3-D array of brightness values:

Viewpoint variation. 
A single instance of an object can be oriented in many ways with respect to the camera.
Scale variation. 
Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image).
Deformation. 
Many objects of interest are not rigid bodies and can be deformed in extreme ways.
Occlusion. 
The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels) could be visible.
Illumination conditions. The effects of illumination are drastic on the pixel level.
Background clutter. 
The objects of interest may blend into their environment, making them hard to identify.
Intra-class variation. The classes of interest can often be relatively broad, such as chair. There are many different types of these objects, each with their own appearance.

A good image classification model must be invariant to the cross product of all these variations, while simultaneously retaining sensitivity to the inter-class variations.
"""