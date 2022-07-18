# Object Detection with SIFT and GHT
Object recognition algorithm for multiple instances using the Generalized Hough Transform (GHT) and Scale-invariant feature transform (SIFT) descriptors.

## Method
The general idea behind the Hough Transform is that objects are found through a voting procedure in a configuration or **Hough space**. If evidence for a particular object configuration is found in the image, a vote is added for this configuration. In our case such evidence is a matched feature vector from the object image to the scene image. The **location, orientation** and **scale** of the feature with respect to the object's coordinate system is known from the object image. If the feature is also found somewhere in the scene image, it is possible to calculate the likely location, orientation and scale of the whole object and add a "vote" for that configuration. After all matched feature vectors have voted, clusters in the configuration space correspond to possible object instances

<p align="center">
  <img width="820" height="350" src="https://user-images.githubusercontent.com/63703454/179578291-5d4a8443-9995-49e6-818f-a88a1805cb06.png">
</p>

## Results
![image](https://user-images.githubusercontent.com/63703454/179576554-f63bdd06-8d10-4669-a009-dec3e431c5f1.png)
![image](https://user-images.githubusercontent.com/63703454/179576652-791857a0-c670-4154-abed-db037ce48584.png)
![image](https://user-images.githubusercontent.com/63703454/179576796-8f409429-c559-428a-8ddc-96c8104f9b4f.png)
![image](https://user-images.githubusercontent.com/63703454/179576882-316f711a-1708-45c1-8a27-123ff3928c02.png)

