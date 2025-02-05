# Env

* Windows X_64 , Visual Studio_2019
* OpenCV 3.4.13
* OpenCV_contrib 3.4.13
* Cplex  Community Edition 20.1 (Community) [參數限制 < 1000]
* CGAL 4.13.1
* OpenGL(glfw & glew)

# VS setting

* ### include

  <img src=".\images\include.jpg" alt="include" style="zoom:50%;" />

* ### lib

  <img src=".\images\lib.jpg" alt="lib" style="zoom:50%;" />

* ### dependency

  <img src=".\images\dependency.jpg" alt="dependency" style="zoom:50%;" />

# Method

### implement the paper：Patch-Based Image Warping for Content-Aware Retargeting

* use segmented image to get each patch 

* use saliency image to get significant image

  <img src=".\images\1.png" alt="`1" style="zoom:50%;" />

  ​		

* Cplex to solve optimization problem based on each patch's significant

  * Two energy terms are deﬁned with an optimization solver to preserve image content.

    * Patch transformation constraint： The degree of deformation of the same patch should be as consistent as possible
    * Grid orientation constraint：  make the grids keep their shapes
    
  * **Patch transformation constraint**

    * for each patches, we need to find a center edge(to compare with other edges in the patch). In the program, I use the first edge in the patch as the center edge.
    * According to the relationship of center edge and the other edges in the patch, we can get the transform matrix of the two edges.

    ​                                  	<img src=".\images\2.png" alt="2" style="zoom:50%;" />

    ​			**T** matrix can be thought of as a rotated matrix.

    > 這邊找到當前patch中所有edges與center edge的關係，如果這個patch全重較高，越會維持這個相對關係。

    * When we get the $s, r$ , we can start to solve the optimization problem. if the patch is important, we don't want it to deform too much. Different between edge after deformed($e^\prime$ ) and origin edge($e = TC$)  should be as little as possible. Therefore we get the formula. 

    ​	   

    <img src=".\images\3.png" alt="3" style="zoom:50%;" />

    ​			The greater the weight, the more it maintains its original appearance.

    * To avoid over-deformation in low-significant patches, we also need to constraint the linear scaling.

    <img src=".\images\4.png" alt="4" style="zoom:50%;" />

    ​       the less significant, the more weight. $m/n, m^\prime/n^\prime $ are origin size and target size, perspectively.

    > 讓不重要的patch縮放效果接近線性內插，避免過度變形。

    * The total patch transformation energy is defined by summing up the individual patch energy term with its weight:

  <img src=".\images\5.png" alt="5" style="zoom: 67%;" />

  * **Grid orientation constraint**

    <img src=".\images\6.png" alt="6" style="zoom:50%;" />

  * **Total energy**：

    <img src=".\images\7.png" alt="7" style="zoom:80%;" />

  * In hard constraint, we need below restricts.

    * boundary constraint to target image size
    * avoid grid flipping

# Result

### Origin

<img src=".\Patch_Base_Resizing\Patch_Base_Resizing\res\butterfly.jpg" alt="butterfly" style="zoom:50%;" />

### 300x300

<img src=".\Patch_Base_Resizing\Patch_Base_Resizing\result\300x300.png" alt="300x300" style="zoom:50%;" />

### 1200x600

<img src=".\Patch_Base_Resizing\Patch_Base_Resizing\result\1200x600.png" alt="1200x600" style="zoom:50%;" />
