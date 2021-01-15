![alt text](http://zhejiang.cs.ua.edu/img/terrain.png)
## Introduction 
Spatial structured models are predictive models that capture dependency structure between samples based on their locations in the space. Learning such models plays an important role in many geoscience applications such as water surface mapping, but it also poses significant challenges due to implicit dependency structure in continuous space and high computational costs. Existing models often assume that the dependency structure is based on either spatial proximity or network topology, and thus cannot incorporate complex dependency structure such as contour and flow direction on a 3D potential surface. To fill the gap, this paper proposes a novel spatial structured model called hidden Markov contour tree (HMCT), which generalizes the traditional hidden Markov model from a total order sequence to a partial order polytree. HMCT also advances existing work on hidden Markov trees through capturing complex contour structures on a 3D surface. We propose efficient model construction and learning algorithms. Evaluations on real world hydrological datasets show that our HMCT outperforms multiple baseline methods in classification performance and that HMCT is scalable to large data sizes (e.g., classifying millions of samples in seconds).

## Compilation Code
  ```
  g++ -std=c++11 HMCT.cpp -O3 -o HMCT.exe
```
	
## Usage:

1. Create input data:
	a. Elevation Data (See: Setup.xlsx->"ElevationFileSample" worksheet for more details) Example: Area1Elevation.txt
	b. Feature Data   (See: Setup.xlsx->"FeatureFileSample" worksheet for more details)   Example: Area1Feature.txt
	c. Parameters     (See: Setup.xlsx->"ParameterSetup" worksheet for more details)      Example: Area1Parameter.txt 
	d. Training Data  (See: Setup.xlsx->"TrainFileSample" worksheet for more details)     Example: Area1Train.txt
	Save all input data in one input folder.

2. Create a folder("Results") to save the result of HMCT.  
	
3. Create Configuration File (See: Setup.xlsx->"ConfigurationSetup" worksheet for more details) Example: Area1Config.txt

4. Run 
	HMCT.exe Area1Config.txt

## Reference

Jiang, Zhe, and Arpan Man Sainju. "Hidden Markov Contour Tree: A Spatial Structured Model for Hydrological Applications." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2019.
