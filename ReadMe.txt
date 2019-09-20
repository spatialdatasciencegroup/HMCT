Compilation Code:
	 g++ -std=c++11 HMCT.cpp -O3 -o HMCT.exe
	
Usage:
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