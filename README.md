###### WELCOME TO TEAM HASHIRA PYTHON CODE ######

There are 2 folder. 
The images folder has the input and output obtained by running the code. The inputs include composite, trimap and ground truth. 

The code folder has all the core functions and the main function.
The files are:
1. fWindow
2. fProcess
3. fGaussian
4. fMeanCovariance
5. fFBALikelihood
6. Matte
7. UnitTestCase

There is also 3 sample input images that have been called in the code.

****Code Execution****

The main function takes 3 inputs.
Using the trimap, the fixed foreground, fixed unknown and unknown regions are extracted. The position of unknown pixels is obtained. An alpha matrix is defined which has values 0 and 1 for background and foreground respectively. The initial parameters is also mentioned. All this is passed as an input for the fProcess function.

The fProcess function calls all the main mathematical functions. It iterates over all unknown pixels. For all pixels, a window is generated around the pixel usinf the fWindow function. The fWindow function generates windows based on the minimum number of fixed pixels counted. Depending on the size of the window, and the position of the pixel, a Gaussian window is generated and clipped using the fGaussian function. THe weight matrices are generated for foreground and background. Using the window and the weights, the mean and covariance of foreground and background is obtained. 

With the mean and covariance, the next function fFBALikelihood is called. This function performs 3 tasks. It calculates F, B and alpha for every iteration. It also governs over the stopping of the code. The function returns a single dimensional alpha matte. This is converted to 3 channels and the compared with the ground truth.
