#include<opencv2\highgui\highgui.hpp>
#include<opencv2\core\core.hpp>
#include<imgproc\imgproc.hpp>
#include<cv.h>
#include "opencv2/ml/ml.hpp"
#include<iostream>
#include <stdlib.h>
#include "opencv2/opencv.hpp"

#include <string.h>
#include <fstream>

using namespace std;
using namespace cv;


#define TRAINING_SAMPLES 57       //Number of samples in training dataset
#define ATTRIBUTES 256  //Number of pixels per sample.16X16
#define TEST_SAMPLES 18       //Number of samples in test dataset
#define CLASSES 3                  //Number of distinct labels.


void read_dataset(char *filename, cv::Mat &data, cv::Mat &classes,  int total_samples)
{

    int label;
    float pixelvalue;
    //open the file
    FILE* inputfile = fopen( filename, "r" );
 if(!inputfile) return;
    //read each row of the csv file
   for(int row = 0; row < total_samples; row++)
   {
       //for each attribute in the row
     for(int col = 0; col <=ATTRIBUTES; col++)
        {
            //if its the pixel value.
            if (col < ATTRIBUTES){

                fscanf(inputfile, "%f,", &pixelvalue);
			//	cout<<pixelvalue<<" ";
                data.at<float>(row,col) = pixelvalue;

            }//if its the label
            else if (col == ATTRIBUTES){
                //make the value of label column in that row as 1.
                fscanf(inputfile, "%i", &label);
                classes.at<float>(row,label) = 1.0;

            }
        }
    }

    fclose(inputfile);

}

/******************************************************************************/




void scaleDownImage(cv::Mat &originalImg,cv::Mat &scaledDownImage )
{
    for(int x=0;x<16;x++)
    {
        for(int y=0;y<16 ;y++)
        {
            int yd =ceil((float)(y*originalImg.cols/16));
            int xd = ceil((float)(x*originalImg.rows/16));
            scaledDownImage.at<uchar>(x,y) = originalImg.at<uchar>(xd,yd);

        }
    }
}


/**
 * Function to check if the color of the given image
 * is the same as the given color
 *
 * Parameters:
 *   edge        The source image
 *   color   The color to check
 */
bool is_border(cv::Mat& edge, cv::Vec3b color)
{
    cv::Mat im = edge.clone().reshape(0,1);

    bool res = true;
    for (int i = 0; i < im.cols; ++i)
        res &= (color == im.at<cv::Vec3b>(0,i));

    return res;
}

/**
 * Function to auto-cropping image
 *
 * Parameters:
 *   src   The source image
 *   dst   The destination image
 */
void autocrop(cv::Mat& src, cv::Mat& dst)
{
    cv::Rect win(0, 0, src.cols, src.rows);

    std::vector<cv::Rect> edges;
    edges.push_back(cv::Rect(0, 0, src.cols, 1));
    edges.push_back(cv::Rect(src.cols-2, 0, 1, src.rows));
    edges.push_back(cv::Rect(0, src.rows-2, src.cols, 1));
    edges.push_back(cv::Rect(0, 0, 1, src.rows));

    cv::Mat edge;
    int nborder = 0;
    cv::Vec3b color = src.at<cv::Vec3b>(0,0);

    for (int i = 0; i < edges.size(); ++i)
    {
        edge = src(edges[i]);
        nborder += is_border(edge, color);
    }

    if (nborder < 4)
    {
        src.copyTo(dst);
        return;
    }

    bool next;

    do {
        edge = src(cv::Rect(win.x, win.height-2, win.width, 1));
        if (next = is_border(edge, color))
            win.height--;
    }
    while (next && win.height > 0);

    do {
        edge = src(cv::Rect(win.width-2, win.y, 1, win.height));
        if (next = is_border(edge, color))
            win.width--;
    }
    while (next && win.width > 0);

    do {
        edge = src(cv::Rect(win.x, win.y, win.width, 1));
        if (next = is_border(edge, color))
            win.y++, win.height--;
    }
    while (next && win.y <= src.rows);

    do {
        edge = src(cv::Rect(win.x, win.y, 1, win.height));
        if (next = is_border(edge, color))
            win.x++, win.width--;
    }
    while (next && win.x <= src.cols);

    dst = src(win);
}



void convertToPixelValueArray(cv::Mat &img,int pixelarray[])
{
    int i =0;
    for(int x=0;x<16;x++)
    {
        for(int y=0;y<16;y++)
        {
            pixelarray[i]=(img.at<uchar>(x,y)==255)?1:0;
            i++;

        }

    }
}
string convertInt(int number)
{
    stringstream ss;//create a stringstream
    ss << number;//add number to the stream
    return ss.str();//return a string with the contents of the stream
}

void readFile(std::string datasetPath,int samplesPerClass,std::string outputfile )
{
    fstream file(outputfile,ios::out);
       for(int digit=0;digit<samplesPerClass;digit++)
        {   //creating the file path string
            std::string imagePath = datasetPath+"triangle\\img("+convertInt(digit)+").png";
            //reading the image
            cv::Mat img = cv::imread(imagePath,0);
            cv::Mat output = img;

            //declaring mat to hold the scaled down image
            cv::Mat scaledDownImage(16,16,CV_8U,cv::Scalar(0));
            //declaring array to hold the pixel values in the memory before it written into file
            int pixelValueArray[256];

            //cropping the image.

             autocrop(output, output);
            //reducing the image dimension to 16X16
          //  scaleDownImage(output,scaledDownImage);
 cv::resize(output, scaledDownImage, scaledDownImage.size());
            //reading the pixel values.
            convertToPixelValueArray(scaledDownImage,pixelValueArray);
            //writing pixel data to file
            for(int d=0;d<256;d++){
                file<<pixelValueArray[d]<<",";
            }
            //writing the label to file
            file<<0<<"\n";
	   }

			for(int digit=0;digit<samplesPerClass;digit++)
        {   //creating the file path string
            std::string imagePath = datasetPath+"rectangle\\img("+convertInt(digit)+").png";
            //reading the image
            cv::Mat img = cv::imread(imagePath,0);
            cv::Mat output = img;

            //declaring mat to hold the scaled down image
            cv::Mat scaledDownImage(16,16,CV_8U,cv::Scalar(0));
            //declaring array to hold the pixel values in the memory before it written into file
            int pixelValueArray[256];
            autocrop(output, output);

 cv::resize(output, scaledDownImage, scaledDownImage.size());
            //reading the pixel values.
            convertToPixelValueArray(scaledDownImage,pixelValueArray);
            //writing pixel data to file
            for(int d=0;d<256;d++){
                file<<pixelValueArray[d]<<",";
            }
            //writing the label to file
            file<<1<<"\n";
			}

			for(int digit=0;digit<samplesPerClass;digit++)
        {   //creating the file path string
            std::string imagePath = datasetPath+"circle\\img("+convertInt(digit)+").png";
            //reading the image
            cv::Mat img = cv::imread(imagePath,0);
            cv::Mat output = img;
            //Applying gaussian blur to remove any noise
				 Mat element = getStructuringElement( 1,Size( 31, 31 ), Point( 10, 10) );


            //declaring mat to hold the scaled down image
            cv::Mat scaledDownImage(16,16,CV_8U,cv::Scalar(0));
            //declaring array to hold the pixel values in the memory before it written into file
            int pixelValueArray[256];

            //cropping the image.
			autocrop(output, output);
            //cropImage(output,output);

            //reducing the image dimension to 16X16
            //scaleDownImage(output,scaledDownImage);
 cv::resize(output, scaledDownImage, scaledDownImage.size());
            //reading the pixel values.
            convertToPixelValueArray(scaledDownImage,pixelValueArray);
            //writing pixel data to file
            for(int d=0;d<256;d++){
                file<<pixelValueArray[d]<<",";
            }
            //writing the label to file
            file<<2<<"\n";


    }
    file.close();
}

void feature_set()
{
	 cout<<"Reading the training set......\n";
    readFile("C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\dataset\\",19,"C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\dataset\\inputdata.txt");
    cout<<"Reading the test set.........\n";
    readFile("C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\testset\\",6,"C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\dataset\\testdata.txt");
    cout<<"operation completed";
}


void train_data_set()
{

    //matrix to hold the training sample
    cv::Mat training_set(TRAINING_SAMPLES,ATTRIBUTES,CV_32F);
    //matrix to hold the labels of each taining sample
    cv::Mat training_set_classifications(TRAINING_SAMPLES, CLASSES, CV_32F);
    //matric to hold the test samples
    cv::Mat test_set(TEST_SAMPLES,ATTRIBUTES,CV_32F);
    //matrix to hold the test labels.
    cv::Mat test_set_classifications(TEST_SAMPLES,CLASSES,CV_32F);

    //
    cv::Mat classificationResult(1, CLASSES, CV_32F);
    //load the training and test data sets.
    read_dataset("C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\dataset\\inputdata.txt", training_set, training_set_classifications, TRAINING_SAMPLES);
    read_dataset("C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\dataset\\testdata.txt", test_set, test_set_classifications, TEST_SAMPLES);

        // define the structure for the neural network (MLP)
        // The neural network has 3 layers.
        // - one input node per attribute in a sample so 256 input nodes
        // - 16 hidden nodes
        // - 3 output node, one for each class.

        cv::Mat layers(3,1,CV_32S);
        layers.at<int>(0,0) = ATTRIBUTES;//input layer
        layers.at<int>(1,0)=16;//hidden layer
        layers.at<int>(2,0) =CLASSES;//output layer

        //create the neural network.
        //for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
        CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,0.6,1);

        CvANN_MLP_TrainParams params(

                                        // terminate the training after either 1000
                                        // iterations or a very small change in the
                                        // network wieghts below the specified value
                                        cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),
                                        // use backpropogation for training
                                        CvANN_MLP_TrainParams::BACKPROP,
                                        // co-efficents for backpropogation training
                                        // recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
                                        0.1,
                                        0.1);

        // train the neural network (using training data)

        printf( "\nUsing training dataset\n");
        int iterations = nnetwork.train(training_set, training_set_classifications,cv::Mat(),cv::Mat(),params);
        printf( "Training iterations: %i\n\n", iterations);

        // Save the model generated into an xml file.
        CvFileStorage* storage = cvOpenFileStorage( "C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\dataset\\param.xml", 0, CV_STORAGE_WRITE );
        nnetwork.write(storage,"shaperecognize");
        cvReleaseFileStorage(&storage);

        // Test the generated model with the test samples.
        cv::Mat test_sample;
        //count of correct classifications
        int correct_class = 0;
        //count of wrong classifications
        int wrong_class = 0;

        //classification matrix gives the count of classes to which the samples were classified.
        int classification_matrix[CLASSES][CLASSES]={{}};

        // for each sample in the test set.
        for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {

            // extract the sample

            test_sample = test_set.row(tsample);

            //try to predict its class

            nnetwork.predict(test_sample, classificationResult);
            /*The classification result matrix holds weightage  of each class.
            we take the class with the highest weightage as the resultant class */

            // find the class with maximum weightage.
            int maxIndex = 0;
            float value=0.0f;
            float maxValue=classificationResult.at<float>(0,0);
            for(int index=0;index<CLASSES;index++)
            {   value = classificationResult.at<float>(0,index);
				//cout<<value<<" ";
                if(value>maxValue)
                {   maxValue = value;
                    maxIndex=index;

                }
            }
            printf("Testing Sample %i -> class result (shape %d)\n", tsample, maxIndex);

            //Now compare the predicted class to the actural class. if the prediction is correct then\
            //test_set_classifications[tsample][ maxIndex] should be 1.
            //if the classification is wrong, note that.
            if (test_set_classifications.at<float>(tsample, maxIndex)!=1.0f)
            {
                // if they differ more than floating point error => wrong class

                wrong_class++;

                //find the actual label 'class_index'
                for(int class_index=0;class_index<CLASSES;class_index++)
                {
                    if(test_set_classifications.at<float>(tsample, class_index)==1.0f)
                    {

                        classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
                        break;
                    }
                }

            } else {

                // otherwise correct

                correct_class++;
                classification_matrix[maxIndex][maxIndex]++;
            }
        }

        printf( "\nResults on the testing dataset\n"
        "\tCorrect classification: %d (%g%%)\n"
        "\tWrong classifications: %d (%g%%)\n",
        correct_class, (double) correct_class*100/TEST_SAMPLES,
        wrong_class, (double) wrong_class*100/TEST_SAMPLES);
        cout<<"   ";
        for (int i = 0; i < CLASSES; i++)
        {
            cout<< i<<"\t";
        }
        cout<<"\n";
        for(int row=0;row<CLASSES;row++)
        {cout<<row<<"  ";
            for(int col=0;col<CLASSES;col++)
            {
                cout<<classification_matrix[row][col]<<"\t";
            }
            cout<<"\n";
        }


}
int predict_class(Mat image)
{

	 CvANN_MLP nnetwork;
    CvFileStorage* storage = cvOpenFileStorage( "C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\dataset\\param.xml", 0, CV_STORAGE_READ );
    CvFileNode *n = cvGetFileNodeByName(storage,0,"shaperecognize");
    nnetwork.read(storage,n);
    cvReleaseFileStorage(&storage);



    //your code here
    // ...Generate cv::Mat data(1,ATTRIBUTES,CV_32S) which will contain the pixel
    // ... data for the digit to be recognized
    // ...
 cv::Mat data(1,ATTRIBUTES,CV_32F);
	 cv::Mat img = image;
            cv::Mat output = img;


            //declaring mat to hold the scaled down image
            cv::Mat scaledDownImage(16,16,CV_8U,cv::Scalar(0));


            //declaring array to hold the pixel values in the memory before it written into file
            int pixelValueArray[256];

            //cropping the image.
           autocrop(output, output);


            //reducing the image dimension to 16X16
 			cv::resize(output, scaledDownImage, scaledDownImage.size());


            //reading the pixel values.
           convertToPixelValueArray(scaledDownImage,pixelValueArray);
            //writing pixel data to file
            for(int d=0;d<256;d++){
                data.at<float>(0,d) = pixelValueArray[d];
            }




    int maxIndex = 0;

    cv::Mat classOut(1,CLASSES,CV_32F);
    //prediction
    nnetwork.predict(data, classOut);
    float value;
    float maxValue=classOut.at<float>(0,0);
    for(int index=0;index<CLASSES;index++)
    {   value = classOut.at<float>(0,index);
	cout<<value<<" ";
            if(value>maxValue)
            {   maxValue = value;
                maxIndex=index;
            }
    }
    return maxIndex;

}

Mat_<Vec3b> output(400,400, Vec3b(255,255,255));

	Point P;
	Point oldP;
int main()

{ static int digit = 0;

	VideoCapture cap(0);
    if(!cap.isOpened())
	return -1;

	Mat_<Vec3b> img1(400,400, Vec3b(0,180,160));
	Mat_<Vec3b> img2(400,400, Vec3b(100,255,255));
	imshow("LB",img1);
	imshow("UB",img2);
	feature_set();
	train_data_set();

	 CvANN_MLP nnetwork;
    CvFileStorage* storage = cvOpenFileStorage( "C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\dataset\\param.xml", 0, CV_STORAGE_READ );
    CvFileNode *n = cvGetFileNodeByName(storage,0,"shaperecognize");
    nnetwork.read(storage,n);
    cvReleaseFileStorage(&storage);

	while(1)
{
    Mat image,img;
    cap >> img;
	resize(img,image,Size(400,400));
	Mat dst;

	inRange(image,img1,img2,dst);



	Mat dil,can;
//    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
	 Mat element = getStructuringElement( 1,Size( 31, 31 ), Point( 10, 10) );
	//inRange(image,hsv1,hsv2,dst);
    dilate( dst,dil, element);


	static const int thickness = 1;//CV_FILLED - filled contour
static const int lineType = 8;//8:8-connected,  4:4-connected line, CV_AA: anti-aliased line.
Scalar           color = CV_RGB(255, 20, 20); // line color - light red

//Segmented image
Mat Segmented = dil > 128;

////////////////////////////////////////////////////////////////////
/// Find contours - use old style C since C++ version has bugs.

//Target image for sketching contours of segmentation
Mat             drawing;
cvtColor(dil ,   drawing, CV_GRAY2RGB);
IplImage        drawingIpl = drawing; //just clone header with no copying of data

//Data containers for FindContour
IplImage        SegmentedIpl = Segmented;//just clone header with no copying of data
CvMemStorage*   storage = cvCreateMemStorage(0);
CvSeq*          contours = 0;
int             numCont = 0;
int             contAthresh = 45;

numCont = cvFindContours(&SegmentedIpl, storage, &contours, sizeof(CvContour),
    CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

int k =waitKey(27) ;

if(k==(int)('c'))
{
for(int y=0;y<output.rows;y++)
{
    for(int x=0;x<output.cols;x++)
    {
        // get pixel

		output.at<Vec3b>(Point(x,y))[0] = 255;
		output.at<Vec3b>(Point(x,y))[1] = 255;
		output.at<Vec3b>(Point(x,y))[2] = 255;


    }
}
}
for (; contours != 0; contours = contours->h_next)
{
     CvRect rect = cvBoundingRect(contours, 0); //extract bounding box for current contour

                     //drawing rectangle
                     cvRectangle(&drawingIpl,
                                  cvPoint(rect.x, rect.y),
                                  cvPoint(rect.x+rect.width, rect.y+rect.height),
                                  cvScalar(0, 0, 255, 0),
                                  2, 8, 0);
					 P.x =  cvPoint(rect.x+rect.width/2, rect.y+rect.height/2).x;
					 P.y =  cvPoint(rect.x+rect.width/2, rect.y+rect.height/2).y;

					/*
					 output.at<Vec3b>(P.y,400-P.x) = 0;

					 output.at<Vec3b>(P.y+1,400-P.x) = 0;
					 output.at<Vec3b>(P.y-1,400-P.x) = 0;
					 output.at<Vec3b>(P.y,400-P.x+1) = 0;
					 output.at<Vec3b>(P.y,400-P.x-1) = 0;
					 output.at<Vec3b>(P.y+1,400-P.x+1) = 0;
					 output.at<Vec3b>(P.y-1,400-P.x-1) = 0;
					 output.at<Vec3b>(P.y+1,400-P.x-1) = 0;
					 output.at<Vec3b>(P.y-1,400-P.x+1) = 0;*/
					 P.x = 400-P.x;

					  line(output, P, oldP, 'r', 2, 8,0);
					 oldP.x = P.x;
					 oldP.y = P.y;


	//cvDrawContours(&drawingIpl, contours, color, color, -1, thickness, lineType, cvPoint(0, 0));
}

/// Show in a window
string win_name = "Contour";
//Mat drawing = cvarrToMat(&drawingIpl); //Mat(&IplImage) is soon to be deprecated OpenCV 3.X.X
namedWindow(win_name, CV_WINDOW_NORMAL);
imshow(win_name, drawing);
	imshow("x   ",image);


	imshow("g",drawing);
	imshow("output",output);





if(k== ((int)('a')))
{



	//cout<<k<<endl;
//imwrite( "C:\\Users\\Piyush\\Documents\\Visual Studio 2012\\Projects\\gestureTyping\\gestureTyping\\testset\\triangle\\img("+convertInt(digit)+").png", output );
//digit++;



int h= predict_class(output);
if(h==0) cout<<"triangle"<<endl;
if(h==1) cout<<"rectangle"<<endl;
if(h==2) cout<<"circle"<<endl;
}
	//cout<<numCont<<endl;
waitKey(10);

   if(k == 27) break;
}
	return 0;

	//feature_set();
	//train_data_set();
	//getchar();
    //return 0;
}
