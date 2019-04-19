#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using std::cout;
using std::endl;
int w3, h3;
std::vector<cv::Point> controlPoints;
Mat orig;
const std::string name = "Display Image";


struct UserData {
    Mat orig;
    Mat image;
//    UserData(Mat& im, Mat& o)
//    {

//    }
};


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
//     if  ( event == EVENT_LBUTTONDOWN )
//     {
//          std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//     }
//     else if  ( event == EVENT_RBUTTONDOWN )
//     {
//          std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//     }
//     else if  ( event == EVENT_MBUTTONDOWN )
//     {
//          cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
//     }
    static int activeCP = -1;
    static cv::Point startP(0,0);
    cv::Point currentP(x,y);
    int px = x / w3;
    int py = y / h3;
    UserData* ud = (UserData*) userdata;
    Mat& image = ud->image;
    Mat& orig = ud->orig;
//    static cv::Point del(w3,h3);
    if ( event == EVENT_LBUTTONUP)
    {
        activeCP = -1;
    }
    if ( event == EVENT_LBUTTONDOWN)
    {
        for (int i = 0; i < controlPoints.size(); i++)
        {
            if (cv::norm(currentP - controlPoints[i]) < 10)
            {
                activeCP = i;
                startP = {x,y};
            }
        }
    }
    if ( event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_LBUTTON)
    {
      cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
//      cout << "Part(" << px << ", " << py << ")" << endl;
      if (activeCP)
      {
          controlPoints[activeCP] = currentP;

          orig.copyTo(image);
          for (auto point : controlPoints)
          {
              cv::circle(image, point, 5, cv::Scalar(10, 10,250), 10);
          }
          imshow(name, image);
      }
    }
}



int main(int argc, char** argv )
{
    Mat image;
    image = imread("../girl.jpg", 1);
    int w = image.cols;
    int h = image.rows;
    w /= 2;
    h /= 2;
    w3 = w/3;
    h3 = h/3;
    cv::resize(image, image, cv::Size(w,h));
    image.copyTo(orig);
    UserData ud;
    ud.image = image;
    ud.orig = orig;
    controlPoints = std::vector<cv::Point>({{0,0},      {w/3,0},    {2*w/3,0},      {w,0},
                                           {0,h/3},     {w/3,h/3},  {2*w/3,h/3},    {w,h/3},
                                           {0,2*h/3},   {w/3,2*h/3},{2*w/3,2*h/3},  {w,2*h/3},
                                           {0,h},       {w/3,h},    {2*w/3,h},      {w,h}});
    if (image.empty())
    {
        printf("No image data \n");
        return -1;
    }
    for (auto point : controlPoints)
    {
        cv::circle(image, point, 5, cv::Scalar(10, 10, 250), 10);
    }
    namedWindow(name, WINDOW_AUTOSIZE );
    setMouseCallback(name, CallBackFunc, &ud);
    imshow(name, image);
    waitKey(0);
    return 0;
}
