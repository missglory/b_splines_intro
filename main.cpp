#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using std::cout;
using std::endl;
int w3, h3;
std::vector<cv::Point> controlPoints;
Mat orig;
const std::string name = "Display Image";
static float eps = 0.000001;

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

const int q = 3;
const int tn = 100;
const int n = 4;

std::vector<float> knots({0,0,0,0.5,1,1,1});

//cv::Mat N0(cv::Size(tn, n+q+1), CV_32F, cv::Scalar(0.0));
cv::Mat N0;

int dimsN[] = {tn, n, q};
std::vector<cv::Mat> N(q, cv::Mat(cv::Size(tn, n), CV_32F, cv::Scalar(0)));


int main(int argc, char** argv )
{
    Mat image;
    image = imread("../../girl.jpg", 1);
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
    N0 = cv::Mat::zeros(cv::Size(tn, n+q+1), CV_32F);
    for (int i = 0; i < N.size(); i++)
    {
        N[i] = cv::Mat::zeros(cv::Size(tn, n+q+1), CV_32F);
    }
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

    namedWindow("n0", cv::WINDOW_FREERATIO );
    namedWindow("n1", cv::WINDOW_FREERATIO );
    namedWindow("n2", cv::WINDOW_FREERATIO );

    for (int j = 0; j < n + q; j++)
    {
        cout << endl;
        for (int i = 0; i < tn; i++)
        {
            cout << N0.at<float>(j ,i) << " ";
        }
    }
    cout << endl;

    for (int i = q - 1; i < n; i++)
    {
        cv::line(N0, cv::Point(knots[i] * N0.cols, i), cv::Point(std::max(0.0, knots[i+1] - 0.0001) * N0.cols, i), cv::Scalar(1.0));
    }
    N[0] = N0;

    for (int j = 0; j < n + q; j++)
    {
        cout << endl;
        for (int i = 0; i < tn; i++)
        {
            cout << N0.at<float>(j, i) << " ";
        }
    }
    cout << endl;

    for (int dq = 1; dq < 3; dq++)
    {
        for (int j = q - dq - 1; j < n + dq - 1; j++)
        {
            for (int i = 0; i < tn; i++)
            {
                float t = (float)i / tn;
                float mul1 = N[dq - 1].at<float>(j, i);
                float mul2 = N[dq - 1].at<float>(j + 1, i);
                float del1 = knots[j + dq] - knots[j];
                float del2 = knots[j + dq + 1] - knots[j + 1];
                float dist1 = (t - knots[j]);
                float dist2 = (knots[j + dq + 1] - t);

                if (dist1 > del1 || dist2 > del2)
                {
//                    cout << "pzdc";
                }

                if (del1 > eps && mul1 > eps)
                {
                    N[dq].at<float>(j, i) = dist1 / del1 * mul1;
                }
                if (del2 > eps && mul2 > eps) {
                    N[dq].at<float>(j, i) += dist2 / del2 * mul2;
                }
            }
        }
    }

//    resizeWindow("n0", N0.cols, N0.rows);
//    resizeWindow("n0", N0.cols, N0.rows);
//    cv::resize(N0, N0, cv::Size(N0.cols,N0.rows * 20), 0, 0, INTER_NEAREST);
    imshow("n0", N0);
    imshow("n1", N[1]);
    imshow("n2", N[2]);
    waitKey(0);
    return 0;
}
