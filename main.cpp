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


//std::vector<cv::Point> R(cv::Mat& N, std::vector<cv::Point>& P)
//{
//    cv::Mat xs(cv::Size(P.size(), 1), cv::DataType<int>::type, cv::Scalar(0));
//    cv::Mat ys(cv::Size(P.size(), 1), cv::DataType<int>::type, cv::Scalar(0));
//    for (int i = 0; i < P.size(); i++)
//    {
//        cv::Point& p = P[i];
//        xs.at<int>(i, 0) = p.x;
//        ys.at<int>(i, 0) = p.y;
//    }

//}

const int q = 3;
const int tn = 100;
const int n = 4;

//std::vector<float> knots({0,0,0,0.5,1,1,1});
//std::vector<float> knots({0.0, 0.2, 0.4, 0.6, 0.8, 1.0});
std::vector<int> knots({0, 0, 0, 50, 100, 100, 100});
//std::vector<int> knots({0, 20, 40, 60, 80, 100});
int dimsN[] = {tn, n, q};
std::vector<cv::Mat> N(q, cv::Mat(cv::Size(tn, n), CV_32F, cv::Scalar(0)));

std::vector<std::vector<cv::Scalar> >colors(100, std::vector<cv::Scalar>(100));
//cv::Mat src_image;/// CV_8UC3
//cv::Mat img_float;
//src_image.converTo(img_float, CV_32FC3, 1 / 255.O);


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

            for (int i = 0; i < tn; i++)
            {
                for (int ii = 0; ii < tn; ii++)
                {
                    cv::Point linePoint(0,0);
                    for (int j = 0; j < n; j++)
                    {
                        for (int k = 0; k < n; k++)
                        {
                            linePoint += N[q - 1].at<float>(k, ii) * N[q - 1].at<float>(j, i) * controlPoints[j * 4 + k];
                        }
                    }
//                    cv::circle(image, linePoint, 1, cv::Scalar(i * 2, 250 - ii * 2, ii + i), 1);
                    cv::circle(image, linePoint, 1, colors[i][ii], 2);
                }
            }

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
    image = imread("../../girl.jpg", 1);
//    image = cv::Mat::zeros(cv::Size(600, 600), CV_8UC4);


    int w = image.cols;
    int h = image.rows;
    w /= 2;
    h /= 2;
    w3 = w/3;
    h3 = h/3;
    cv::resize(image, image, cv::Size(w,h));
    image.copyTo(orig);

    for (int i = 0; i < tn; i++)
    {
        for(int j = 0; j < tn; j++)
        {
            Vec3b clr = orig.at<cv::Vec3b>(h / 100.0 * i, w / 100.0 * j);
            colors[i][j] = cv::Scalar({(double)clr[0], (double)clr[1], (double)clr[2]});
        }
    }

    UserData ud;
    ud.image = image;
    ud.orig = orig;
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
//    namedWindow("n0", cv::WINDOW_FREERATIO );
//    namedWindow("n1", cv::WINDOW_FREERATIO );
//    namedWindow("n2", cv::WINDOW_FREERATIO );

    for (int i = 0; i < n + 1; i++)
    {
        if (knots[i+1] - knots[i] > 0)
        {
//            cv::line(N[0], cv::Point(knots[i] * N[0].cols, i), cv::Point(std::max((float)0.0, knots[i+1] - eps) * N[0].cols, i), cv::Scalar(1.0));
            int t = knots[i];
            while (t < knots[i+1])
            {
                N[0].at<float>(i, t++) = 1.0;
            }
        }
    }


    for (int dq = 1; dq < 3; dq++)
    {
        for (int j = 0; j < n + dq - 1; j++)
        {
            for (int i = 0; i < tn; i++)
            {
                float t = (float)i / tn;
                float mul1 = N[dq - 1].at<float>(j, i);
                float mul2 = N[dq - 1].at<float>(j + 1, i);
                float del1 = (float)(knots[j + dq] - knots[j]) / tn;
                float del2 = (float)(knots[j + dq + 1] - knots[j + 1]) / tn;
                float dist1 = (t - (float)knots[j]/tn);
                float dist2 = ((float)(knots[j + dq + 1])/tn - t);
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
//    imshow("n0", N[0]);
//    imshow("n1", N[1]);
//    imshow("n2", N[2]);
    waitKey(0);
    return 0;
}
