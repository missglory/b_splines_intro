#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using std::cout;
using std::endl;
int w3, h3;
std::vector<cv::Point2f> controlPoints;
//const std::vector<cv::Point> originalPoints;

Mat orig;
const std::string name = "Display Image";
static float eps = 0.0001;

struct UserData {
    Mat orig;
    Mat image;
    cv::Mat Nx;
    cv::Mat Ny;
//    UserData(Mat& im, Mat& o)
//    {
//    }
};

const int q = 3;
const int tn = 100;
const int n = 4;

//std::vector<float> knots({0,0,0,0.5,1,1,1});
//std::vector<float> knots({0.0, 0.2, 0.4, 0.6, 0.8, 1.0});
//std::vector<int> knots({0, 0, 0, 50, 100, 100, 100});

std::vector<std::vector<cv::Scalar> >colors(100, std::vector<cv::Scalar>(100));
//cv::Mat src_image;/// CV_8UC3
//cv::Mat img_float;
//src_image.converTo(img_float, CV_32FC3, 1 / 255.O);


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    static int activeCP = -1;
    static cv::Point2f startP(0,0);
    cv::Point2f currentP(x,y);
    UserData* ud = (UserData*) userdata;
    Mat& image = ud->image;
    Mat& orig = ud->orig;
    float w = image.cols;
    float h = image.rows;
    Mat& Nx = ud->Nx;
    Mat& Ny = ud->Ny;
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
                startP = {(float)x, (float)y};
            }
        }
    }
    if ( event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_LBUTTON)
    {
        cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
        if (activeCP)
        {
            controlPoints[activeCP] = currentP;

            orig.copyTo(image);

            for (int i = 0; i < h; i++)
            {
                for (int ii = 0; ii < w; ii++)
                {
                    cv::Point2f linePoint(0,0);
                    for (int j = 0; j < n; j++)
                    {
                        for (int k = 0; k < n; k++)
                        {
                            linePoint += Nx.at<float>(k, ii) * Ny.at<float>(j, i) * controlPoints[j * 4 + k];
                        }
                    }
                    cv::Point2f coordsPoint(ii, i);
                    cv::Point2f delta = 2 * coordsPoint - linePoint;
                    cv::Scalar color = orig.at<cv::Vec3b>(delta.y, delta.x);
//                    cv::circle(image, coordsPoint, 1, color, 2);
                    image.at<Vec3b>(i, ii) = orig.at<cv::Vec3b>(delta.y, delta.x);
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



std::vector<cv::Mat> initBasisFunc(std::vector<float>& knots, int n, int tn)
{
    std::vector<cv::Mat>N (q, cv::Mat::zeros(cv::Size(tn, n + q), CV_32F));
    float t = (knots[0] + eps) * tn;
    for (int i = 0; i < knots.size() - 1; i++)
    {
        if (knots[i+1] - knots[i] > 0)
        {
            float r = knots[i+1] * tn;
            while (t < knots[i+1] * tn)
            {
                N[0].at<float>(i, t++) = 1.0;
            }
        }
    }

    for (int dq = 1; dq < 3; dq++)
    {
        for (int j = 0; j < n + q - dq; j++)
        {
            for (int i = 0; i < tn; i++)
            {
                float t = (float)i / tn;
                float mul1 = N[dq - 1].at<float>(j, i);
                float mul2 = N[dq - 1].at<float>(j + 1, i);
                float del1 = knots[j + dq] - knots[j];
                float del2 = knots[j + dq + 1] - knots[j + 1];
                float dist1 = t - knots[j];
                float dist2 = (knots[j + dq + 1] - t);
                float res = 0;
                if (del1 > eps && mul1 > eps)
                {
                    res = dist1 / del1 * mul1;
                }
                if (del2 > eps && mul2 > eps) {
                    res += dist2 / del2 * mul2;
                }
                N[dq].at<float>(j, i) = res;
            }
        }
    }

    cv::normalize(N[q - 1], N[q - 1], 0., 1., NORM_MINMAX, CV_32F);

    return N;
}


int main(int argc, char** argv )
{
    Mat image;
    image = imread("../../girl.jpg", 1);
//    image = cv::Mat::zeros(cv::Size(600, 600), CV_8UC4);

    float w = image.cols;
    float h = image.rows;
    w /= 2;
    h /= 2;
    cv::resize(image, image, cv::Size(w,h));
    image.copyTo(orig);


    cv::Mat p(cv::Size(2, 4), CV_32F, cv::Scalar(0));
    cv::Mat nn(cv::Size(4, 5), CV_32F, cv::Scalar(0));

//    cv::Mat res = nn * p;




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
    controlPoints = std::vector<cv::Point2f>({{0.,0.},      {w/3,0.},    {2*w/3,0.},      {w,0.},
                                           {0.,h/3},     {w/3,h/3},  {2*w/3,h/3},    {w,h/3},
                                           {0.,2*h/3},   {w/3,2*h/3},{2*w/3,2*h/3},  {w,2*h/3},
                                           {0.,h},       {w/3,h},    {2*w/3,h},      {w,h}});
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
    imshow(name, image);
    namedWindow("n0", cv::WINDOW_FREERATIO );
    namedWindow("n1", cv::WINDOW_FREERATIO );
    namedWindow("n2", cv::WINDOW_FREERATIO );

    std::vector<float> localKnots({0,0,0,0.5, 1,1,1});
    std::vector<cv::Mat>Nx = initBasisFunc(localKnots, n, w);
    std::vector<cv::Mat>Ny = initBasisFunc(localKnots, n, h);
    imshow("n0", Ny[0]);
    imshow("n1", Ny[1]);
    imshow("n2", Ny[2]);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    cout.precision(3);
    for (int j = 0; j < n + q + 1; j++)
    {
        cout << "[";
        for (int i = 0; i < h; i++)
            cout << Ny[2].at<float>(j,i) << ", ";
        cout << "], "<< endl;
    }

    ud.Nx = Nx[Nx.size() - 1];
    ud.Ny = Ny[Ny.size() - 1];
    setMouseCallback(name, CallBackFunc, &ud);

    waitKey(0);
    return 0;
}
