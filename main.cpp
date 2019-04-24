#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using std::cout;
using std::endl;
std::vector<cv::Point2f> controlPoints;
Mat orig;
const std::string name = "Display Image";
static float eps = 0.0001;

struct UserData {
    Mat orig;
    Mat image;
};

const int q = 3;
const int tn = 100;
const int n = 15;

//std::vector<float> knots({0,0,0,0.5,1,1,1});
//std::vector<float> knots({0.0, 0.2, 0.4, 0.6, 0.8, 1.0});
std::vector<float> knots({0.0, 1.0 / 18, 2.0 / 18, 3.0 / 18, 4.0 / 18, 5.0 / 18,
                         6.0 / 18, 7.0/ 18, 8.0 / 18, 9.0 / 18, 10.0 / 18, 11.0 / 18,
                         12.0/18, 13./18, 14./18, 15./18, 16./18, 17./18, 1.});
const int n2 = 2;
std::vector<float> knots2({0., 0., 0., 1., 1., 1.});
//std::vector<int> knots({0, 0, 0, 50, 100, 100, 100});
//std::vector<int> knots({0, 20, 40, 60, 80, 100});

int dimsN[] = {tn, n, q};
std::vector<cv::Mat> N;

std::vector<cv::Mat> initBasisFunc(std::vector<float>& knots, int n, int tn, bool circle = 0)
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

    if (circle)
    {
        int gi = 0;
        for (int i = 0; i < tn; i++)
        {
            if (N[q - 1].at<float>(1, i) > N[q - 1].at<float>(1, i + 1))
            {
                gi = i;
                break;
            }
        }
        //    int ei = tn - 1;
        //    for (int i = tn - 1; i > -1; i--)
        //    {
        //        if (N[q - 1].at<float>(n - 1, i) < N[q - 1].at<float>(n - 1, i + 1))
        //        {
        //            ei = i;
        //            break;
        //        }
        //    }

        namedWindow("roil", cv::WINDOW_FREERATIO );
        namedWindow("roir", cv::WINDOW_FREERATIO );
        cv::Rect roirectR(0, 0, gi /*+ 1*/, n + 1);
        //    cv::Rect roirectL(ei + 1, 0,
        //                     tn - ei, n + q - 1);
        cv::Rect roirectL(tn - gi, 0, gi, n + 1);
        cv::Mat roiR = N[q - 1](roirectR);
        cv::Mat roiL = N[q - 1](roirectL);
        cv::Mat roiCp;
        roiR.copyTo(roiCp);
        roiR += roiL;
        roiL += roiCp;
        imshow("roir", roiR);
        imshow("roil", roiL);
    }

    cv::normalize(N[q - 1], N[q - 1], 0., 1., NORM_MINMAX, CV_32F);

    return N;
}


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    static int activeCP = -1;
    static cv::Point2f startP(0,0);
    cv::Point2f currentP(x,y);
    UserData* ud = (UserData*) userdata;
    Mat& image = ud->image;
    float w = image.cols;
    float h = image.rows;
    int px = x / (w / 3);
    int py = y / (h / 3);
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
                startP = {(float)x,(float)y};
//                cout << "active " << activeCP << endl;
            }
        }
    }
    if ( event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_LBUTTON)
    {
//        cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
        //      cout << "Part(" << px << ", " << py << ")" << endl;
        if (activeCP > -1)
        {
            controlPoints[activeCP] = currentP;

            orig.copyTo(image);

            for (int i = 0; i < tn ; i++)
            {
                float norm = 0;
                for (int j = 0; j < n; j++)
                {
                    norm += N[q - 1].at<float>(j, i);
                }
                cv::Point2f linePoint(0,0);
                if(norm < 1)
                {
                    linePoint += cv::Point2f(w/2*(1 - norm), h/2*(1-norm));
                }
                for (int j = 0; j < n; j++)
                {
                    float mul = N[q - 1].at<float>(j, i);
                    linePoint += N[q - 1].at<float>(j, i) * controlPoints[j];
                }
                cv::circle(image, linePoint, 1, cv::Scalar(10, 250 - 2 * i, 10 + 2 * i), 1);
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
    float w = image.cols;
    float h = image.rows;
    w /= 2;
    h /= 2;
    cv::resize(image, image, cv::Size(w,h));
    controlPoints.resize(n);
    cv::Point2f center(w/2, h/2);
    for (int i = 0; i < controlPoints.size(); i++)
    {
        float amp = 100;
        float angle = (float)i / controlPoints.size() * 2 * 3.14;
        cv::Point2f vec(amp * cos(angle), amp * sin(angle));
//        cv::circle(image, vec + center, 5, cv::Scalar(10, 10,250), 10);
        controlPoints[i] = vec + center;
    }

    image.copyTo(orig);
    UserData ud;
    ud.image = image;
    ud.orig = orig;
    for (int i = 0; i < N.size(); i++)
    {
        N[i] = cv::Mat::zeros(cv::Size(tn, n+q+1), CV_32F);
    }
//    controlPoints = std::vector<cv::Point>({{0,h/3},     {w/3,h/3},  {2*w/3,h/3},    {w,h/3},
//                                           {0,2*h/3},   {w/3,2*h/3},{2*w/3,2*h/3},  {w,2*h/3},
//                                           {0,h},       {w/3,h},    {2*w/3,h},      {w,h}});
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

    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    cout.precision(3);


    N = initBasisFunc(knots, n, tn, 1);

    std::vector<cv::Mat> Nn = initBasisFunc(knots2, 3, tn);

    for (int j = 0; j < n2 + q; j++)
    {
        cout << "[";
        for (int i = 0; i < tn; i++)
            cout << Nn[2].at<float>(j,i) << ", ";
        cout << "], "<< endl;
    }

    imshow("n0", N[0]);
    imshow("n1", N[1]);
    imshow("n2", N[2]);
    setMouseCallback(name, CallBackFunc, &ud);
    waitKey(0);
    return 0;
}
