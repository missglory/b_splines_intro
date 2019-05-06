#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

struct UserData
{
	cv::Mat orig;
	cv::Mat image;
	cv::Mat Nx;
	cv::Mat Ny;
	std::vector<cv::Point2f> controls;
	std::string window_name;
    std::vector<float> knots_x, knots_y;
    int tn, tn2, n;
    cv::Mat colors;
};

float func_y(float y)
{
//    return std::sqrt(y);
    return y;
}


float N(const std::vector<float> &knot, float t, int k, int q)
{
	if (q == 1) return (t >= knot[k] && t < knot[k + 1]) ? 1.f : 0.f;
	
	float div1 = knot[k + q - 1] - knot[k];
	float div2 = knot[k + q] - knot[k + 1];
	float val1 = (div1 != 0) ? (t - knot[k]) * N(knot, t, k, q - 1) / div1 : 0;
	float val2 = (div2 != 0) ? (knot[k + q] - t) * N(knot, t, k + 1, q - 1) / div2 : 0; 
	
	return val1 + val2;
}

cv::Vec3b BilinInterp(const cv::Mat &src, float x, float y)
{
	int x1 = (int)std::floor(x);
	int y1 = (int)std::floor(y);
	int x2 = (int)std::ceil(x);
	int y2 = (int)std::ceil(y);

	if (x1 < 0 || y1 < 0 || x2 >= src.cols || y2 >= src.rows) return cv::Vec3b();

	cv::Vec3f p1 = src.at<cv::Vec3b>(y1, x1);
	cv::Vec3f p2 = src.at<cv::Vec3b>(y1, x2);
	cv::Vec3f p3 = src.at<cv::Vec3b>(y2, x1);
	cv::Vec3f p4 = src.at<cv::Vec3b>(y2, x2);

	cv::Vec3f c1 = p1 + (p2 - p1) * (x - x1);
	cv::Vec3f c2 = p3 + (p4 - p3) * (x - x1);
	return c1 + (c2 - c1) * (y - y1);
}

cv::Point2f ComputePoint(const cv::Mat& Nx, const cv::Mat& Ny, const std::vector<cv::Point2f>& controls, int x, int y)
{
    cv::Point2f ret;
    float norm = 0;
    for (int i = 0; i < Nx.rows; i++)
    {
        norm += Nx.at<float>(i, x);
    }
//    ret += (1.f - norm) * cv::Point2f(startx, starty);
    if (norm < 0.9999)
    {
        return {0.f, 0.f};
    }
    for (int i = 0; i < Nx.rows; i++) {
		for (int j = 0; j < Ny.rows; j++) {
            ret += Nx.at<float>(i, x) * Ny.at<float>(j, y) * controls[Nx.rows * j + i];
		}
	}
	return ret;
}

cv::Point2f ComputePoint(const cv::Mat& Nx, const cv::Mat& Ny, const std::vector<cv::Point2f>& controls, cv::Point2f point)
{
    return ComputePoint(Nx, Ny, controls, point.x, point.y);
}

cv::Point2f ComputePoint(const cv::Mat& Nx, const std::vector<cv::Point2f>& controls, int x, int tn = 0)
{
    if (!tn) tn = Nx.rows;
    cv::Point2f ret;
    for (int i = 0; i < tn; i++) {
        ret += Nx.at<float>(i, x) * controls[i];
    }
    return ret;
}

float find_u(const cv::Point2f& v1, const cv::Point2f& center)
{
    float w = center.x;
    float h = center.y;
    float x = v1.x - center.x;
    float y = v1.y - center.y;
    if (x < 0)
        return .5 + std::atan(y / x) / CV_PI / 2;
    if (y < 0)
        return 1 + std::atan(y / x) / CV_PI / 2;
    return (std::atan(y / x)) / CV_PI / 2;
}

cv::Point2f findP(float u, float v, cv::Point2f p00, cv::Point2f p01, cv::Point2f p10, cv::Point2f p11)
{
    cv::Point2f dx = p00 + (p01 - p00) * u;
    cv::Point2f dx2 = p10 + (p11 - p10) * u;
    return cv::Point2f(dx + (dx2 - dx) * func_y(v));
}



cv::Mat _rendertestimage(cv::Mat& testimage)
{
    cv::Mat rendertestimage = cv::Mat::zeros(testimage.cols, testimage.rows, CV_32FC3);
    int from_to[] = { 0,0, 1,1, 2,2 };
    cv::mixChannels({{testimage}}, {{rendertestimage}}, from_to, testimage.channels());
    return rendertestimage;
}





void mouse_callback(int event, int x, int y, int flags, void* userdata)
{
	static int activeCP = -1;
	static cv::Point2f startP(0, 0);
	cv::Point2f currentP(x, y);
	UserData* ud = (UserData*)userdata;
	
	cv::Mat& image	= ud->image;
	cv::Mat& orig	= ud->orig;
	cv::Mat& Nx		= ud->Nx;
	cv::Mat& Ny		= ud->Ny;
    cv::Mat& colors = ud->colors;
    std::vector<float>& knots_x = ud->knots_x;
    std::vector<float>& knots_y = ud->knots_y;
	float w			= image.cols;
	float h			= image.rows;
	std::vector<cv::Point2f> &controls = ud->controls;
	std::string &name = ud->window_name;
    int tn = ud->tn;
    int tn2 = ud->tn2;
    int n = ud->n;

    cv::Mat indmask = cv::Mat::zeros(w, h, CV_8U);
    cv::Mat testmask = cv::Mat::zeros(w, h, CV_32FC2);
    cv::Mat rendertestmask = cv::Mat::zeros(w, h, CV_32FC3);

    static const cv::Point2f originalPivot(100, 100);
    static cv::Point2f pivot = originalPivot;
    static const int pivotId = 21486234;

	if (event == cv::EVENT_LBUTTONUP) {
		activeCP = -1;
	}
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (cv::norm(currentP - pivot) < 10)
        {
            activeCP = pivotId;
            startP = {(float)x,(float)y};
        }

        for (int xx = 0; xx < n - 3; xx++) {
            for (int yy = 0; yy < 2; yy++)
            {
                int i = xx + yy * n;
                if (cv::norm(currentP - controls[i]) < 3) {
                    activeCP = i;
                    startP = { (float)x, (float)y };
                }
            }
		}
	}
	if (event == cv::EVENT_MOUSEMOVE && flags == cv::EVENT_FLAG_LBUTTON) {
        if (activeCP >= -1)
        {
            if (activeCP == pivotId)
            {
                for (int i = 0; i < n; i++)
                {
                    controls[i] -= pivot - originalPivot;
                }
                pivot = currentP;
                for (int i = 0; i < n; i++)
                {
                    controls[i] += pivot - originalPivot;
                }
            }
            else
            {
                controls[activeCP] = currentP;
            }

			orig.copyTo(image);



            int ind = 1;
            cv::Point2f uv_coords[4];
            uv_coords[0] = {0,0};
            uv_coords[1] = {0,1};
            uv_coords[2] = {1,1};
            uv_coords[3] = {1,0};

            typedef struct{
                std::vector<cv::Point2f> points;
                cv::Mat transform;
                std::vector<cv::Point2f> quantize_params;
            } pointsTransform;
            pointsTransform pT[n - 2];
            for(int i = 1; i < n - 2; i++)
            {
                cv::Point2f xy_coords[4];
                cv::Point xy_coords_int[4];

                xy_coords[0] = controls[i];
                xy_coords[1] = controls[i+n];
                xy_coords[2] = controls[i+n+1];
                xy_coords[3] = controls[i+1];
                for(int i = 0; i < 4; i++)
                {
                    xy_coords_int[i] = xy_coords[i];
                }

                cv::fillConvexPoly(indmask, xy_coords_int, 4, cv::Scalar(ind++));
                pT[i].transform = cv::getPerspectiveTransform(xy_coords, uv_coords);

            }
            imshow("mask", indmask);
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    cv::Point2f cur_p(j,i);
                    int ind = indmask.at<unsigned char>(cur_p);
                    if (!ind)
                        continue;
                    pT[ind].points.push_back(cv::Point2f(cur_p));
                }
            }
            for(int i = 1; i < n - 2; i++)
            {
                pT[i].quantize_params.resize(pT[i].points.size());
                cv::perspectiveTransform(pT[i].points, pT[i].quantize_params, pT[i].transform);
                for (int j = 0; j < pT[i].points.size(); j++)
                {
                    float u = (pT[i].quantize_params[j].x + (i - 1)) / (n - 2);
                    float v = pT[i].quantize_params[j].y;
                    v = std::min(1.f, std::max(0.f, v));
                    testmask.at<cv::Vec2f>(pT[i].points[j]) = cv::Vec2f(u, pT[i].quantize_params[j].y);
//                    rendertestmask.at<cv::Vec3f>(pT[i].points[j]) = cv::Vec3f(u / 3, 0.f, v / 3);
                }
            }

            for(int i = 0; i < h; i++)
            {
                for(int j = 0; j < w; j++)
                {

                    cv::Point2f cur_p(j,i);
                    int ind = indmask.at<unsigned char>(cur_p);
                    if (!ind)
                        continue;

                    cv::Point2f curPoint(j, i);
                    cv::Vec2f u_v = testmask.at<cv::Vec2f>(curPoint);
                    u_v[0] = std::max(0.f, u_v[0]);
                    u_v[0] = std::min(1.f, u_v[0]);
                    u_v[1] = std::max(0.f, u_v[1]);
                    u_v[1] = std::min(1.f, u_v[1]);
                    u_v[0] *= tn;
                    u_v[1] *= tn2;
                    cv::Point2f new_coord = ComputePoint(Nx, Ny, controls, u_v);
                    image.at<cv::Vec3b>(curPoint) = BilinInterp(orig, new_coord.x, new_coord.y);
                }
            }

            rendertestmask = _rendertestimage(testmask);
            imshow("testmask", rendertestmask);

            for (int i = 0; i < n - 3; i++) {
                for (int j = 0; j < 2; j++)
                {
                    cv::circle(image, controls[j * n + i], 2, cv::Scalar(10, 10, 255), -1);
                }
            }
            cv::circle(image, pivot, 5, cv::Scalar(10, 250, 0), 10);
            cv::imshow(name, image);
		}
	}
}



int main()
{
	cv::Mat image = cv::imread("../girl.jpg");
	cv::Mat orig = image.clone();

	float w = (float)image.cols;
	float h = (float)image.rows;

	std::vector<cv::Point2f> controls =
	{ 
		{0.f,0.f},			{w / 3,0.f},		{2 * w / 3,0.f},		{w-1,0.f},
		{0.f,h / 3},		{w / 3,h / 3},		{2 * w / 3,h / 3},		{w-1,h / 3},
		{0.f,2 * h / 3},	{w / 3,2 * h / 3},	{2 * w / 3,2 * h / 3},  {w-1,2 * h / 3},
		{0.f,h-1},			{w / 3,h-1},		{2 * w / 3,h-1},		{w-1,h-1}
	};

    int n = 50, n2 = 2;
    int tn = 1000, tn2 = 200;
    controls.resize(n2 * n);
    cv::Point2f center(w/2, h/2);
    for (int i = 0; i < n; i++)
    {
        float amp = 140;
        float amp2 = 1.4;
        float angle = (float)(i % (n - 3)) / (n - 3) * 2 * 3.14;
        cv::Point2f vec(amp * cos(angle), amp * sin(angle));
        for (int j = 0; j < n2; j++)
        {
            controls[j * n + i] = vec + center;
            vec *= amp2;
        }
    }


    const int q = 4;
    std::vector<float> knots = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };
    std::vector<float> knots_y(knots.size());

    for (size_t i = 0; i < knots.size(); i++) {
        knots_y[i] = tn2 * knots[i];
    }

    int kx_size = n + q;
    knots.resize(kx_size + 1);
    for (int i = 0; i < kx_size + 1; i++)
    {
        knots[i] = (float)i / kx_size;
    }
    std::vector<float> knots_x(knots.size());
	for (size_t i = 0; i < knots.size(); i++) {
        knots_x[i] = tn * knots[i];
	}

    cv::Mat Nfunc_x(n, tn, CV_32FC1), Nfunc_y(n2, tn2, CV_32FC1);
    for (int i = 0; i < n; i++) {
        for (int t = 0; t < tn; t++) {
            Nfunc_x.at<float>(i, t) = N(knots_x, t, i, q);
		}
	}
//    for (int i = 0; i < n2; i++) {
//        for (int t = 0; t < tn2; t++) {
//            Nfunc_y.at<float>(i, t) = N(knots_y, t, i, 3);
//        }
//    }
    for (int t = 0; t < tn2; t++)
    {
        Nfunc_y.at<float>(1, t) = func_y((float)t / (tn2 - 1));
        Nfunc_y.at<float>(0, t) = 1 - Nfunc_y.at<float>(1, t);
    }

	for (auto &pt : controls) {
        cv::circle(image, pt, 2, cv::Scalar(10, 10, 250), -1);
	}


	std::string name = "Display Image";
	UserData ud;
	ud.image	= image;
	ud.orig		= orig;
	ud.Nx		= Nfunc_x;
    ud.Ny		= Nfunc_y;
	ud.controls = controls;
	ud.window_name = name;
    ud.knots_x = knots_x;
    ud.knots_y = knots_y;
    ud.tn = tn;
    ud.tn2 = tn2;
    ud.n = n;

//    cv::namedWindow("n", cv::WINDOW_FREERATIO);
//    cv::imshow("n", Nfunc_y);
//    cv::namedWindow("nx", cv::WINDOW_FREERATIO);
//    cv::imshow("nx", Nfunc_x);

	cv::namedWindow(name);
    cv::namedWindow("mask");
    cv::namedWindow("testmask");

    cv::setMouseCallback(name, mouse_callback, &ud);
	cv::imshow(name, image);
	cv::waitKey();

	return 0;
}
