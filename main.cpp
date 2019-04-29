#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


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

cv::Point2f ComputePoint(const cv::Mat& Nx, const cv::Mat& Ny, const std::vector<cv::Point2f>& controls, int x, int y, float startx = 0, float starty = 0)
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
    return cv::Point2f(dx + (dx2 - dx) * v);
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

	if (event == cv::EVENT_LBUTTONUP) {
		activeCP = -1;
	}
	if (event == cv::EVENT_LBUTTONDOWN) {
        for (int xx = 0; xx < n - 3; xx++) {
            for (int yy = 0; yy < 2; yy++)
            {
                int i = xx + yy * n;
                if (cv::norm(currentP - controls[i]) < 10) {
                    activeCP = i;
                    startP = { (float)x, (float)y };
                }
            }
		}
	}
	if (event == cv::EVENT_MOUSEMOVE && flags == cv::EVENT_FLAG_LBUTTON) {
        if (activeCP >= -1) {
			controls[activeCP] = currentP;
			orig.copyTo(image);

            int indx = 0;
            int indy = 0;

            for (int i = 0; i < tn; i++) {
				cv::Vec3b *p_im = image.ptr<cv::Vec3b>(i);
                while (indx < (knots_x.size() - 1) && knots_x[indx + 1] < i)
                {
                    indx++;
                }
                float u = (i - knots_x[indx]) / (knots_x[indx+1] - knots_x[indx]);


                for (int j = 0; j < tn2; j++) {
                    cv::Point2f new_coord = ComputePoint(Nx, Ny, controls, i, j, w/2, h/2);
//                    cv::Point2f uv_coord = 2 * cv::Point2f(j, i) - new_coord;
//                    p_im[j] = BilinInterp(orig, uv_coord.x, uv_coord.y);

//                    cv::circle(image, new_coord, 1, colors[i][j]/*cv::Scalar(10,250 - u * 250.f, u * 250.f)*/, -1);
//                    image.at<cv::Vec3b>(new_coord.y, new_coord.x) = colors.at<cv::Vec3b>(j,i);
                    image.at<cv::Vec3b>(new_coord.y, new_coord.x) = BilinInterp(colors, i, j);

//					cv::Point2f uv_coord = 2 * cv::Point2f(j, i) - new_coord;
//					p_im[j] = BilinInterp(orig, uv_coord.x, uv_coord.y);
                }
			}

            for (int i = 0; i < n - 3; i++) {
                for (int j = 0; j < 2; j++)
                {
                    cv::circle(image, controls[j * n + i], 2, cv::Scalar(10, 10, 255), -1);
                }
            }


//            cv::Point2f center(w/2, h/2);
//            cv::Point2f temp(x, y);
//            cv::line(image, center, temp, cv::Scalar(250, 250, 250));
//            float u = find_u(temp, center);
//            std::string angl = std::string("u: ") + std::to_string(u);
//            cv::putText(image, angl, cv::Point2f(100, 100), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));

//            u /= 17./14;
//            indx = 0;
//            while (indx < (knots_x.size() - 1) && knots_x[indx + 1] < u * (tn - 1))
//            {
//                indx++;
//            }
//            float u = (i - knots_x[indx]) / (knots_x[indx+1] - knots_x[indx]);

//            cv::circle(image, controls[indx], 7, cv::Scalar(250, 0, 0), -1);




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
        float amp = 100;
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
        Nfunc_y.at<float>(1, t) = (float)t / (tn2 - 1);
        Nfunc_y.at<float>(0, t) = 1 - (float)t / (tn2 - 1);
    }

	for (auto &pt : controls) {
		cv::circle(image, pt, 5, cv::Scalar(10, 10, 250), -1);
	}

//    colors.resize(tn, std::vector<cv::Vec3b>(tn2, {0,0,0}));
    cv::Mat colors = cv::Mat::zeros(tn2, tn, CV_8UC3);

    int indx = 0;

    for (int i = 0; i < tn; i++)
    {
        while (indx < (knots_x.size() - 1) && knots_x[indx + 1] < i)
        {
            indx++;
        }
        float u = (i - knots_x[indx]) / (knots_x[indx+1] - knots_x[indx]);

        for (int j = 0; j < tn2; j++) {
//            cv::Point2f new_coord = ComputePoint(Nfunc_x, Nfunc_y, controls, i, j, w/2, h/2);
            cv::Point2f uv = findP(u, (float)j/(tn2), controls[(indx+n-5)%(n-3)], controls[(indx+n-4)%(n-3)], controls[(indx+n-5)%(n-3)+n], controls[(indx+n-4)%(n-3)+n]);
//            colors.at<cv::Vec3b>(j,i) = orig.at<cv::Vec3b>(uv.y, uv.x);
            colors.at<cv::Vec3b>(j,i) = BilinInterp(orig, uv.x, uv.y);
//            colors[i][j] = orig.at<cv::Vec3b>(uv.y, uv.x);

//                    cv::circle(image, new_coord, 1, cv::Scalar(10,250 - u * 250.f, u * 250.f), -1);

        }
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
    ud.colors =  colors;
    ud.n = n;


    cv::namedWindow("colors");
    cv::imshow("colors", colors);
//    cv::namedWindow("n", cv::WINDOW_FREERATIO);
//    cv::imshow("n", Nfunc_y);
//    cv::namedWindow("nx", cv::WINDOW_FREERATIO);
//    cv::imshow("nx", Nfunc_x);

	cv::namedWindow(name);
    cv::setMouseCallback(name, mouse_callback, &ud);
	cv::imshow(name, image);
	cv::waitKey();

	return 0;
}
