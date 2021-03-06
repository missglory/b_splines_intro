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

cv::Point2f ComputePoint(const cv::Mat& Nx, const cv::Mat& Ny, const std::vector<cv::Point2f>& controls, int x, int y)
{
	cv::Point2f ret;
	for (int i = 0; i < Nx.rows; i++) {
		for (int j = 0; j < Ny.rows; j++) {
			ret += Nx.at<float>(i, x) * Ny.at<float>(j, y) * controls[4 * j + i];
		}
	}
	return ret;
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
	float w			= image.cols;
	float h			= image.rows;
	std::vector<cv::Point2f> &controls = ud->controls;
	std::string &name = ud->window_name;

	if (event == cv::EVENT_LBUTTONUP) {
		activeCP = -1;
	}
	if (event == cv::EVENT_LBUTTONDOWN) {
		for (int i = 0; i < controls.size(); i++) {
			if (cv::norm(currentP - controls[i]) < 20) {
				activeCP = i;
				startP = { (float)x, (float)y };
			}
		}
	}
	if (event == cv::EVENT_MOUSEMOVE && flags == cv::EVENT_FLAG_LBUTTON) {
		if (activeCP >= 0) {
			controls[activeCP] = currentP;
			orig.copyTo(image);

			for (int i = 0; i < h; i++) {
				cv::Vec3b *p_im = image.ptr<cv::Vec3b>(i);
				for (int j = 0; j < w; j++) {
					cv::Point2f new_coord = ComputePoint(Nx, Ny, controls, j, i);
					cv::Point2f uv_coord = 2 * cv::Point2f(j, i) - new_coord;
					p_im[j] = BilinInterp(orig, uv_coord.x, uv_coord.y);
				}
			}

			for (auto &pt : controls) {
				cv::circle(image, pt, 5, cv::Scalar(10, 10, 255), -1);
			}
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

	
	const int control_num = 4;
	const int q = 4;
	std::vector<float> knots = { 0.f, 0.f, 0.f, 0.f, 1.0f, 1.f, 1.f, 1.f };
	std::vector<float> knots_x(knots.size()), knots_y(knots.size());
	for (size_t i = 0; i < knots.size(); i++) {
		knots_x[i] = w * knots[i];
		knots_y[i] = h * knots[i];
	}


	cv::Mat Nfunc_x(control_num, w, CV_32FC1), Nfunc_y(control_num, h, CV_32FC1);

	for (int n = 0; n < control_num; n++) {
		for (int t = 0; t < w; t++) {
			Nfunc_x.at<float>(n, t) = N(knots_x, t, n, q);
		}

		for (int t = 0; t < h; t++) {
			Nfunc_y.at<float>(n, t) = N(knots_y, t, n, q);
		}
	}

	for (auto &pt : controls) {
		cv::circle(image, pt, 5, cv::Scalar(10, 10, 250), -1);
	}

	std::string name = "Display Image";
	UserData ud;
	ud.image	= image;
	ud.orig		= orig;
	ud.Nx		= Nfunc_x;
	ud.Ny		= Nfunc_y;
	ud.controls = controls;
	ud.window_name = name;
	
	cv::namedWindow(name);
	cv::setMouseCallback(name, mouse_callback, &ud);
	cv::imshow(name, image);
	cv::waitKey();

	return 0;
}