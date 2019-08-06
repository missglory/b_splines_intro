#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using std::cout;
using std::endl;
int w3, h3;
std::vector<cv::Point2f> controlPoints;
Mat orig;
Mat origg;
const std::string name = "Display Image";
static float eps = 0.000001;

Mat wr1, wr2;
bool is1 = 0;
//Mat wr11, wr22;
Mat tex = cv::Mat::zeros(4096, 4096, 16);

struct UserData {
    Mat orig;
    Mat image;

};
std::vector<cv::Point2f> src_pts;


float alph = 0.7;



bool SameSide(double x0, double y0, double x1, double y1, double x2, double y2, double x3, double y3)
{
	double x = (x3 - x2) * (y0 - y2) - (x0 - x2) * (y3 - y2);
	double y = (x3 - x2) * (y1 - y2) - (x1 - x2) * (y3 - y2);
	return x * y >= 0;
}


class Delaunay : public cv::Subdiv2D
{
public:
	Delaunay() : Subdiv2D() {}
	Delaunay(cv::Rect rect) : Subdiv2D(rect) {}


	void indices(std::vector<int>& ind) const
	{
		int i, total = (int)(qedges.size() * 4);
		std::vector<bool> edgemask(total, false);
		for (i = 4; i < total; i += 2)
		{
			if (edgemask[i]) continue;

			cv::Point2f a, b, c;

			int edge = i;
			int A = edgeOrg(edge, &a);
			if (A < 4) continue;

			edgemask[edge] = true;
			edge = getEdge(edge, NEXT_AROUND_LEFT);
			int B = edgeOrg(edge, &b);
			if (B < 4) continue;

			edgemask[edge] = true;
			edge = getEdge(edge, NEXT_AROUND_LEFT);
			int C = edgeOrg(edge, &c);
			if (C < 4) continue;

			edgemask[edge] = true;
			ind.push_back(A - 4); /// bz id begin with 4
			ind.push_back(B - 4);
			ind.push_back(C - 4);
		}
	}
};

template<class Type> void bilinInterp(const cv::Mat& I, double x, double y, Type* dst)
{
	int x1 = (int)std::floor(x);
	int y1 = (int)std::floor(y);
	int x2 = (int)std::ceil(x);
	int y2 = (int)std::ceil(y);
	if (x1 < 0 || x2 >= I.cols || y1 < 0 || y2 >= I.rows) return;

	const Type* p1 = (Type*)I.ptr(y1, x1);
	const Type* p2 = (Type*)I.ptr(y1, x2);
	const Type* p3 = (Type*)I.ptr(y2, x1);
	const Type* p4 = (Type*)I.ptr(y2, x2);
	for (int i = 0; i < I.channels(); i++)
	{
		double c1 = p1[i] + ((double)p2[i] - p1[i]) * (x - x1);
		double c2 = p3[i] + ((double)p4[i] - p3[i]) * (x - x1);
		*dst++ = (Type)(c1 + (c2 - c1) * (y - y1));
	}
}


template<class Point, class Type> void RenderFace(cv::Mat img, cv::Mat tex, const std::vector<Point>& points, const std::vector<Point>& tex_coord, std::vector<int> triangles)
{
	if (img.channels() != tex.channels()) std::logic_error("Different channels");

	const int n = triangles.size() / 3;
	for (int t = 0; t < n; t++)
	{
		int    ix_a = triangles[3 * t];
		int    ix_b = triangles[3 * t + 1];
		int    ix_c = triangles[3 * t + 2];
		Point  p_a = points[ix_a];
		Point  p_b = points[ix_b];
		Point  p_c = points[ix_c];
		double vz = (p_a.x - p_c.x) * (p_b.y - p_c.y) - (p_a.y - p_c.y) * (p_b.x - p_c.x);
		if (vz > 0)
		{
			Point t_a = tex_coord[ix_a];
			Point t_b = tex_coord[ix_b];
			Point t_c = tex_coord[ix_c];
			cv::Mat1d A(3, 3);  A << p_a.x, p_b.x, p_c.x, p_a.y, p_b.y, p_c.y, 1, 1, 1;
			cv::Mat1d B(3, 3);  B << t_a.x, t_b.x, t_c.x, t_a.y, t_b.y, t_c.y, 1, 1, 1;
			cv::Mat1d M = B * A.inv();

			double* affine = M.ptr<double>();

			int xmax = (int)std::ceil(std::max(std::max(p_a.x, p_b.x), p_c.x));
			int ymax = (int)std::ceil(std::max(std::max(p_a.y, p_b.y), p_c.y));
			int xmin = (int)std::floor(std::min(std::min(p_a.x, p_b.x), p_c.x));
			int ymin = (int)std::floor(std::min(std::min(p_a.y, p_b.y), p_c.y));
			if (xmax > img.cols - 1) xmax = img.cols - 1;
			if (ymax > img.rows - 1) ymax = img.rows - 1;
			if (xmin < 0) xmin = 0;
			if (ymin < 0) ymin = 0;

			for (int i = ymin; i <= ymax; i++)
			{
				Type* dst_p = (Type*)img.ptr(i, xmin);
				for (int j = xmin; j <= xmax; j++, dst_p += img.channels())
				{
					if (SameSide(j, i, p_a.x, p_a.y, p_b.x, p_b.y, p_c.x, p_c.y) && SameSide(j, i, p_b.x, p_b.y, p_c.x, p_c.y, p_a.x, p_a.y) && SameSide(j, i, p_c.x, p_c.y, p_a.x, p_a.y, p_b.x, p_b.y))
					{
						double x = affine[0] * j + affine[1] * i + affine[2];
						double y = affine[3] * j + affine[4] * i + affine[5];
						bilinInterp<Type>(tex, x, y, dst_p);
					}
				}
			}
		}
	}
}


float res = 2.f;

std::vector<cv::Point2f> readPoints(std::string file) {
	std::ifstream ifs;
	ifs.open(file);
	std::vector<cv::Point2f> ret;

	std::string st, st2;
	while (true) {
		ifs >> st >> st2;
		ret.push_back(cv::Point(std::stof(st) * res, std::stof(st2) * res));
		if (ret.size() > 84) break;
	}
	ifs.close();
	return ret;
}


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (flags & EVENT_FLAG_SHIFTKEY)
	{
		alph += 0.05f;

	}
	if (flags & EVENT_FLAG_CTRLKEY)
	{
		alph -= 0.05f;
	}
	if (flags & EVENT_FLAG_ALTKEY) {
		is1 = !is1;
		imwrite("C:/Users/User/Pictures/wrinkle_" + std::to_string((int)is1+1) + ".png", tex);
	}

	alph = std::max(0.f, std::min(1.0f, alph));

	static int activeCP = -1;
	static cv::Point startP(0, 0);
	cv::Point2f currentP(x, y);
	UserData* ud = (UserData*)userdata;
	//Mat& image = ud->image;
	Mat image;

	//Mat& orig = ud->orig;
	Mat& orig = is1 ? wr1 : wr2;

	//    static cv::Point del(w3,h3);
	if (event == EVENT_LBUTTONUP)
	{
		activeCP = -1;
	}
	if (event == EVENT_LBUTTONDOWN)
	{
		for (int i = 0; i < controlPoints.size(); i++)
		{
			if (cv::norm(currentP - controlPoints[i]) < 10)
			{
				activeCP = i;
				startP = { x,y };
			}
		}
	}
	if (event == EVENT_MOUSEMOVE && flags & EVENT_FLAG_LBUTTON)
	{
		//cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
		//      cout << "Part(" << px << ", " << py << ")" << endl;
		if (activeCP)
		{
			controlPoints[activeCP] = currentP;

			orig.copyTo(image);


			cv::Rect bbox = cv::boundingRect(controlPoints);
			Delaunay delaunay(bbox);
			delaunay.insert(controlPoints);
			std::vector<int> rough_tri;
			delaunay.indices(rough_tri);


			tex = cv::Mat::zeros(image.rows * 4, image.cols * 4, origg.type());
			RenderFace<cv::Point2f, uchar>(tex, image, src_pts, controlPoints, rough_tri);

			//std::string iter = "_wr2";
			//cv::imwrite("C:/Users/User/Desktop/wr/tex" + iter + ".png", tex);

			cv::Mat texx;
			cv::resize(tex, texx, cv::Size(tex.cols/4, tex.rows/4));

			imshow("res", origg - texx * alph);


			for (auto point : controlPoints)
			{
				cv::circle(image, point, 3, cv::Scalar(10, 250, 250), -1);
			}
			imshow(name, image);
		}
	}
}



int main(int argc, char** argv )
{
    Mat image;

    image = imread("C:/Users/User/Pictures/wrinkle2.bmp", 1);
	origg = imread("C:/Users/User/Pictures/V5BreeHeadM.jpg");
	
	float w = image.cols;
	float h = image.rows;
	w *= res;
	h *= res;
	cv::resize(origg, origg, cv::Size(w, h));
	const std::string txtname = "C:/Users/User/Desktop/wr/licop2.txt";

	controlPoints = readPoints(txtname);

	cv::resize(image, image, cv::Size(w,h));
	image.copyTo(orig);
    UserData ud;
    ud.image = image;
    ud.orig = orig;
    
	image.copyTo(wr2);
	wr1 = imread("C:/Users/User/Pictures/wrinkle1.bmp", 1);
	cv::resize(wr1, wr1, cv::Size(w, h));

	//std::vector<cv::Point2f> added = { {0.f,0.f},      {w / 3.f,0.f},    {2 * w / 3,0},      {w,0},
	//									{0,h / 3},     /*{w / 3,h / 3},  {2 * w / 3,h / 3},*/    {w,h / 3},
	//									{0,2 * h / 3},   /*{w / 3,2 * h / 3},{2 * w / 3,2 * h / 3},*/  {w,2 * h / 3},
	//									{0,h},       {w / 3,h},    {2 * w / 3,h},      {w,h} };
	//for (int i = 0; i < added.size(); i++) {
	//	controlPoints.push_back(added[i]);
	//	src_pts.push_back(added[i]);
	//}


	src_pts = readPoints("C:/Users/User/Desktop/wr/lico_src.txt");
	for (int i = 0; i < src_pts.size(); i++) {
		src_pts[i] /= 2.f;
	}

	


	if (image.empty())
    {
        printf("No image data \n");
        return -1;
    }
    for (auto point : controlPoints)
    {
        cv::circle(image, point, 3, cv::Scalar(10, 250, 250), -1);
    }
    namedWindow(name, WINDOW_AUTOSIZE );
	moveWindow(name, 0, -1080);
	namedWindow("res");
	moveWindow("res", 1920 - 1080, -1080);

	setMouseCallback(name, CallBackFunc, &ud);
    //imshow(name, image);
    
	while (true) {
		char key = waitKey() & 0xff;
		if (key == 'c')
			break;
	}
    
	std::ofstream ofs;
	ofs.open(txtname);
	for (int i = 0; i < controlPoints.size(); i++) {
		ofs << controlPoints[i].x / res << "\n" << controlPoints[i].y / res << "\n";
	}
	ofs.close();

	//imwrite("C:/Users/User/Pictures/wrinkle1.png", wr1);



	return 0;
}
