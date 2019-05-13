#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

cv::Mat UV_MASK;

struct UserData
{
	cv::Mat orig;
	cv::Mat image;
	cv::Mat Nx;
	cv::Mat Ny;
	std::vector<cv::Point2f> controls;
	std::string window_name;
    std::vector<float> knots_x, knots_y;
    int quantize_u, quantize_v, n_u, n_v;
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
            cv::Point2f control = controls[Nx.rows * j + i];
            ret += Nx.at<float>(i, x) * Ny.at<float>(j, y) * controls[Nx.rows * j + i];
		}
	}
	return ret;
}

cv::Point2f ComputePoint(const cv::Mat& Nx, const cv::Mat& Ny, const std::vector<cv::Point2f>& controls, cv::Point2f point)
{
    return ComputePoint(Nx, Ny, controls, point.x, point.y);
}

cv::Mat _rendertestmask(cv::Mat& testimage)
{
    cv::Mat rendertestimage = cv::Mat::zeros(testimage.cols, testimage.rows, CV_32FC3);
    int from_to[] = { 0,0, 1,1, -1,2 };
    cv::mixChannels({{testimage}}, {{rendertestimage}}, from_to, testimage.channels());
    return rendertestimage;
}

typedef struct{
    std::vector<cv::Point2f> points;
    cv::Mat transform;
    std::vector<cv::Point2f> quantize_params;
} pointsTransform;


cv::Mat find_uv_coords(UserData* ud)
{
    int n_u								= ud->n_u;
	int n_v								= ud->n_v;
    int w								= ud->orig.cols;
    int h								= ud->orig.rows;
	int quant							= ud->quantize_u;
	std::vector<cv::Point2f> &controls	= ud->controls;
    std::vector<float> &knots_x			= ud->knots_x;  

    cv::Mat indmask = cv::Mat::zeros(w, h, CV_8U);
    cv::Mat uv_mask = cv::Mat::zeros(w, h, CV_32FC2);
    
    // 1. FIND PERSPECTIVE TRANSFORM MATRIX FOR EACH REGION
	int ind = 1;
	std::vector<pointsTransform> pT((n_u - 1) * (n_v - 1) + 1);

	for(int i = 0; i < n_u - 1; i++) {
		for (int j = 0; j < n_v - 1; j++) {
			cv::Point2f uv_coords[4];
			uv_coords[0] = { knots_x[i + 2], 0.33f * j };
			uv_coords[1] = { knots_x[i + 2], 0.33f * j + 0.33f };
			uv_coords[2] = { knots_x[i + 3], 0.33f * j + 0.33f };
			uv_coords[3] = { knots_x[i + 3], 0.33f * j };

			cv::Point2f xy_coords[4];
			xy_coords[0] = controls[i + j * n_u];
			xy_coords[1] = controls[i +	(j + 1) * n_u];
			xy_coords[2] = controls[i + 1 + (j + 1) * n_u];
			xy_coords[3] = controls[i + 1 + j * n_u];

			cv::Point xy_coords_int[4];
			for (int i = 0; i < 4; i++) xy_coords_int[i] = cv::Point(std::round(xy_coords[i].x), std::round(xy_coords[i].y));

			cv::fillConvexPoly(indmask, xy_coords_int, 4, cv::Scalar(ind));
			pT[ind].transform = cv::getPerspectiveTransform(xy_coords, uv_coords);
			ind++;
		}
    }

    // 2. FIND ALL POINTS TO TRANSFORM
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            cv::Point2f cur_p(j,i);
            int ind = indmask.at<unsigned char>(cur_p);
            if (!ind) continue;
            pT[ind].points.push_back(cv::Point2f(cur_p));
        }
    }

    // 3. TRANSFORM POINTS
    for(int i = 1; i < pT.size(); i++) {
		if (!pT[i].points.size()) continue;
        pT[i].quantize_params.resize(pT[i].points.size());
        cv::perspectiveTransform(pT[i].points, pT[i].quantize_params, pT[i].transform);
        
		for (int j = 0; j < pT[i].points.size(); j++)
        {
            float u = pT[i].quantize_params[j].x;
            float v = pT[i].quantize_params[j].y;
            uv_mask.at<cv::Vec2f>(pT[i].points[j]) = cv::Vec2f(u, v);
        }
    }

    return uv_mask;
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
    std::vector<float>& knots_x = ud->knots_x;
    std::vector<float>& knots_y = ud->knots_y;
	float w			= image.cols;
	float h			= image.rows;
	std::vector<cv::Point2f> &controls = ud->controls;
	std::string &name = ud->window_name;
    int normalization_u = ud->quantize_u;
    int normalization_v = ud->quantize_v;
    int n_u = ud->n_u;
    int n_v = ud->n_v;

    static cv::Mat uv_coord;
    cv::Mat rendertestmask = cv::Mat::zeros(w, h, CV_32FC3);

    static const cv::Point2f originalPivot(100, 100);
    static cv::Point2f pivot = originalPivot;
	static cv::Point2f prev_pivot = originalPivot;
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

        for (int xx = 0; xx < n_u - 3; xx++) {
            for (int yy = 0; yy < 2; yy++)
            {
                int i = xx + yy * n_u;
                if (cv::norm(currentP - controls[i]) < 3) {
                    activeCP = i;
                    startP = { (float)x, (float)y };
                }
            }
		}
	}

	if (event == cv::EVENT_MOUSEMOVE && flags == cv::EVENT_FLAG_LBUTTON) {
        if (activeCP > -1)
        {
            if (activeCP == pivotId)
            {
				prev_pivot = pivot;
                pivot = currentP;
                for (int i = 0; i < n_u; i++)
                {
                    for (int j = 0; j < 1; j++)
                    {
                        controls[i + j*n_u] += pivot - prev_pivot;
                    }
                }
            }
            else
            {
                controls[activeCP] = currentP;
            }

            orig.copyTo(image);
			image.setTo(cv::Scalar::all(0));
            //uv_coord = find_uv_coords(ud);
			uv_coord = UV_MASK;

            for(int i = 0; i < h; i++)
            {
                for(int j = 0; j < w; j++)
                {
                    cv::Point2f curPoint(j, i);
                    cv::Vec2f u_v = uv_coord.at<cv::Vec2f>(curPoint);
                    if (!u_v[0] && !u_v[1]) continue;

                    u_v[0] = std::max(0.f, std::min((float)normalization_u - 1, u_v[0] * (normalization_u - 1)));
                    u_v[1] = std::max(0.f, std::min((float)normalization_v - 1, u_v[1] * (normalization_v - 1)));
                    cv::Point2f new_coord = ComputePoint(Nx, Ny, controls, u_v);
                    cv::Point2f uv_coord = curPoint * 2 - new_coord;
                    //image.at<cv::Vec3b>(curPoint) = BilinInterp(orig, uv_coord.x, uv_coord.y);
					image.at<cv::Vec3b>(new_coord) = orig.at<cv::Vec3b>(curPoint);
                }
            }

            rendertestmask = _rendertestmask(uv_coord);
            imshow("testmask", rendertestmask);

            for (int i = 0; i < n_u - 3; i++) {
                for (int j = 0; j < n_v; j++)
                {
                    cv::circle(image, controls[j * n_u + i], 2, cv::Scalar(10, 10, 255, 120), -1);
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

	int n_u = 50, n_v = 4;
	std::vector<cv::Point2f> controls(n_v * n_u);
    cv::Point2f center(w/2, h/2);
	float r0 = 120;
	float dr = 30;

	for (int i = 0; i < n_u; i++) {
        float angle = (float)i / (n_u - 1) * 2.f * 3.14f;
       
        for (int j = 0; j < n_v; j++) {
			float r = r0 + j * dr;
			cv::Point2f coord(r * cos(angle), r * sin(angle));

            controls[j * n_u + i] = center + coord;
        }
    }

    const int q = 4;
	int kx_size = n_u + q + 1;

	std::vector<float> knots_y = { 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f };
	std::vector<float> knots_x(kx_size);

    for (int i = 0; i < kx_size; i++) {
        knots_x[i] = i / (kx_size - 1.f);
    }
    
	int quantize = 1000;
    cv::Mat Nfunc_x(n_u, quantize, CV_32FC1), Nfunc_y(n_v, quantize, CV_32FC1);
    for (int i = 0; i < n_u; i++) {
        for (int t = 0; t < quantize; t++) {
            Nfunc_x.at<float>(i, t) = N(knots_x, t / (quantize - 1.f), i, q);
		}
	}
    for (int i = 0; i < n_v; i++) {
        for (int t = 0; t < quantize; t++) {
            Nfunc_y.at<float>(i, t) = N(knots_y, t / (quantize - 1.f), i, q);
        }
    }

#if 0
	int w2 = 500;
	cv::Mat plot(w2, quantize, CV_8UC3, cv::Scalar::all(0));
	for (int i = 0; i < Nfunc_y.rows; i++) {
		for (int j = 0; j < Nfunc_y.cols; j++) {
			plot.at<cv::Vec3b>(500 * (1 - Nfunc_y.at<float>(i, j)), j) = cv::Vec3b(0, 0, 255);
		}
	}
	cv::imshow("plot", plot);
	cv::waitKey();
#endif

	for (auto &pt : controls) {
        cv::circle(image, pt, 2, cv::Scalar(10, 10, 250), -1);
	}

	std::string name = "Display Image";
	UserData ud;
	ud.image		= image;
	ud.orig			= orig;
	ud.Nx			= Nfunc_x;
    ud.Ny			= Nfunc_y;
	ud.controls		= controls;
	ud.window_name	= name;
    ud.knots_x		= knots_x;
    ud.knots_y		= knots_y;
    ud.quantize_u	= quantize;
    ud.quantize_v	= quantize;
    ud.n_u			= n_u;
    ud.n_v			= n_v;

	UV_MASK = find_uv_coords(&ud);

	cv::namedWindow(name);
    cv::namedWindow("testmask");
    cv::setMouseCallback(name, mouse_callback, &ud);
	cv::imshow(name, image);
	cv::waitKey();

	return 0;
}
