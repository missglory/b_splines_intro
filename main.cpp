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
	cv::Mat uv_mask;
    cv::Mat U_V_map;
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

    float norm = 0.f;
    for (int i = 0; i < Nx.rows; i++) {
        for (int j = 0; j < Ny.rows; j++) {
            norm += Nx.at<float>(i, x) * Ny.at<float>(j, y);
        }
    }
    if (norm < 0.9999)
        return cv::Point2f(-1,-1);

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


void putAffine(cv::Mat& dest, int i1, int i2, int i3, cv::Point2f* uv_coords, cv::Point2f* xy_coords, cv::Mat& U_V_map, cv::Mat& used)
{
    float sz = U_V_map.cols;
    std::vector<cv::Point2f> tri1 = {{uv_coords[i1].y * (sz-1), uv_coords[i1].x * (sz-1)},
                                     {uv_coords[i2].y * (sz-1), uv_coords[i2].x * (sz-1)},
                                     {uv_coords[i3].y * (sz-1), uv_coords[i3].x * (sz-1)}
                                    };
    std::vector<cv::Point2f> tri2 = {xy_coords[i1], xy_coords[i2], xy_coords[i3]};
    cv::Rect r1 = cv::boundingRect(tri1);
    cv::Rect r2 = cv::boundingRect(tri2);

    std::vector<cv::Point2f> tri1Cropped, tri2Cropped;
    tri1Cropped.reserve(3);
    tri2Cropped.reserve(3);
    std::vector<cv::Point> tri2CroppedInt;
    tri2CroppedInt.reserve(3);
    for(int i = 0; i < 3; i++)
    {
        tri1Cropped.emplace_back( cv::Point2f( tri1[i].x - r1.x, tri1[i].y -  r1.y) );
        tri2Cropped.emplace_back( cv::Point2f( tri2[i].x - r2.x, tri2[i].y - r2.y) );
        tri2CroppedInt.emplace_back( cv::Point(std::round(tri2[i].x - r2.x), std::round(tri2[i].y - r2.y)) );
    }

    // Apply warpImage to small rectangular patches
    cv::Mat img1Cropped;
    U_V_map(r1).copyTo(img1Cropped);

    cv::Mat warpMat = cv::getAffineTransform( tri1Cropped, tri2Cropped );
    cv::Mat img2Cropped = cv::Mat::zeros(r2.height, r2.width, img1Cropped.type());
    warpAffine( img1Cropped, img2Cropped, warpMat, img2Cropped.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

    // Get mask by filling triangle
    cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, img2Cropped.type());
    cv::fillConvexPoly(mask, tri2CroppedInt, cv::Scalar::all(1.), 16, 0);

    // Copy triangular region of the rectangular patch to the output image
    cv::multiply(img2Cropped, mask, img2Cropped);
    used(r2) += mask;



    cv::Mat used_gray(used.cols, used.rows, CV_32F);
    cv::cvtColor(used, used_gray, cv::COLOR_BGR2GRAY);
    used_gray = 2.f - used_gray;
    threshold(used_gray, used_gray, .9, 1., cv::THRESH_BINARY);
    cv::Mat dest_add;
    cv::Mat used_merge(r2.width, r2.height, CV_32FC3);
    cv::Mat used_merge_[] = {used_gray(r2), used_gray(r2), used_gray(r2)};
    cv::merge(used_merge_, 3, used_merge);
    cv::multiply(img2Cropped, used_merge, dest_add);


    dest(r2) += dest_add;
}



cv::Mat FindUVCoords(const UserData* ud)
{
    int n_u								= ud->n_u;
    int n_u_shift                       = 2;
    int n_u_active                      = n_u - n_u_shift;
	int n_v								= ud->n_v;
    int w								= ud->orig.cols;
    int h								= ud->orig.rows;
	int quant							= ud->quantize_u;
	const std::vector<cv::Point2f> &controls	= ud->controls;
    const std::vector<float> &knots_x			= ud->knots_x;  
    cv::Mat& U_V_map = (cv::Mat&)ud->U_V_map;
    cv::Mat uv_mask = cv::Mat::zeros(w, h, CV_32FC3);
    cv::Mat used = cv::Mat::zeros(w, h, CV_32FC3);

    for(int i = 0; i < n_u_active - 1; i++) {
		for (int j = 0; j < n_v - 1; j++) {
			cv::Point2f uv_coords[4];
            uv_coords[0] = { knots_x[i + n_u_shift + 1], 1.f / (n_v - 1) * j };
            uv_coords[1] = { knots_x[i + n_u_shift + 1], 1.f / (n_v - 1) * j + 1.f / (n_v - 1) };
            uv_coords[2] = { knots_x[i + n_u_shift + 2], 1.f / (n_v - 1) * j + 1.f / (n_v - 1) };
            uv_coords[3] = { knots_x[i + n_u_shift + 2], 1.f / (n_v - 1) * j };

			cv::Point2f xy_coords[4];
			xy_coords[0] = controls[i + j * n_u];
			xy_coords[1] = controls[i +	(j + 1) * n_u];
			xy_coords[2] = controls[i + 1 + (j + 1) * n_u];
			xy_coords[3] = controls[i + 1 + j * n_u];

            putAffine(uv_mask, 0, 2, 3, uv_coords, xy_coords, U_V_map, used);
            putAffine(uv_mask, 0, 1, 2, uv_coords, xy_coords, U_V_map, used);
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
	
	cv::Mat& image		= ud->image;
	cv::Mat& orig		= ud->orig;
	cv::Mat& Nx			= ud->Nx;
	cv::Mat& Ny			= ud->Ny;
	cv::Mat& uv_coord	= ud->uv_mask;
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

        for (int xx = 0; xx < n_u; xx++) {
            for (int yy = 0; yy < n_v; yy++)
            {
                int i = xx + yy * n_u;
                if (cv::norm(currentP - controls[i]) < 5) {
                    activeCP = i;
                    startP = { (float)x, (float)y };
                }
            }
		}
	}

    if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)) {
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

            //uv_coord = FindUVCoords(ud);
            (orig).copyTo(image);
            image /= 5.f;
//			image.setTo(cv::Scalar::all(0));

            for(int i = 0; i < h; i++)
            {
                for(int j = 0; j < w; j++)
                {
                    cv::Point2f curPoint(j, i);
                    cv::Vec3f u_v = uv_coord.at<cv::Vec3f>(curPoint);
                    if (!u_v[0] && !u_v[1]) continue;

                    u_v[0] = std::max(0.f, std::min((float)normalization_u - 1, u_v[0] * (normalization_u - 1)));
                    u_v[1] = std::max(0.f, std::min((float)normalization_v - 1, u_v[1] * (normalization_v - 1)));

                    cv::Point2f new_coord = ComputePoint(Nx, Ny, controls, u_v[0], u_v[1]);
                    if(new_coord.x == -1) continue;
                    image.at<cv::Vec3b>(new_coord) = orig.at<cv::Vec3b>(curPoint); // BilinInterp(orig, curPoint.x, curPoint.y);
                }
            }

            for (int i = 0; i < n_u; i++) {
                for (int j = 0; j < n_v; j++)
                {
                    cv::circle(image, controls[j * n_u + i], 2, cv::Scalar(255 / n_v * j, 255 / n_u * i, 255 - 255 / n_u * i), -1);
                }
            }
            cv::circle(image, pivot, 5, cv::Scalar(10, 250, 0), 10);
            cv::imshow(name, image);
		}
	}
}

int main()
{
    cv::Mat image = cv::imread("../boy.jpg");
    cv::resize(image, image, cv::Size(998,1089));
    //image.resize(998,998);
    image = image(cv::Rect(0, 0,998,998)).clone();

    cv::Mat orig = image.clone();
	float w = (float)image.cols;
	float h = (float)image.rows;

    std::vector<cv::Point2f> controls = {
        {361.838f, 	261.161f},
        {393.884f, 	250.429f},
        {427.584f, 	243.46f},
        {458.01f, 	239.389f},
        {483.072f, 	237.162f},
        {504.201f, 	236.74f},
        {525.344f, 	237.31f},
        {550.223f, 	239.704f},
        {580.256f, 	243.961f},
        {613.362f, 	251.105f},
        {644.714f, 	261.963f},
        {645.777f, 	287.308f},
        {673.578f, 	301.361f},
        {695.803f, 	318.801f},
        {689.557f, 	343.908f},
        {710.289f, 	367.025f},
        {724.755f, 	393.325f},
        {720.871f, 	416.786f},
        {731.93f, 	449.989f},
        {727.433f, 	464.442f},
        {733.875f, 	492.751f},
        {730.377f, 	502.424f},
        {735.554f, 	526.468f},
        {732.015f, 	530.796f},
        {735.683f, 	551.196f},
        {738.065f, 	568.699f},
        {738.472f, 	583.393f},
        {737.903f, 	594.375f},
        {736.487f, 	603.292f},
        {734.432f, 	612.774f},
        {739.425f, 	628.165f},
        {735.465f, 	640.381f},
        {736.357f, 	658.911f},
        {731.094f, 	672.669f},
        {729.507f, 	688.394f},
        {724.627f, 	698.345f},
        {719.95f, 	715.598f},
        {715.052f, 	740.086f},
        {710.458f, 	768.554f},
        {700.196f, 	785.361f},
        {690.058f, 	821.449f},
        {677.557f, 	850.261f},
        {665.428f, 	866.57f},
        {649.302f, 	890.465f},
        {636.362f, 	904.538f},
        {617.077f, 	925.033f},
        {603.409f, 	937.094f},
        {584.595f, 	949.892f},
        {571.584f, 	957.17f},
        {557.047f, 	961.633f},
        {543.451f, 	964.359f},
        {529.636f, 	966.448f},
        {513.479f, 	967.481f},
        {501.093f, 	967.77f},
        {488.885f, 	967.47f},
        {472.764f, 	966.167f},
        {458.579f, 	964.177f},
        {445.367f, 	961.157f},
        {430.764f, 	957.105f},
        {417.801f, 	949.477f},
        {398.98f, 	936.82f},
        {384.797f, 	925.252f},
        {365.859f, 	904.203f},
        {353.3f, 	889.853f},
        {337.195f, 	865.729f},
        {324.994f, 	849.411f},
        {312.691f, 	820.711f},
        {302.98f, 	784.098f},
        {292.64f, 	767.532f},
        {287.827f, 	739.237f},
        {282.863f, 	714.784f},
        {278.284f, 	697.27f},
        {273.176f, 	687.762f},
        {271.945f, 	672.201f},
        {266.438f, 	658.38f},
        {267.712f, 	640.177f},
        {263.508f, 	627.454f},
        {268.851f, 	611.998f},
        {266.559f, 	602.573f},
        {264.971f, 	593.528f},
        {264.5f, 	582.55f},
        {264.878f, 	567.811f},
        {267.225f, 	550.19f},
        {271.115f, 	529.86f},
        {267.434f, 	525.439f},
        {272.81f, 	501.544f},
        {269.187f, 	491.807f},
        {276.038f, 	463.551f},
        {271.481f, 	449.045f},
        {283.147f, 	415.873f},
        {279.344f, 	392.348f},
        {294.436f, 	366.082f},
        {315.941f, 	343.059f},
        {309.55f, 	317.797f},
        {332.506f, 	300.483f},
        {361.128f, 	286.532f}
    };
    int n_u_add = 3;
    int n_u = controls.size() + n_u_add, n_v = 4, n_u_active = n_u - n_u_add;
    float xmin = w, xmax = 0, ymin = h, ymax = 0;
    for(auto p : controls)
    {
        xmin = std::min(xmin, p.x);
        xmax = std::max(xmax, p.x);
        ymin = std::min(ymin, p.y);
        ymax = std::max(ymax, p.y);
    }
    controls.resize(n_u * n_v);
    cv::Point2f center((xmin + xmax)/2.f, (ymin+ymax)/2.f);
    float r0 = 1.f;
    float dr = .1f;

    for (int i = 0; i < n_u_add; i++)
    {
        controls[n_u - i - 1] = controls[n_u_add - 1 - i];
    }
    for (int i = 0; i < n_u; i++) {
//        float angle = (float)(i%(n_u_active-1)) / (n_u_active - 1) * 2.f * 3.14f;
        cv::Point2f delta = controls[i] - center;
        for (int j = 0; j < n_v; j++) {
			float r = r0 + j * dr;
            cv::Point2f coord(center + delta * r);
            controls[j * n_u + i] = coord - cv::Point2f{0, 100};
        }
    }

    const int q = 4;
    int kx_size = n_u + q;

	std::vector<float> knots_y = { 0.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 1.f };
    std::vector<float> knots_x(kx_size, 0.f);

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
			int x = j;
			int y = w2 * (1 - Nfunc_y.at<float>(i, j));

			x = std::min(quantize - 1, std::max(0, x));
			y = std::min(w2 - 1, std::max(0, y));

			plot.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
		}
	}
	cv::imshow("plot", plot);
	cv::waitKey(10);
#endif

	for (auto &pt : controls) {
        cv::circle(image, pt, 2, cv::Scalar(10, 10, 250), -1);
	}
	cv::circle(image, cv::Point(100,100), 5, cv::Scalar(10, 250, 0), 10);

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

    float sz =  1000;
    ud.U_V_map = cv::Mat::zeros(sz, sz, CV_32FC3);
    for(float x = 0; x < sz; x++)
    {
        for(float y = 0; y < sz; y++)
            ud.U_V_map.at<cv::Vec3f>(y, x) = cv::Vec3f(y/sz, x/sz);
    }
    ud.uv_mask		= FindUVCoords(&ud);

    cv::namedWindow(name);

    cv::setMouseCallback(name, mouse_callback, &ud);
	cv::imshow(name, image);
	cv::waitKey();

	return 0;
}
