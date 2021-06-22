// M10915106_HW3.cpp : 此檔案包含 'main' 函式。程式會於該處開始執行及結束執行。
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <direct.h>
#include <opencv2\imgproc\types_c.h>
#include <opencv2/highgui/highgui_c.h>


#define OUTIMGNAME "M10915106.jpg"

//#define DEBUG_HSV
//#define DEBUG
//#define DEBUG_TRPZ
#define VIDEO_MODE true

using namespace cv;
using namespace std;

Mat ROI(Mat img) {
    float height, width;

    height = img.rows;
    width = img.cols;

    //cout << height << " "  << width << endl;

    // trapezoid
    Point trpz[1][4];
    if (VIDEO_MODE)
    {
        trpz[0][0] = Point(0, height * 0.9);
        trpz[0][1] = Point(width * 0.2, height * 3 / 5);
        trpz[0][2] = Point(width * 0.8, height * 3 / 5);
        trpz[0][3] = Point(width * 0.9, height * 0.9);
    }
    else
    {
        trpz[0][0] = Point(width * 0.1, height);
        trpz[0][1] = Point(width * 0.4, height * 3 / 5);
        trpz[0][2] = Point(width * 0.6, height * 3 / 5);
        trpz[0][3] = Point(width * 0.9, height);
    }
    
    const Point* ppt[1] = { trpz[0] };
    int npt[] = { 4 };

    Mat mask(height, width, img.type(), Scalar(0,0,0));
    fillPoly(mask, ppt, npt, 1, Scalar(255, 255, 255));
    
    Mat dst;
    img.copyTo(dst, mask);

#ifdef DEBUG_TRPZ
    imshow("After mask", dst);
    imshow("mask", mask);
#endif // DEBUG

    return dst;
}

float* polyfit1d(Point p1, Point p2)
{
    /*
        params[0]: slope
        params[1]: y-intercept 
    */
    
    // slope
    float y = p2.y - p1.y;
    float x = p2.x - p1.x;
    float m = y / x ;
    
    // y_intercept
    float y_int = p1.y - (m * p1.x);
    

    float params[2] = {m, y_int};

    return params;
}

float average_vector(vector<float> vec)
{
    float average = 0.0;
    int size = vec.size();
    for (int i=0; i < size; i++)
    {
        average += vec[i];
    }

    return average / size;
}

vector<int> make_points(Mat img, float avg_slope, float avg_yint)
{
    /*
        Return float[4] {x1, y1, x2, y2 }
    */
    int y1 = img.rows;
    // how long we want our lines to be-- > 3 / 5 the size of the image
    int y2 = (int) (y1 * 3 / 5);

    // determine algebraically
    int x1 = (int)(floor((y1 - avg_yint) / avg_slope));
    int x2 = (int)(floor((y2 - avg_yint) / avg_slope));

    //int rst[4] = {x1, y1, x2, y2};
    vector<int> rst(4);
    rst[0] = x1;
    rst[1] = y1;
    rst[2] = x2;
    rst[3] = y2;


    return rst;
}

Mat display_lines(Mat img, vector<vector<int>>lines)
{
    Mat line_img = Mat(img.rows, img.cols, img.type(), Scalar(0,0,0));
    
    for (int i = 0 ; i < lines.size(); i++)
    {
        int x1 = lines[i][0], y1 = lines[i][1], x2 = lines[i][2], y2 = lines[i][3];
        cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
        line(line_img, Point(x1, y1), Point(x2, y2), Scalar(0,0,255), 10);
    }
    return line_img;
}

vector<vector<int>> average(Mat img, vector<Vec4i> lines)
{
    /*
        Average the line found form HoughlineP,
        Return left_line, right_line;
    */
    vector<float>left_slope;
    vector<float>left_yint;

    vector<float>right_slope;
    vector<float>right_yint;

    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        float* params = polyfit1d(Point(l[0], l[1]), Point(l[2], l[3]));
        float slope = params[0];
        float y_int = params[1];

#ifdef DEBUG
        cout << i << ": " << "slope: " << slope << "  y_int: " << y_int << endl;
#endif // DEBUG


        // left line
        if (slope < 0)
        {
            left_slope.push_back(slope);
            left_yint.push_back(y_int);
        }
        // right line
        else
        {
            right_slope.push_back(slope);
            right_yint.push_back(y_int);
        }
        //delete[] params;
    }

    // takes average among all the columns
    float left_slope_avg = 0, left_yint_avg = 0;
    float right_slope_avg = 0, right_yint_avg = 0;

    left_slope_avg = average_vector(left_slope);
    right_slope_avg = average_vector(right_slope);
    left_yint_avg = average_vector(left_yint);
    right_yint_avg = average_vector(right_yint);

    // create lines based on averages calculates
    vector<int> left_line = make_points(img, left_slope_avg, left_yint_avg);
    vector<int> right_line = make_points(img, right_slope_avg, right_yint_avg);
    
    vector<vector<int>> rst;
    rst.push_back(left_line);
    rst.push_back(right_line);

    return rst;
}

void process_img(Mat im1)
{
    // Gray scale
    Mat gray1;
    cvtColor(im1, gray1, COLOR_BGR2GRAY);

    // HSV filter white and yellow
    Mat imHSV;
    cvtColor(im1, imHSV, COLOR_BGR2HSV);
    Scalar lower_yellow = Scalar(20, 100, 100);
    Scalar upper_yellow = Scalar(30, 255, 255);

    Mat maskY, maskW;
    inRange(imHSV, lower_yellow, upper_yellow, maskY);
    inRange(imHSV, Scalar(0, 0, 220), Scalar(180, 255, 255), maskW);

    Mat mask;
    bitwise_or(maskW, maskY, mask);
    bitwise_and(gray1, mask, gray1);

    // Gaussian Blur
    GaussianBlur(gray1, gray1, Size(5, 5), 0);
    
    // Canny
    Mat canny;
    Canny(gray1, canny, 100, 200);

    // ROI
    Mat canny_roi = ROI(canny);

    // Hough line
    vector<Vec4i> lines;
    HoughLinesP(canny_roi, lines, 2, CV_PI / 180, 100, 30, 5);

    cout << lines.size() << " lines detected" << endl;
    if (lines.size() < 1)
    {
        return;
    }

    vector<vector<int>> avg_line = average(im1, lines);
    Mat blacklines = display_lines(im1, avg_line);


    Mat lanes;

    addWeighted(im1, 0.8, blacklines, 1, 1, lanes);

    // output frame
    imshow("lanes", lanes);

    

#ifdef DEBUG_HSV
    imshow("img1", im1);
    imshow("HSV", imHSV);
    imshow("maskw", maskW);
    imshow("masky", maskY);
    imshow("mask", mask);
    imshow("after HSV", gray1);

#endif // DEBUG

#ifdef DEBUG
    
    imshow("Blur", gray1);
    imshow("Canny", canny);
    imshow("simple lines", im1);
    imshow("lines", blacklines);
    // Draw the lines
    Mat im_line;
    im1.copyTo(im_line);
    for (size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(im_line, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3);
    }
    imshow("simple lines", im_line);
#endif // DEBUG
}

int main(int argc, char const* argv[])
{
    try
    {
        if (! VIDEO_MODE)
        {
            // load source images.
            Mat im1 = imread("./image/lane4.jpg");
            Mat im2 = imread("./image/lane2.jpg");

            process_img(im2);
        }
        else
        {
            VideoCapture vc;
            string filename = "./video/test2.mp4";

            cout << "WE" << endl;

            vc.open(filename);
            if (!vc.isOpened())
            {
                cout << "Video " + filename + " not opened!" << endl;
                return 1;
            }

            cout << "Start " + filename << " playing..." << endl;

            while (true)
            {
                Mat frame;
                bool success = vc.read(frame);

                if (!success)
                {
                    cout << "No frame are read" << endl;
                    break;
                }

                process_img(frame);

                // Press ESC to quit
                if (waitKey(1) == 27)
                {
                    break;
                }
            }

            vc.release();
        }

        waitKey(0);
        destroyAllWindows();
    }
    catch (const std::exception& e)
    {
        cout << "exception: " << e.what() << endl;  // output error message.
        return 1;
    }
    
    return 0;
}