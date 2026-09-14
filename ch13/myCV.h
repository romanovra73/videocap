#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <pqxx/pqxx>
#include <cpr.h>

using namespace cv;
using namespace std;

#ifndef CV_CLASS_H
#define CV_CLASS_H

class myCV {
public:
    static string get_db_connection_string();
    static bool motion_detected(Mat new_frame, Mat prev_frame, double pixel_threshold, double total_threshold=1000);
    static bool face_detected(Mat frame, CascadeClassifier face_cascade, Size min_size, Size max_size);
    static bool mark_faces(Mat &frame, CascadeClassifier face_cascade, Size min_size, Size max_size);
    static double blur_varience(Mat frame);
    static bool person_detected(Mat frame);
    static int npersons_detected(Mat frame);
    static double pool_detected(Mat frame);
    static void run_tracking(int channel);
};
#endif
