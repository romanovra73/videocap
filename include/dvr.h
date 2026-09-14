#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <stdio.h>
#include <pqxx/pqxx>
#include <cmath>

using namespace cv;
using namespace std;

#ifndef DVR_CLASS_H
#define DVR_CLASS_H
class DVR_channel {
    private:
	string ID;
	unsigned int channel;
	bool main_stream;
	string connection_string;
	VideoCapture capture;
	unsigned int frame_width;
	unsigned int frame_height;
	string error;
	
	void connect_DVR();
	void grab_frame();
	bool check_frame(Mat frame, int blur_filter, bool ignore_size, int &empty_frames);
	double get_blur_varience(Mat frame);
    public:
	DVR_channel(pqxx::work &db_work, string id, unsigned chan, bool main=false);
	~DVR_channel();
	string get_last_error();
	bool is_connected();
	bool is_main_stream();
	Mat get_frame(double blur_filter=0, bool ignore_size=false);
	void connect();
	void release();
	double get_buffer_size();
};
#endif