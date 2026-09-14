#include "myCV.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>

    string myCV::get_db_connection_string() {
	return "dbname=ml user=videocap password=some-passwd host=localhost port=5432";
    }

    bool myCV::motion_detected(Mat new_frame, Mat prev_frame, double pixel_threshold, double total_threshold) {
	if (prev_frame.empty()) 
	    return false;
	if (new_frame.empty())
	    return false;
	Mat new_gray, prev_gray, diff, dst;
	cvtColor(new_frame, new_gray, COLOR_BGR2GRAY);
	GaussianBlur(new_gray, new_gray, Size(21,21), 0);
	
	cvtColor(prev_frame, prev_gray, COLOR_BGR2GRAY);
	GaussianBlur(prev_gray, prev_gray, Size(21,21), 0);
	
	absdiff(new_gray, prev_gray, diff);
	threshold(diff, dst, pixel_threshold, 1, THRESH_BINARY);
	
	double total = sum(dst).dot(Scalar::ones());
	if (total>=total_threshold)
	    return true;
	return false;
    }
    
    bool myCV::face_detected(Mat frame, CascadeClassifier face_cascade, Size min_size, Size max_size) {
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	vector<Rect> faces;
	face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, min_size, max_size);
	if (faces.size()>0)
	    return true;
	return false;
    }
    
    bool myCV::mark_faces(Mat &frame, CascadeClassifier face_cascade, Size min_size, Size max_size) {
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	vector<Rect> faces;
	//cout << "Try to detect big face " << endl;
	face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, min_size, max_size);
	//cout << "Rect cycle" << endl;
	for (const auto& face : faces) {
	    //cout << "Rectangle " << face.tl() << " x " << face.br() << endl;
	    rectangle(frame, face, Scalar(255,0,255),2);
	}
	if (faces.size()>0)
	    return true;
	return false;
    }
    
    double myCV::blur_varience(Mat frame) {
	Mat gray, laplacian;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	Laplacian(gray, laplacian, CV_64F);
	
	Scalar mean, stddev;
	meanStdDev(laplacian, mean, stddev, Mat());
	return *stddev.val;
    }
    
    static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
    }
    
    bool myCV::person_detected(Mat frame) {
	std::vector<uchar> buffer;
	bool enc = imencode(".jpg", frame, buffer);
	if (enc) {
	    CURL *curl = curl_easy_init();
	    CURLcode res;
	    std::string response;
	    if (curl) {
		curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8000/img_detector/person_detected");
		curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, buffer.size());
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buffer.data());
		
		struct curl_slist *headers = NULL;
		headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
		
		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
		    std::cerr << "Failed to request ML API: " << curl_easy_strerror(res) << endl;
		} else {
		    
		    auto data = nlohmann::json::parse(response);
		    if (data["success"].get<bool>()) {
			curl_slist_free_all(headers);
			curl_easy_cleanup(curl);
			curl_global_cleanup();
			return data["data"].get<bool>();
		    }
		}
		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
		curl_global_cleanup();
	    }
	} else {
	    cerr << "Error encoding frame" << endl;
	}
	return false;
    }
    
    int myCV::npersons_detected(Mat frame) {
	std::vector<uchar> buffer;
	bool enc = imencode(".jpg", frame, buffer);
	if (enc) {
	    CURL *curl = curl_easy_init();
	    CURLcode res;
	    std::string response;
	    if (curl) {
		curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8000/img_detector/nperson_detected");
		curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, buffer.size());
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buffer.data());
		
		struct curl_slist *headers = NULL;
		headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
		
		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
		    std::cerr << "Failed to request ML API: " << curl_easy_strerror(res) << endl;
		} else {
		    
		    auto data = nlohmann::json::parse(response);
		    if (data["success"].get<bool>()) {
			curl_slist_free_all(headers);
			curl_easy_cleanup(curl);
			curl_global_cleanup();
			return data["data"].get<int>();
		    }
		}
		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
		curl_global_cleanup();
	    }
	} else {
	    cerr << "Error encoding frame" << endl;
	}
	return false;
    }
    
    int myCV::pool_npersons_detected(Mat frame) {
	std::vector<uchar> buffer;
	bool enc = imencode(".jpg", frame, buffer);
	if (enc) {
	    CURL *curl = curl_easy_init();
	    CURLcode res;
	    std::string response;
	    if (curl) {
		curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8000/pool_detect/npersons");
		curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, buffer.size());
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buffer.data());
		
		struct curl_slist *headers = NULL;
		headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
		
		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
		    std::cerr << "Failed to request ML API: " << curl_easy_strerror(res) << endl;
		} else {
		    
		    auto data = nlohmann::json::parse(response);
		    if (data["success"].get<bool>()) {
			curl_slist_free_all(headers);
			curl_easy_cleanup(curl);
			curl_global_cleanup();
			return data["data"].get<int>();
		    }
		}
		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
		curl_global_cleanup();
	    }
	} else {
	    cerr << "Error encoding frame" << endl;
	}
	return false;
    }
    
    double myCV::pool_detected(Mat frame) {
	std::vector<uchar> buffer;
	bool enc = imencode(".jpg", frame, buffer);
	if (enc) {
	    CURL *curl = curl_easy_init();
	    CURLcode res;
	    std::string response;
	    if (curl) {
		curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8000/pool_detect");
		curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, buffer.size());
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, buffer.data());
		
		struct curl_slist *headers = NULL;
		headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
		
		res = curl_easy_perform(curl);
		if (res != CURLE_OK) {
		    std::cerr << "Failed to request ML API: " << curl_easy_strerror(res) << endl;
		} else {
		    std::cout << "Response: " << response << std::endl;
		    auto data = nlohmann::json::parse(response);
		    if (data["success"].get<bool>()) {
			curl_slist_free_all(headers);
			curl_easy_cleanup(curl);
			curl_global_cleanup();
			return data["data"].get<double>();
		    }
		}
		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
		curl_global_cleanup();
	    }
	} else {
	    cerr << "Error encoding frame" << endl;
	}
	return false;
    }
    
    void myCV::run_tracking(int channel) {
	CURL *curl = curl_easy_init();
	CURLcode res;
	//std::string response;
	if (curl) {
	    if (channel==1)
		curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8000/ch1_tracker");
	    		
	    struct curl_slist *headers = NULL;
	    headers = curl_slist_append(headers, "Content-Type: application/json");
	    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	
	    //curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
	    //curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
	
	    res = curl_easy_perform(curl);
	    if (res != CURLE_OK) {
	        std::cerr << "Failed to request ML API: " << curl_easy_strerror(res) << endl;
	    } 
	    curl_slist_free_all(headers);
	    curl_easy_cleanup(curl);
	    curl_global_cleanup();
	}
    }
    
