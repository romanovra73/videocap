#include "dvr.h"
#include "../include/myCV.h"


void save_face_big(pqxx::work &work, Mat frame, string date_formated) {
    const char* query = "INSERT INTO video_raw.person_detections (datetime, dvr_id, channel, frame) VALUES(to_timestamp($1, 'YYYY-MM-DD HH24:MI:SS.MS'), $2, 15, $3);";
    std::vector<unsigned char> vec;
    imencode(".jpg", frame, vec);
    work.exec_params(query, date_formated, "SVRD", pqxx::binary_cast(vec));
    work.commit();
}


int main() {
    
    pqxx::connection db(myCV::get_db_connection_string());
    pqxx::work db_work(db);
    DVR_channel sub_stream(db_work, "SVRD", 15);
    
    Mat prev_frame, frame;
    
    while(true) {
	frame = sub_stream.get_frame();
	bool motion_detected = myCV::motion_detected(frame, prev_frame, 10, 1000);
	if (motion_detected) {
	    time_t now = time(nullptr);
	    cout << ctime(&now) << "Motion detected" << endl;
	    motion_detected = myCV::person_detected(frame);
	    if (!motion_detected) {
		prev_frame = frame.clone();
		usleep(100000);
		continue;
	    }
	    sub_stream.release();
	    DVR_channel main_stream(db_work, "SVRD", 15, true);
	    int frame_count=0;
	    const int max_frames = 8;
	    Mat bframe, prev_bframe, best_frame;
	    double blur = 0;
	    double max_blur=0;
	    auto best_frame_time = chrono::system_clock::now();
	    auto big_frame_time = chrono::system_clock::now();
	    while(motion_detected) {
		bframe = main_stream.get_frame();
		big_frame_time = chrono::system_clock::now();
		if (prev_bframe.empty()) {
		    max_blur = myCV::blur_varience(bframe);
		    best_frame = bframe.clone();
		    best_frame_time = big_frame_time;
		    prev_bframe = bframe.clone();
		    frame_count = 1;
		    continue;
		}
		motion_detected = myCV::person_detected(bframe);
		
		if (motion_detected) {
		    frame_count++;
		    blur = myCV::blur_varience(bframe);
		    if (blur>max_blur) {
			max_blur = blur;
			best_frame = bframe.clone();
			best_frame_time = big_frame_time;
		    }
		}
		if (!motion_detected || frame_count>=max_frames) {
		    frame_count = 0;
		    max_blur = 0;
		    time_t ft = chrono::system_clock::to_time_t(best_frame_time);
		    auto duration_since_epoch = best_frame_time.time_since_epoch();
		    auto milliseconds = chrono::duration_cast<chrono::milliseconds>(duration_since_epoch).count() % 1000;
		    stringstream ss;
		    ss << put_time(localtime(&ft), "%Y-%m-%d %H:%M:%S") << "." << setfill('0') << setw(3) << milliseconds;
		    cout << "save frame time: " << ss.str() << endl;
		    save_face_big(db_work, best_frame, ss.str());
		}
		prev_bframe = bframe.clone();
	    }
	    main_stream.release();
	    sub_stream.connect();
	}
	prev_frame = frame.clone();
	usleep(100000);
    }
}
