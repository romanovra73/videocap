#include "../include/dvr.h"
#include "../include/myCV.h"
#include <unistd.h>
const int channel = 12;

void save_frame(pqxx::work &work, Mat frame, time_t datetime) {
    cout << "Save frane" << endl;
    const char* query = "INSERT INTO video_raw.person_detections (dvr_id, channel, frame) VALUES($1, $2, $3);";
    std::vector<unsigned char> vec;
    imencode(".jpg", frame, vec);
    work.exec_params(query, "SVRD", channel, pqxx::binary_cast(vec));
    work.commit();
}


int main() {
    
    pqxx::connection db(myCV::get_db_connection_string());
    pqxx::work db_work(db);
    DVR_channel sub_stream(db_work, "SVRD", channel);
    
    Mat prev_frame, frame;
    bool motion_detected = false;
    while(true) {
	frame = sub_stream.get_frame();
	if (!motion_detected)
	    motion_detected = myCV::motion_detected(frame, prev_frame, 10, 500);
	if (motion_detected) {
	    motion_detected = myCV::person_detected(frame);
	    if (!motion_detected) {
		prev_frame = frame.clone();
		usleep(1000000);
		continue;
	    }
	    time_t now = time(nullptr);
	    save_frame(db_work, frame, now);
	    prev_frame = frame.clone();
	    continue;
	} else {
	    prev_frame = frame.clone();
	    usleep(100000);
	}
    }
}
