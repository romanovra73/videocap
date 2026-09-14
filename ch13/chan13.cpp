#include "../include/dvr.h"
#include "../include/myCV.h"
#include <unistd.h>
#include <chrono>
#include <thread>

const int channel=13;
const int state_ttl =600;
const double proba_threshold = 0.5;

void save_state(pqxx::work &work, bool state, Mat frame, time_t datetime, double proba=0, int persons=0) {
    int int_state = 0;
    if (state)
	int_state =1;
    
    const char* query = "INSERT INTO cv.svrd_pool_detect (datetime, state, data, proba, persons) VALUES(to_timestamp($1), $2, $3, $4, $5);";
    std::vector<unsigned char> vec;
    imencode(".jpg", frame, vec);
    
    work.exec_params(query, datetime, int_state, pqxx::binary_cast(vec), proba, persons);
    work.commit();
}

void get_last_state(pqxx::work &work, bool &state, time_t &since) {
    const char* query = "SELECT EXTRACT(EPOCH FROM datetime)::integer, state FROM cv.svrd_pool_detect ORDER BY datetime DESC LIMIT 1";
    pqxx::result result = work.exec(query);
    if (std::size(result)>0) {
	pqxx::row const row = result[0];
        since = row[0].as<time_t>();
	state = row[1].as<bool>();
    }
}

void check_state_ttl(pqxx::work &work, bool &mstate, time_t &msince, Mat frame) {
    if (!mstate)
	return;
    time_t now = time(nullptr);
    if (now-msince>state_ttl) {
	cout << "Save state: False" << endl;
	save_state(work, false, frame, now);
	mstate = false;
	msince = now;
    }
}

int main() {
    
    pqxx::connection db(myCV::get_db_connection_string());
    pqxx::work  db_work(db);
    
    cout << "DB connected" << endl;
    
    bool state = false;
    time_t since = time(nullptr)-86400;
    get_last_state(db_work, state, since);
    
    cout << "Last state: " << state << "  " << since << endl;
    
    DVR_channel sub_stream(db_work, "SVRD", channel);
    
    Mat prev_frame, frame, best_frame;
    int detections = 0;
    int persons = 0;
    int det_persons = 0;
    double max_blur = 0;
    double max_proba = 0.0;
    int cur_persons = 0;
    int bf_persons = 0;
    cout << "Start loop: " << endl;
    while(true) {
	frame = sub_stream.get_frame();
	bool motion_detected = myCV::motion_detected(frame, prev_frame, 10, 500);
	if (motion_detected) {
	    time_t now = time(nullptr);
	    cout << ctime(&now) << "Motion detected" << endl;
	    int persons_detected = myCV::pool_npersons_detected(frame);
	    if (!persons_detected) {
		check_state_ttl(db_work, state, since, frame);
		prev_frame = frame.clone();
		std::this_thread::sleep_for(chrono::seconds(1));
		detections = 0;
		persons = 0;
		det_persons = 0;
		max_blur = 0;
		max_proba = 0.0;
		continue;
	    }
	    cout << "Detected persons: " << persons_detected << endl;
	    double proba = myCV::pool_detected(frame);
	    cout << "Pool probability: " << proba << endl;
	    if (proba<proba_threshold) {
		check_state_ttl(db_work, state, since, frame);
		prev_frame = frame.clone();
		std::this_thread::sleep_for(chrono::seconds(1));
		detections = 0;
		persons = 0;
		max_blur = 0;
		max_proba = 0.0;
		det_persons = 0;
		continue;
	    }
	    if (persons_detected!=persons) {
		persons = persons_detected;
		det_persons = 1;
	    } else {
		det_persons++;
	    }
	    if (!state) {
		detections++;
		double blur = myCV::blur_varience(frame);
		if (blur>max_blur) {
		    max_blur = blur;
		    best_frame = frame.clone();
		    bf_persons = persons_detected;
		}
		if (max_proba<proba) {
		    max_proba = proba;
		}
	    } else {
		since = time(nullptr);
		prev_frame = frame.clone();
		if (cur_persons!=persons && det_persons>=2) {
		    save_state(db_work, true, frame, since, proba, persons);
		    cur_persons = persons;
		}
		std::this_thread::sleep_for(chrono::seconds(1));
		continue;
	    }
	    if (detections>=3) {
		state = true;
		since = time(nullptr);
		cout << "Save state: True" << endl;
		save_state(db_work, true, best_frame, since, max_proba, bf_persons);
		cur_persons = persons_detected;
		std::this_thread::sleep_for(chrono::seconds(1));
	    }
	} else {
	    check_state_ttl(db_work, state, since, frame);
	}
	prev_frame = frame.clone();
	sleep(1);
    } 
}
