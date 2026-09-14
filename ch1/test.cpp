#include "../include/dvr.h"
#include "../include/myCV.h"

void save_face_frame_small(pqxx::work &work, Mat frame) {
    const char* query = "INSERT INTO video_raw.face_detect_small (frame) VALUES($1);";
    std::vector<unsigned char> vec;
    imencode(".jpg", frame, vec);
    work.exec_params(query, pqxx::binary_cast(vec));
    work.commit();
}

void save_face_big(pqxx::work &work, Mat frame, string date_formated) {
    const char* query = "INSERT INTO video_raw.face_detect_big (datetime, frame) VALUES(to_timestamp($1, 'YYYY-MM-DD HH24:MI:SS.MS'), $2);";
    std::vector<unsigned char> vec;
    imencode(".jpg", frame, vec);
    work.exec_params(query, date_formated, pqxx::binary_cast(vec));
    work.commit();
}


int main() {
    myCV::run_tracking(1);

}
