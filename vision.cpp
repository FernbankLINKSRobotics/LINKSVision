#include <iostream>
#include <time.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

int cx1, cx2, cx, cy1, cy2, cy;
bool isTimed = true;
bool display = false;
bool debug = false;

int main(){

    cv::VideoCapture cap(0);
    clock_t start, end;

    if(debug){
        isTimed = display = debug;
    }

    for(;;){
        // starts time on the cycle
        if(isTimed){ start = clock(); }
        // Take in image from the camera
        cv::Mat cam;
        cap.read(cam);
        // Down res the image
        cv::resize(cam, cam, cv::Size(640, 360), 0, 0, cv::INTER_CUBIC);
        // Convert to HSV for color filtering
        cv::Mat HSV_img;
        cv::cvtColor(cam, HSV_img, cv::COLOR_BGR2HSV);
        // Blur and remove noise
        //cv::GaussianBlur(HSV_img, HSV_img, cv::Size(9,9), 2, 2);
        //cv::erode (HSV_img, HSV_img, cv::Mat());
        //cv::dilate(HSV_img, HSV_img, cv::Mat());
        // Displays colors between two values
        cv::Mat mask;
        cv::inRange(HSV_img, cv::Scalar(20,  100, 50 )
                           , cv::Scalar(100, 190, 140), mask);
        // Slowing Image through filter
        cv::Mat filtered;
        cv::bitwise_and(cam, HSV_img, filtered, mask);
        // Converting to grayscale and finding the threshold
        cv::Mat drawing = cv::Mat::zeros(filtered.size(), CV_8UC3 );
        cv::Mat thr(filtered.rows, filtered.cols, CV_8UC1); 
        cv::cvtColor(filtered, thr, CV_BGR2GRAY);
        cv::threshold(thr, thr, 25, 255, cv::THRESH_BINARY);
        // Finding Contours
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(thr, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        // ---------- Center of the two largest contours ----------
        // Setting initial values
        int c_size = contours.size();
        int fst_area = 1;
        int snd_area = 0;
        if(c_size > 2){
            
            // Initializing contours and hulls for later calulcation 
            std::vector<cv::Point> fst_contour;
            std::vector<cv::Point> snd_contour;
            
            // Finds the largest and second largest contour areas
            for(int i=0; i < c_size; i++){
                int area = (int) cv::contourArea(contours[i]);
                if(area > fst_area){
                    fst_contour = contours[i];
                    fst_area = area;
                }
                if((area < fst_area) && (area > snd_area)){
                    snd_contour = contours[i];
                    snd_area = area;
                }
            }
            // filteres out situatations with no information
            if((fst_area == 0) && snd_area == 0){ continue; }

            // Creates moments from the hulls to find center x and y
            cv::Moments m1 = cv::moments(fst_contour);
            cv::Moments m2 = cv::moments(snd_contour);

            // Finds centers of the hulls and converts to ints
            if((m1.m00 != 0) && (m2.m00 != 0)){
                cx1 = (int) m1.m10 / m1.m00;
                cx2 = (int) m2.m10 / m2.m00;
                cx  = (int) (cx1 + cx2) / 2;

                cy1 = (int) m1.m01 / m1.m00;
                cy2 = (int) m2.m01 / m2.m00;
                cy  = (int) (cy1 + cy2) / 2;
                
                std::cout << "center x: " << cx << "\tcenter y: " << cy << "\n";
            } else {
                std::cout << "m1.m00: " << m1.m00 << "\tm2.m00: " << m2.m00 << "\n";
            }
        } else {
            std::cout << "ERROR\n";
        } 
        // Display images
        if(display){
            cv::drawContours(filtered, contours, -1, cv::Scalar(255, 191, 0), 2);
            cv::circle(filtered, cv::Point(cx2, cy2), 10, cv::Scalar(0, 0, 255), 10);
            cv::imshow("Webcam", filtered);
            cv::waitKey(1);
        }

        // Finds the end time of a loop and displays speed
        if(isTimed){
            end = clock();
            std::cout << "Seconds: " << (double) (end - start)/CLOCKS_PER_SEC << "\n";
        }
    }
}
