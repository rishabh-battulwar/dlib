// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <iostream>
#include <chrono>
#include <string>

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{
    // try
    // {
        cv::VideoCapture cap(0);
        cap.set(CV_CAP_PROP_FPS, 60);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor shape_predictor_object;
        deserialize("shape_predictor_68_face_landmarks.dat") >> shape_predictor_object;
        int width = 400;
        int height = 225;
        if (argc >= 3) {
            width = std::atoi(argv[1]);
            height = std::atoi(argv[2]);
        }
        
        float report_after_time = 0.0;
        if (argc >= 4) report_after_time = std::atof(argv[3]);
        int frame_count = 0;
        double process_time = 0.0;
        const int extra_crop_pad = 10;
        std::vector<rectangle> faces;
        std::vector<rectangle> prev_faces;
        rectangle pre_pad_crop_boundary;
        rectangle crop_boundary;
        bool keep_cropping = false;
        cv::Rect cropROI;
        cv::Mat temp_orig;
        cv::Mat temp;
        std::chrono::time_point<std::chrono::steady_clock> global_timer;
        int global_count = 0;
        int global_tick = 5;
            
        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {

            if (global_count == 0) {
                global_timer = std::chrono::steady_clock::now();
            }
            global_count++;

            // Grab a frame
            if (!cap.read(temp_orig))
            {
                std::exit(EXIT_FAILURE);
            }
            cv::resize(temp_orig, temp, cv::Size(width, height));

            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            // cv::Mat temp_resized;
            // cv_image<bgr_pixel> cimg(temp_resized);
            cv_image<bgr_pixel> cimg_display(temp);
            cv_image<bgr_pixel> cimg(temp);
            frame_count = 0;

            auto face_detector_func = [&]() {
                prev_faces = faces;
                auto start = std::chrono::steady_clock::now();
                
                // Detect faces 
                faces = std::vector<rectangle>(detector(cimg));
                if (!faces.size()) {
                    faces = prev_faces;
                    return false;
                }

                // slide the window to crop top left corner
                faces[0] = translate_rect(faces[0], cropROI.x, cropROI.y);
                pre_pad_crop_boundary = faces[0];

                frame_count++;
                auto diff = std::chrono::steady_clock::now()-start;
                process_time = std::chrono::duration<double>(diff).count();
                std::cout << "\n----------------------------\n";
                std::cout << "Image Process time (" << faces.size() << ") : " << process_time << " s & " << 1.0 / std::chrono::duration<double>(diff).count() << " fps";
                std::cout << "\n----------------------------\n";
                
                // std::cout << "tl corner : " << faces[0].tl_corner()(0) << " " << faces[0].tl_corner()(1) << "\n";
                // std::cout << "br corner : " << faces[0].br_corner()(0) << " " << faces[0].br_corner()(1) << "\n";
                // std::cout << "box size : " << faces[0].width() << " " << faces[0].height() << "\n";
                return true;
            };
            
            if (keep_cropping) {
                // dlib::array2d<unsigned char> cimggray;
                // dlib::assign_image(cimggray, cimg);
                // std::cout << "Temp image : " << temp.size().width << ", " << temp.size().height << "\n"; 
                std::cout << "Pre-Pad : " << pre_pad_crop_boundary.tl_corner()(0) << ", " << pre_pad_crop_boundary.tl_corner()(1) << " - " << pre_pad_crop_boundary.width() << ", " << pre_pad_crop_boundary.height() << "\n"; 
                std::cout << "CropROI : " << cropROI.x << ", " << cropROI.y << " - " << cropROI.width << ", " << cropROI.height << "\n"; 
                cv::Mat cropped_image = temp(cropROI);
                // std::cout << "Cropped image : " << cropped_image.size().width << ", " << cropped_image.size().height << "\n"; 
                cimg = cv_image<bgr_pixel>(cropped_image);
                // dlib::assign_image(cimggray, cimg);
            }
            else {
                // cv::resize(temp, temp_resized, cv::Size(width,height));
                cimg = cv_image<bgr_pixel>(temp);
            }

            if (!face_detector_func()) {
                std::cout << "\n----------------------------\n";
                std::cout << "Face not found! \n";
                win.clear_overlay();
                win.set_image(cimg_display);
                win.add_overlay(faces[0], rgb_pixel(255,0,0), "fail");
                keep_cropping = false;
                continue;
            }
            else {
                if (keep_cropping == false or true) {
                    int tlx, tly, crop_width, crop_height;
                    if (faces[0].tl_corner()(0) < temp.cols && faces[0].tl_corner()(1) < temp.rows) {
                        tlx = (faces[0].tl_corner()(0) - extra_crop_pad) < 0 ? 0 : faces[0].tl_corner()(0) - extra_crop_pad;
                        tly = (faces[0].tl_corner()(1) - extra_crop_pad) < 0 ? 0 : faces[0].tl_corner()(1) - extra_crop_pad;
                        crop_width = (faces[0].tl_corner()(0) + faces[0].width() + extra_crop_pad) > temp.cols ? (temp.cols - 1 - faces[0].tl_corner()(0)) : (faces[0].width() + 2*extra_crop_pad);
                        crop_height = (faces[0].tl_corner()(1) + faces[0].height() + extra_crop_pad) > temp.rows ? temp.rows - 1 - faces[0].tl_corner()(1) : (faces[0].height() + 2*extra_crop_pad);
                    }
                    else {
                        tlx = 0;
                        tly = 0;
                        crop_width = temp.cols;
                        crop_height = temp.rows;
                    }
                    cropROI = cv::Rect(
                                // faces[0].tl_corner()(0) - extra_crop_pad, 
                                // faces[0].tl_corner()(1) - extra_crop_pad, 
                                // faces[0].width() + extra_crop_pad, 
                                // faces[0].height() + extra_crop_pad);
                                tlx, 
                                tly, 
                                crop_width, 
                                crop_height);
                    crop_boundary = rectangle(tlx, tly, tlx+crop_width, tly+crop_height);
                    keep_cropping = true;
                }
            }
            
            // // Setup a rectangle to define your region of interest
            // cv::Rect myROI(faces[0].tl_corner()(0) - extra_crop_pad, 
            //                faces[0].tl_corner()(1) - extra_crop_pad, 
            //                faces[0].width() + extra_crop_pad, 
            //                faces[0].height() + extra_crop_pad);

            // // Crop the full image to that image contained by the rectangle myROI
            // // Note that this doesn't copy the data
            // cv::Mat cropped_image = temp_resized(myROI);
            // std::cout << "Cropped image : " << cropped_image.size().width << ", " << cropped_image.size().height << "\n"; 
            // cimg = cv_image<bgr_pixel>(cropped_image);
            // // dlib::assign_image(cimggray, cimg);
            // face_detector_func();
            
            // if (process_time > report_after_time) {
            //     std::cout << frame_count << " faces detected in image of size [" << width << ", " << height << "] at "
            //               << process_time / frame_count << " s per image and "
            //               << frame_count / process_time << " fps\n";
            //     frame_count = 0;
            //     process_time = 0.0;
            // }

            auto start = std::chrono::steady_clock::now();
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(shape_predictor_object(cimg_display, faces[i]));
            auto diff = std::chrono::steady_clock::now()-start;
            process_time = std::chrono::duration<double>(diff).count();
            std::cout << "Face Alignment Process time (" << faces.size() << ") : " << process_time << " s & " << 1.0 / std::chrono::duration<double>(diff).count() << " fps";
            std::cout << "\n----------------------------\n";

            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg_display);
            win.add_overlay(render_face_detections(shapes));
            win.add_overlay(faces[0], rgb_pixel(0,255,0), "pass");
            // win.add_overlay(pre_pad_crop_boundary, rgb_pixel(0,255,255), "crop");
            win.add_overlay(crop_boundary, rgb_pixel(0,0,255), "crop");

            if (global_count % global_tick == 0) {
                global_count = 0;
                auto diff = std::chrono::steady_clock::now() - global_timer;
                double proc_time = std::chrono::duration<double>(diff).count();
                std::cout << "Process per " << global_tick << " frames running in " << proc_time << " s & at " << (1.0*global_tick) / std::chrono::duration<double>(diff).count() << " fps";
                std::cout << "\n----------------------------\n";
            }
        }
    // }
    // catch(serialization_error& e)
    // {
    //     cout << "You need dlib's default face landmarking model file to run this example." << endl;
    //     cout << "You can get it from the following URL: " << endl;
    //     cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
    //     cout << endl << e.what() << endl;
    // }
    // catch(exception& e)
    // {
    //     cout << e.what() << endl;
    // }
}
