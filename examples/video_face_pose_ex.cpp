// #define USE_DLIB_WINDOW
#define USE_COLOR_IMG

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#include <dlib/image_io.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>
#include <dlib/pixel.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef USE_DLIB_WINDOW
#include <dlib/gui_widgets.h>
#endif

using namespace dlib;
using namespace std;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Please provide at least the first two args : \n";
    std::cout << "./video_face_pose_ex <input-video-file> "
                 "<path-to-shape-predictor-data-file> [resize-width] "
                 "[resize-height]\n";
    return 1;
  }

  cv::VideoCapture cap;
  bool is_reading_video = false;
  if (std::string(argv[1]) == "camera") {
    cap = cv::VideoCapture(0);
  } else {
    cap = cv::VideoCapture(argv[1]);
    is_reading_video = true;
  }

  if (!cap.isOpened()) {
    cerr << "Unable to open the video" << endl;
    return 1;
  }

  // Load face detection and pose estimation models.
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor shape_predictor_object;
  try {
    deserialize(std::string(argv[2])) >> shape_predictor_object;
  } catch (const std::exception &e) {
    std::cout << "Deserializing the shape predictor failed!\n"
              << e.what() << "\n";
    return 1;
  }

  int width = 400;
  int height = 225;
  if (argc > 3) {
    width = std::atoi(argv[3]);
  }
  if (argc > 4) {
    height = std::atoi(argv[4]);
  }
  int extra_crop_pad = 20;
  if (argc > 5) {
    extra_crop_pad = std::atoi(argv[5]);
  }
  bool update_crop_roi_each_frame = false;
  if (argc > 6) {
    if (std::atoi(argv[6]))
      update_crop_roi_each_frame = true;
  }
  bool dont_crop_override = true;
  if (argc > 7) {
    if (std::atoi(argv[7]))
      dont_crop_override = true;
  }
  std::string output_dir = "/var/tmp/dump";
  if (argc > 8) {
    output_dir = std::string(argv[8]);
  }
  std::cout << "Writing images out to " << output_dir << "\n";

#ifdef USE_DLIB_WINDOW
  image_window win;
#endif

  int frame_count = 0;
  double process_time = 0.0;
  std::vector<rectangle> faces;
  std::vector<rectangle> prev_faces;
  rectangle pre_pad_crop_boundary;
  rectangle crop_boundary;
  bool keep_cropping = false;
  cv::Rect cropROI;
  cv::Mat temp_orig;
  cv::Mat temp;
  cv::Mat temp_gray;
#ifdef USE_COLOR_IMG
  cv_image<bgr_pixel> cimg;
#else
  cv_image<uchar> cimg;
  // cv_image<bgr_pixel> cimg;
  // dlib::array2d<unsigned char> cimg_gray;
#endif
  std::chrono::time_point<std::chrono::steady_clock> global_timer;
  int global_count = 0;
  int global_tick = 5;

  // Grab and process frames until the main window is closed by the user.
  while (cap.read(temp_orig)) {
#ifdef USE_DLIB_WINDOW
    if (win.is_closed()) {
      break;
    }
#endif

    frame_count++;

    if (global_count == 0) {
      global_timer = std::chrono::steady_clock::now();
    }
    global_count++;

    cv::resize(temp_orig, temp, cv::Size(width, height));

    // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
    // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
    // long as temp is valid.  Also don't do anything to temp that would cause
    // it to reallocate the memory which stores the image as that will make cimg
    // contain dangling pointers.  This basically means you shouldn't modify
    // temp while using cimg.
    cv_image<bgr_pixel> cimg_display(temp);
#ifndef USE_COLOR_IMG
    cv::cvtColor(temp, temp_gray, cv::COLOR_BGR2GRAY);
#endif

    auto face_detector_func = [&]() {
      prev_faces = faces;
      auto start = std::chrono::steady_clock::now();

      // Detect faces
      faces = std::vector<rectangle>(detector(cimg));
      // #ifdef USE_COLOR_IMG
      //             faces = std::vector<rectangle>(detector(cimg));
      // #else
      //             faces = std::vector<rectangle>(detector(cimg_gray));
      // #endif
      if (!faces.size()) {
        faces = prev_faces;
        return false;
      }

      // slide the window to crop top left corner
      faces[0] = translate_rect(faces[0], cropROI.x, cropROI.y);
      pre_pad_crop_boundary = faces[0];

      auto diff = std::chrono::steady_clock::now() - start;
      process_time = std::chrono::duration<double>(diff).count();
      std::cout << "\n----------------------------\n";
      std::cout << "Face Detection time (" << faces.size()
                << ") : " << process_time << " s & "
                << 1.0 / std::chrono::duration<double>(diff).count() << " fps";
      std::cout << "\n----------------------------\n";

      // std::cout << "tl corner : " << faces[0].tl_corner()(0) << " " <<
      // faces[0].tl_corner()(1) << "\n"; std::cout << "br corner : " <<
      // faces[0].br_corner()(0) << " " << faces[0].br_corner()(1) << "\n";
      // std::cout << "box size : " << faces[0].width() << " " <<
      // faces[0].height() << "\n";
      return true;
    };

    if (dont_crop_override) {
      keep_cropping = false;
      cropROI = cv::Rect(0, 0, temp.size().width, temp.size().height);
    }

    if (keep_cropping) {
      std::cout << "Pre-Pad : " << pre_pad_crop_boundary.tl_corner()(0) << ", "
                << pre_pad_crop_boundary.tl_corner()(1) << " - "
                << pre_pad_crop_boundary.width() << ", "
                << pre_pad_crop_boundary.height() << "\n";
      std::cout << "CropROI : " << cropROI.x << ", " << cropROI.y << " - "
                << cropROI.width << ", " << cropROI.height << "\n";

#ifdef USE_COLOR_IMG
      cv::Mat cropped_image = temp(cropROI);
      cimg = cv_image<bgr_pixel>(cropped_image);
#else
      cv::Mat cropped_image = temp_gray(cropROI);
      cimg = cv_image<uchar>(cropped_image);
      // cv::Mat cropped_image = temp(cropROI);
      // cimg = cv_image<bgr_pixel>(cropped_image);
      // dlib::assign_image(cimg_gray, cimg);
#endif
    } else {
#ifdef USE_COLOR_IMG
      cimg = cv_image<bgr_pixel>(temp);
#else
      cimg = cv_image<uchar>(temp_gray);
      // cimg = cv_image<bgr_pixel>(temp);
      // dlib::assign_image(cimg_gray, cimg);
#endif
    }

    if (!face_detector_func()) {
      std::cout << "\n----------------------------\n";
      std::cout << "Face not found! \n";
#ifdef USE_DLIB_WINDOW
      win.clear_overlay();
      win.set_image(cimg_display);
      win.add_overlay(faces[0], rgb_pixel(255, 0, 0), "fail");
#endif
      keep_cropping = false;
      continue;
    } else {
      if (keep_cropping == false or update_crop_roi_each_frame) {
        int tlx, tly, crop_width, crop_height;
        if (faces[0].tl_corner()(0) < temp.cols &&
            faces[0].tl_corner()(1) < temp.rows) {
          tlx = (faces[0].tl_corner()(0) - extra_crop_pad) < 0
                    ? 0
                    : faces[0].tl_corner()(0) - extra_crop_pad;
          tly = (faces[0].tl_corner()(1) - extra_crop_pad) < 0
                    ? 0
                    : faces[0].tl_corner()(1) - extra_crop_pad;
          crop_width = (faces[0].tl_corner()(0) + faces[0].width() +
                        extra_crop_pad) > temp.cols
                           ? (temp.cols - 1 - faces[0].tl_corner()(0))
                           : (faces[0].width() + 2 * extra_crop_pad);
          crop_height = (faces[0].tl_corner()(1) + faces[0].height() +
                         extra_crop_pad) > temp.rows
                            ? temp.rows - 1 - faces[0].tl_corner()(1)
                            : (faces[0].height() + 2 * extra_crop_pad);
        } else {
          tlx = 0;
          tly = 0;
          crop_width = temp.cols;
          crop_height = temp.rows;
        }
        cropROI = cv::Rect(tlx, tly, crop_width, crop_height);
        crop_boundary =
            rectangle(tlx, tly, tlx + crop_width, tly + crop_height);
        keep_cropping = true;
      }
    }

    auto start = std::chrono::steady_clock::now();
    // Find the pose of each face.
    std::vector<full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i)
      shapes.push_back(shape_predictor_object(cimg_display, faces[i]));
    auto diff = std::chrono::steady_clock::now() - start;
    process_time = std::chrono::duration<double>(diff).count();
    std::cout << "Face Alignment Process time (" << faces.size()
              << ") : " << process_time << " s & "
              << 1.0 / std::chrono::duration<double>(diff).count() << " fps";
    std::cout << "\n----------------------------\n";

#ifdef USE_DLIB_WINDOW
    // Display it all on the screen
    win.clear_overlay();
    win.set_image(cimg_display);
    win.add_overlay(render_face_detections(shapes));
    win.add_overlay(faces[0], rgb_pixel(0, 255, 0), "pass");
    // win.add_overlay(pre_pad_crop_boundary, rgb_pixel(0,255,255), "crop");
    win.add_overlay(crop_boundary, rgb_pixel(0, 0, 255), "crop");
#endif

    dlib::draw_rectangle(cimg_display, faces[0], dlib::rgb_pixel(0, 255, 0), 1);
    for (const auto &s : shapes) {
      for (size_t idx = 0; idx < s.num_parts(); ++idx) {
        dlib::draw_solid_circle(cimg_display, s.part(idx), 1.0,
                                dlib::rgb_pixel(0, 0, 255));
      }
    }

    char out_file_name[80];
    std::snprintf(out_file_name, sizeof(out_file_name), "/img_%05d.png",
                  frame_count);
    dlib::save_png(cimg_display, output_dir + out_file_name);

    if (global_count % global_tick == 0) {
      global_count = 0;
      auto diff = std::chrono::steady_clock::now() - global_timer;
      double proc_time = std::chrono::duration<double>(diff).count();
      std::cout << "Process per " << global_tick << " frames running in "
                << proc_time << " s & at "
                << (1.0 * global_tick) /
                       std::chrono::duration<double>(diff).count()
                << " fps";
      std::cout << "\n----------------------------\n";
    }
  }

  cap.release();
  return 0;
}
