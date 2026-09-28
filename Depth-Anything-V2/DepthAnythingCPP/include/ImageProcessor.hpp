#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <immintrin.h>
#include <string>

class ImageProcessor {
public:
    static constexpr int TARGET_WIDTH = 518;
    static constexpr int TARGET_HEIGHT = 518;
    static constexpr int CHANNELS = 3;

    // Preprocessing
    static void preprocess(const cv::Mat& src, std::vector<float>& dst_tensor);

    // Height Map Postprocessing
    static cv::Mat postprocess(const float* raw_depth, int orig_w, int orig_h);
    static void saveResizedDepthToBin(const float* raw_depth, int orig_w, int orig_h, const std::string& filepath);

    // Confidence Map Postprocessing (NEW)
    static cv::Mat postprocessConfidence(const float* raw_conf, int orig_w, int orig_h);
    static void saveResizedConfidenceToBin(const float* raw_conf, int orig_w, int orig_h, const std::string& filepath);

private:
    // Worker function: processes a slice of rows on a dedicated thread
    static void processRowSlice(const cv::Mat& rgb_img, float* r_plane, float* g_plane, float* b_plane, int start_row, int end_row);
};