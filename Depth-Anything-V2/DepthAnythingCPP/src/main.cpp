#include <iostream>
#include <chrono>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <onnxruntime_cxx_api.h>

#include "ImageProcessor.hpp"

#pragma pack(push, 1) 
struct FrameHeader 
{
     uint32_t magic;         
     uint32_t tile_id;       
     uint32_t width;         
     uint32_t height;        
     uint32_t channels;      
     uint32_t dtype;         
     uint64_t payload_len;   
};
#pragma pack(pop)

int main() 
{
     std::string model_path = "/home/aritra/Desktop/OptimisingDepthAnythingV2/Depth-Anything-V2/DepthAnythingCPP/depth_anything_v2_vitl.onnx";

     // 1. Initialize ONNX Session globally so all threads share it safely
     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DepthAnythingV2Engine");
     Ort::SessionOptions session_options;
     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
     
     std::unordered_map<std::string, std::string> ov_options;
     ov_options["device_type"] = "CPU"; 
     session_options.AppendExecutionProvider("OpenVINO", ov_options);
     
     std::cout << "Compiling model... \n";
     Ort::Session session(env, model_path.c_str(), session_options);
     
     Ort::AllocatorWithDefaultOptions allocator;
     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
     std::string input_name = input_name_ptr.get();
     std::string output_name = output_name_ptr.get();

     // 2. TCP SERVER SETUP (Port 9092)
     int server_fd = socket(AF_INET, SOCK_STREAM, 0);
     int opt = 1;
     setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

     struct sockaddr_in address;
     address.sin_family = AF_INET;
     address.sin_addr.s_addr = INADDR_ANY;
     address.sin_port = htons(9092); 

     bind(server_fd, (struct sockaddr*)&address, sizeof(address));
     listen(server_fd, 10); 

     std::cout << "Model Ready! Local Server Listening on 127.0.0.1:9092...\n\n";

     std::mutex inference_mutex;
     while (true) 
     {
         int client_socket = accept(server_fd, nullptr, nullptr);
         if (client_socket < 0) continue;

         // Spin up a thread. Pass session and names by reference, client_socket by value.
        std::thread([client_socket, &session, input_name, output_name, &inference_mutex]() {
             FrameHeader header;
             ssize_t bytes_read = recv(client_socket, &header, sizeof(FrameHeader), MSG_WAITALL);
             
             if (bytes_read != sizeof(FrameHeader) || header.magic != 0xDEADBEEF) {
                 close(client_socket);
                 return;
             }

             std::cout << ">> Received Tile ID: " << header.tile_id << "\n";

             std::vector<uint8_t> raw_img(header.payload_len);
             recv(client_socket, raw_img.data(), header.payload_len, MSG_WAITALL);

             auto start = std::chrono::high_resolution_clock::now();

             // Multithreaded Preprocessing
             std::vector<float> input_tensor_data;
             cv::Mat raw_rgb(header.height, header.width, CV_8UC3, raw_img.data());
             cv::Mat raw_bgr;
             cv::cvtColor(raw_rgb, raw_bgr, cv::COLOR_RGB2BGR);
             ImageProcessor::preprocess(raw_bgr, input_tensor_data);

             // Create Tensor
             Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
             std::vector<int64_t> input_shape = {1, 3, 518, 518};
             Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                 memory_info,
                 input_tensor_data.data(),
                 input_tensor_data.size(),
                 input_shape.data(),
                 input_shape.size()
             );

             const char* in_names[] = {input_name.c_str()};
             const char* out_names[] = {output_name.c_str()};

             // CRITICAL FIX: Run inference and HOLD ONTO output_tensors in this thread's scope!
             std::vector<Ort::Value> output_tensors;
             {
                 std::lock_guard<std::mutex> lock(inference_mutex);
                 output_tensors = session.Run(
                     Ort::RunOptions{nullptr},
                     in_names,
                     &input_tensor,
                     1,
                     out_names,
                     1
                 );
             } // The lock automatically releases here, allowing the next thread to process!

             // Now the pointer is perfectly safe, because output_tensors lives until the end of this lambda block!
             const float* raw_depth = output_tensors[0].GetTensorMutableData<float>();

             size_t expected_return_bytes = header.width * header.height * sizeof(float);
             send(client_socket, raw_depth, expected_return_bytes, MSG_NOSIGNAL);

             close(client_socket);

             auto end = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
             std::cout << "<< Returned Tile ID: " << header.tile_id << " | Latency: " << duration << " ms\n";
         }).detach();
     }
     return 0;
}