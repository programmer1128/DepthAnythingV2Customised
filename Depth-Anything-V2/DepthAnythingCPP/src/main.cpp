#include <onnxruntime_cxx_api.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
constexpr uint32_t NDSM_MAGIC = 0xDEADBEEF;
constexpr uint32_t MODEL_WIDTH = 518;
constexpr uint32_t MODEL_HEIGHT = 518;
constexpr uint32_t INPUT_CHANNELS = 3;
constexpr uint32_t OUTPUT_CHANNELS = 2;

#pragma pack(push, 1)
struct NdsmRequestHeader
{
    uint32_t magic;
    uint32_t tileId;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint64_t payloadLen;
};

struct NdsmResponseHeader
{
    uint32_t magic;
    uint32_t tileId;
    uint32_t status;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint64_t payloadLen;
};
#pragma pack(pop)

static_assert(sizeof(NdsmRequestHeader) == 28);
static_assert(sizeof(NdsmResponseHeader) == 32);

class ScopedSocket
{
public:
    explicit ScopedSocket(int descriptor) : descriptor_(descriptor) {}
    ScopedSocket(const ScopedSocket&) = delete;
    ScopedSocket& operator=(const ScopedSocket&) = delete;

    ~ScopedSocket()
    {
        if (descriptor_ >= 0)
            ::close(descriptor_);
    }

    int get() const noexcept { return descriptor_; }

private:
    int descriptor_{-1};
};

uint64_t checkedMultiply(uint64_t lhs, uint64_t rhs, const char* field)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
        throw std::overflow_error(std::string("Overflow calculating ") + field);
    return lhs * rhs;
}

void receiveExact(int socketDescriptor, void* destination, size_t length)
{
    auto* bytes = static_cast<uint8_t*>(destination);
    size_t receivedTotal = 0;

    while (receivedTotal < length)
    {
        const ssize_t received = ::recv(
            socketDescriptor,
            bytes + receivedTotal,
            length - receivedTotal,
            0);

        if (received > 0)
        {
            receivedTotal += static_cast<size_t>(received);
            continue;
        }
        if (received < 0 && errno == EINTR)
            continue;
        throw std::runtime_error("Connection closed while receiving request bytes.");
    }
}

void sendExact(int socketDescriptor, const void* source, size_t length)
{
    const auto* bytes = static_cast<const uint8_t*>(source);
    size_t sentTotal = 0;

    while (sentTotal < length)
    {
        const ssize_t sent = ::send(
            socketDescriptor,
            bytes + sentTotal,
            length - sentTotal,
            MSG_NOSIGNAL);

        if (sent > 0)
        {
            sentTotal += static_cast<size_t>(sent);
            continue;
        }
        if (sent < 0 && errno == EINTR)
            continue;
        throw std::runtime_error("Connection closed while sending response bytes.");
    }
}

void sendErrorResponse(
    int socketDescriptor,
    uint32_t tileId,
    uint32_t width,
    uint32_t height) noexcept
{
    try
    {
        const NdsmResponseHeader response{
            NDSM_MAGIC,
            tileId,
            1,
            width,
            height,
            0,
            0
        };
        sendExact(socketDescriptor, &response, sizeof(response));
    }
    catch (...)
    {
    }
}

void configureClientSocket(int socketDescriptor)
{
    const int enabled = 1;
    if (::setsockopt(
            socketDescriptor,
            IPPROTO_TCP,
            TCP_NODELAY,
            &enabled,
            sizeof(enabled)) != 0)
    {
        throw std::runtime_error("Failed to configure TCP_NODELAY.");
    }

    const timeval timeout{30, 0};
    if (::setsockopt(socketDescriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        ::setsockopt(socketDescriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0)
    {
        throw std::runtime_error("Failed to configure worker socket timeouts.");
    }
}

void validateRequest(const NdsmRequestHeader& header)
{
    if (header.magic != NDSM_MAGIC)
        throw std::invalid_argument("Invalid nDSM request magic.");
    if (header.width != MODEL_WIDTH ||
        header.height != MODEL_HEIGHT ||
        header.channels != INPUT_CHANNELS)
    {
        throw std::invalid_argument("nDSM request violates the model tensor shape.");
    }

    const uint64_t pixels = checkedMultiply(header.width, header.height, "pixel count");
    const uint64_t elements = checkedMultiply(pixels, header.channels, "RGB elements");
    const uint64_t expectedBytes = checkedMultiply(elements, sizeof(float), "RGB payload bytes");
    if (header.payloadLen != expectedBytes)
        throw std::invalid_argument("nDSM RGB payload length does not match float32 CHW tensor.");
}

void processClient(
    int clientDescriptor,
    Ort::Session& session,
    const std::string& inputName,
    const std::string& outputName,
    std::mutex& inferenceMutex)
{
    ScopedSocket client(clientDescriptor);
    NdsmRequestHeader header{};
    bool headerReceived = false;
    bool responseStarted = false;

    try
    {
        configureClientSocket(client.get());
        receiveExact(client.get(), &header, sizeof(header));
        headerReceived = true;
        validateRequest(header);

        const size_t pixelCount =
            static_cast<size_t>(header.width) * static_cast<size_t>(header.height);
        std::vector<float> inputTensorData(pixelCount * INPUT_CHANNELS);
        receiveExact(client.get(), inputTensorData.data(), header.payloadLen);

        // The backend sends one validity byte per pixel after the RGB tensor.
        // The model does not consume it, but draining and validating it keeps
        // the request framing deterministic.
        std::vector<uint8_t> validityMask(pixelCount);
        receiveExact(client.get(), validityMask.data(), validityMask.size());
        for (uint8_t value : validityMask)
        {
            if (value != 0 && value != 1)
                throw std::invalid_argument("nDSM request mask is not binary.");
        }

        std::cout << ">> Received nDSM tile " << header.tileId << '\n';
        const auto started = std::chrono::steady_clock::now();

        Ort::MemoryInfo memoryInfo =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::vector<int64_t> inputShape{
            1,
            INPUT_CHANNELS,
            MODEL_HEIGHT,
            MODEL_WIDTH
        };
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorData.data(),
            inputTensorData.size(),
            inputShape.data(),
            inputShape.size());

        const char* inputNames[]{inputName.c_str()};
        const char* outputNames[]{outputName.c_str()};
        std::vector<Ort::Value> outputs;
        {
            std::lock_guard<std::mutex> lock(inferenceMutex);
            outputs = session.Run(
                Ort::RunOptions{nullptr},
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1);
        }

        if (outputs.size() != 1 || !outputs[0].IsTensor())
            throw std::runtime_error("nDSM model returned an invalid output collection.");

        const auto outputInfo = outputs[0].GetTensorTypeAndShapeInfo();
        const size_t expectedOutputElements = pixelCount * OUTPUT_CHANNELS;
        if (outputInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            outputInfo.GetElementCount() != expectedOutputElements)
        {
            throw std::runtime_error("nDSM model output is not float32 [1,2,518,518].");
        }

        const uint64_t outputBytes =
            static_cast<uint64_t>(expectedOutputElements) * sizeof(float);
        const NdsmResponseHeader response{
            NDSM_MAGIC,
            header.tileId,
            0,
            header.width,
            header.height,
            OUTPUT_CHANNELS,
            outputBytes
        };

        responseStarted = true;
        sendExact(client.get(), &response, sizeof(response));
        sendExact(
            client.get(),
            outputs[0].GetTensorData<float>(),
            static_cast<size_t>(outputBytes));

        const auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started);
        std::cout << "<< Returned nDSM tile " << header.tileId
                  << " | Latency: " << latency.count() << " ms\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "nDSM worker rejected tile "
                  << (headerReceived ? std::to_string(header.tileId) : std::string("unknown"))
                  << ": " << error.what() << '\n';

        if (headerReceived && header.magic == NDSM_MAGIC && !responseStarted)
            sendErrorResponse(client.get(), header.tileId, header.width, header.height);
    }
}
}

int main(int argc, char** argv)
{
    const std::string modelPath = argc > 1
        ? argv[1]
        : "/home/aritra/Desktop/OptimisingDepthAnythingV2/Depth-Anything-V2/DepthAnythingCPP/model_a_metric_ndsm.onnx";

    try
    {
        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "DepthWizardNdsmWorker");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::unordered_map<std::string, std::string> providerOptions;
        providerOptions["device_type"] = "CPU";
        sessionOptions.AppendExecutionProvider("OpenVINO", providerOptions);

        std::cout << "Compiling Height Model...\n";
        Ort::Session session(environment, modelPath.c_str(), sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;
        const auto inputNameAllocation = session.GetInputNameAllocated(0, allocator);
        const auto outputNameAllocation = session.GetOutputNameAllocated(0, allocator);
        const std::string inputName = inputNameAllocation.get();
        const std::string outputName = outputNameAllocation.get();

        const int serverDescriptor = ::socket(AF_INET, SOCK_STREAM, 0);
        if (serverDescriptor < 0)
            throw std::runtime_error("Failed to create nDSM listening socket.");
        ScopedSocket server(serverDescriptor);

        const int reuseAddress = 1;
        if (::setsockopt(
                server.get(),
                SOL_SOCKET,
                SO_REUSEADDR,
                &reuseAddress,
                sizeof(reuseAddress)) != 0)
        {
            throw std::runtime_error("Failed to configure nDSM listening socket.");
        }

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(9092);

        if (::bind(server.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
            throw std::runtime_error(std::string("Failed to bind nDSM worker: ") + std::strerror(errno));
        if (::listen(server.get(), 32) != 0)
            throw std::runtime_error("Failed to listen on nDSM worker socket.");

        std::cout << "Height Model Ready! Listening on 127.0.0.1:9092...\n";
        std::mutex inferenceMutex;

        while (true)
        {
            const int clientDescriptor = ::accept(server.get(), nullptr, nullptr);
            if (clientDescriptor < 0)
            {
                if (errno == EINTR)
                    continue;
                std::cerr << "nDSM accept failed: " << std::strerror(errno) << '\n';
                continue;
            }

            std::thread(
                processClient,
                clientDescriptor,
                std::ref(session),
                inputName,
                outputName,
                std::ref(inferenceMutex))
                .detach();
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "Fatal nDSM worker error: " << error.what() << '\n';
        return 1;
    }
}
