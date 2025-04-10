// main.cpp
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "imdb_infer");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, "imdb_sentiment.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* input_name = session.GetInputName(0, allocator);
    std::vector<int64_t> input_shape{1, 500};
    std::vector<int32_t> input_tensor_values(500, 1); // فرضی

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    const char* output_name = session.GetOutputName(0, allocator);
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1, &output_name, 1);

    float* float_array = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "📢 Predicted Sentiment = " << float_array[0] << std::endl;

    return 0;
}
