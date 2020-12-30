#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <thread>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <optional>
#include <Python.h>

#include "third_party/prettyprint.hpp"


namespace py = pybind11;

using InputExample = std::tuple<std::vector<std::string>, std::vector<cv::Point2f>>;
using Example = std::tuple<std::vector<cv::Mat>, std::vector<cv::Point2f>>;
using OutputExample = std::tuple<std::vector<py::handle>, std::vector<cv::Point2f>>;


void check_example(const Example& example) {
    std::cout << "Checking processed example" << std::endl;
    std::cout << "Number of images in example: " << std::get<0>(example).size()  << std::endl;

    for (auto image : std::get<0>(example)) {
        if (image.empty()) {
            std::cout << "ERROR: empty image" << std::endl;
        }
    }
    std::cout << "Checking successful" << std::endl;
}


py::handle image_to_py_array(const cv::Mat& image) {
    // IMPORTANT: Need to acquire the GIL before touching Python internals :) 
    // Reference: https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
    py::gil_scoped_acquire acquire;
    auto result = py::array_t<uint8_t>(
        {image.size().height, image.size().width, 3},
        {image.size().width * 3, 3, 1},
        image.data);

    return result.release();
}

OutputExample example_to_output_example(const Example& example) {
    std::vector<py::handle> output_arrays;
    for (const cv::Mat& image : std::get<0>(example)) {
        output_arrays.push_back(image_to_py_array(image));
    }

    return {output_arrays, std::get<1>(example)};

}

Example process_example(const InputExample& input_example, py::dict config) {
    std::vector<cv::Mat> output_images;

    for (const std::string& image_path : std::get<0>(input_example)) {
        std::cout << "Loading image from " << image_path << std::endl;
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

        if (image.empty()) {
            throw std::runtime_error("Failed to load image " + image_path);
        }

        output_images.push_back(image);
    }

    return {output_images, std::get<1>(input_example)};
}


void thread_job(int thread_number,
                int thread_count,
                const std::vector<InputExample>& input_examples,
                std::vector<std::optional<Example>>& processed_examples,
                py::dict config) {
    
    std::cout << "Starting thread" << std::endl;
    int input_index = thread_number;

    while (input_index < input_examples.size()) {
        Example processed_example = process_example(input_examples[input_index], config);

        int output_index = input_index % processed_examples.size();
        while (processed_examples[output_index].has_value()) {
            //std::cout << "Worker: Waiting for processed_examples[" << output_index << "]" << std::endl;
            sleep(0.0);
        }

        processed_examples[output_index] = processed_example;  // I am 99% sure here is not a race condition :)
        input_index += thread_count;
    }
    std::cout << "Ending thread" << std::endl;
}


class AugmentationsBackend {
public:
    AugmentationsBackend(const std::vector<InputExample>& input_examples, py::dict config)
        : input_examples(input_examples),
          config(config),
          num_threads(1),
          processed_examples(2)
    {
        std::cout << "Input examples size: " << input_examples.size() << std::endl;
        assert (processed_examples.size() >= num_threads);
        assert (num_threads > 0);
    }

    std::optional<OutputExample> get_example() {
        if (input_index >= input_examples.size())
            return {};  // end of epoch

        int output_index = input_index % processed_examples.size();

        while (!processed_examples[output_index].has_value()) {
            //std::cout << "Main thread: Waiting for processed_examples[" << output_index << "]" << std::endl;
            sleep(0.0);
        }

        Example example = processed_examples[output_index].value();
        processed_examples[output_index].reset();

        input_index++;
        return example_to_output_example(example);
    }

    void start_epoch() {
        input_index = 0;

        std::cout << "C++: Starting epoch" << std::endl;

        for(int i=0; i<num_threads; ++i) {
            std::thread t(thread_job, i, num_threads, std::cref(input_examples), std::ref(processed_examples), config);
            t.detach();
        }
    }


private:
    py::dict config;
    std::vector<InputExample> input_examples;
    int num_threads;
    int input_index;

    std::vector<std::optional<Example>> processed_examples;
};


PYBIND11_MODULE(augmentations_backend, m) {

    Py_Initialize();
    PyEval_InitThreads();

    py::class_<AugmentationsBackend>(m, "AugmentationsBackend")
        .def(py::init<const std::vector<InputExample>&, py::dict>())
        .def("start_epoch", &AugmentationsBackend::start_epoch)
        .def("get_example", &AugmentationsBackend::get_example);
};
