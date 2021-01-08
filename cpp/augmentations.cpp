
#include <thread>
#include <cassert>
#include <optional>


#include "third_party/prettyprint.hpp"
#include "header.hpp"


namespace py = pybind11;


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

    std::vector<std::tuple<float, float>> points;
    for (const cv::Point2f& point : std::get<1>(example)) {
        points.push_back({point.x, point.y});
    }

    return {output_arrays, points};
}

Example load_example(const InputExample& input_example) {
    std::vector<cv::Mat> output_images;
    for (const std::string& image_path : std::get<0>(input_example)) {
        std::cout << "Loading image from " << image_path << std::endl;
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

        if (image.empty()) {
            throw std::runtime_error("Failed to load image " + image_path);
        }

        output_images.push_back(image);
    }

    std::vector<cv::Point2f> points;
    for (const std::tuple<float, float>& point : std::get<1>(input_example)) {
        points.push_back({std::get<0>(point), std::get<1>(point)});
    }

    return {output_images, points};
}


void thread_job(int thread_number,
                int thread_count,
                const std::vector<InputExample>& input_examples,
                std::vector<std::optional<Example>>& processed_examples,
                const std::vector<std::shared_ptr<Transformation>>& transformations) {
    
    std::cout << "Starting thread" << std::endl;
    int input_index = thread_number;

    while (input_index < input_examples.size()) {
        Example example = load_example(input_examples[input_index]);
        apply_transformations(example, transformations);

        int output_index = input_index % processed_examples.size();
        while (processed_examples[output_index].has_value()) {
            //std::cout << "Worker: Waiting for processed_examples[" << output_index << "]" << std::endl;
            sleep(0.0);
        }

        processed_examples[output_index] = example;  // I am 99% sure here is not a race condition :)
        input_index += thread_count;
    }
    std::cout << "Ending thread" << std::endl;
}


class AugmentationsBackend {
public:
    AugmentationsBackend(const std::vector<InputExample>& input_examples, const py::dict& config, const py::list& transformations_list)
        : input_examples(input_examples),
          config(config),
          num_threads(config["num_threads"].cast<int>()),
          processed_examples(config["output_queue_size"].cast<int>())
    {
        std::cout << "Input examples size: " << input_examples.size() << std::endl;

        if (processed_examples.size() < num_threads) throw_exception("Output queue size must be higher than number of threads");
        if (num_threads <= 0) throw_exception("Need at least oen thread");
        if (processed_examples.size() % num_threads != 0) throw_exception("Output queue size must be divisible by number of threads");

        for (const py::handle& item : transformations_list) {
            std::cout << "Converting transformation" << std::endl;
            std::shared_ptr<Transformation> transformation = item.cast<std::shared_ptr<Transformation>>();
            transformations.push_back(transformation);
        } 
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
            std::thread t(thread_job, i, num_threads, std::cref(input_examples), std::ref(processed_examples), transformations);
            t.detach();
        }
    }


private:
    py::dict config;
    std::vector<std::shared_ptr<Transformation>> transformations;
    std::vector<InputExample> input_examples;
    int num_threads;
    int input_index;

    std::vector<std::optional<Example>> processed_examples;
};


PYBIND11_MODULE(augmentations_backend, m) {

    Py_Initialize();
    PyEval_InitThreads();

    py::class_<AugmentationsBackend>(m, "AugmentationsBackend")
        .def(py::init<const std::vector<InputExample>&, const py::dict&, const py::list&>())
        .def("start_epoch", &AugmentationsBackend::start_epoch)
        .def("get_example", &AugmentationsBackend::get_example);

    py::class_<Transformation, std::shared_ptr<Transformation>>(m, "Transformation")
        .def(py::init<double, bool>());

    py::class_<Flip, Transformation, std::shared_ptr<Flip>>(m, "Flip")
        .def(py::init<double, bool>());

    py::class_<Crop, Transformation, std::shared_ptr<Crop>>(m, "Crop")
        .def(py::init<double, double>());

    py::class_<Rotate, Transformation, std::shared_ptr<Rotate>>(m, "Rotate")
        .def(py::init<double, double, double>());

    py::class_<BrightnessContrast, Transformation, std::shared_ptr<BrightnessContrast>>(m, "BrightnessContrast")
        .def(py::init<double, int, int, double, double>());

    py::class_<HueSaturation, Transformation, std::shared_ptr<HueSaturation>>(m, "HueSaturation")
        .def(py::init<double, int, int, int, int>());

    py::class_<Resize, Transformation, std::shared_ptr<Resize>>(m, "Resize")
        .def(py::init<int, int>());

    py::class_<BGR2RGB, Transformation, std::shared_ptr<BGR2RGB>>(m, "BGR2RGB")
        .def(py::init<>());
};
