#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


namespace py = pybind11;


using InputExample = std::tuple<std::vector<std::string>, std::vector<std::tuple<float, float>>>;
using Example = std::tuple<std::vector<cv::Mat>, std::vector<cv::Point2f>>;
using OutputExample = std::tuple<std::vector<py::array>, std::vector<std::tuple<float, float>>>;


#define throw_exception(msg) throw std::runtime_error(msg + std::string("\n") + __FILE__ + std::string(":") + std::to_string(__LINE__))

class Transformation;

void apply_transformations(Example& example, const std::vector<std::shared_ptr<Transformation>>& transformations);

inline double rand_uniform(double min, double max) {
    return ((double) rand() / RAND_MAX) * (max - min) + min;
}


class Transformation {
public:
    Transformation(double probability, bool in_place) : probability(probability), in_place(in_place) { }

    virtual ~Transformation() { }

    void process_example(Example& example) {
        if (rand_uniform(0, 1) < probability) {
            setup(example);
            for (cv::Mat& image : std::get<0>(example)) {
                if (in_place) {
                    transform_image(image);
                } else {
                    cv::Mat result;
                    transform_image(image, result);
                    image = result;
                }
            }

            for (cv::Point2f& point : std::get<1>(example)) {
                point = transform_point(point);
            }
        }
    }

    virtual void setup(const Example& example) { }
    virtual void transform_image(cv::Mat& image) {assert(false);}  // In place
    virtual void transform_image(cv::Mat& image, cv::Mat& result) {assert(false);}
    virtual cv::Point2f transform_point(const cv::Point2f& point) {return point;}
    virtual std::shared_ptr<Transformation> clone() {assert(false);}


    std::pair<int, int> get_width_height(const Example& example) {
        assert (std::get<0>(example).size() > 0);
        
        int image_width = std::get<0>(example)[0].cols;
        int image_height = std::get<0>(example)[0].rows;

        return {image_width, image_height};
    }

private:
    double probability;
    bool in_place;
};


class Flip : public Transformation {
public:
    Flip(double probability, bool flip_horizontal) : Transformation(probability, false), flip_horizontal(flip_horizontal) { }

    ~Flip() { }

    void transform_image(cv::Mat& image, cv::Mat& result) {
        cv::flip(image, result, flip_horizontal ? 1 : 0);
    }
   
    cv::Point2f transform_point(const cv::Point2f& point) {
        if (flip_horizontal) {
            return {image_width - point.x, point.y};
        } else {
            return {point.x, image_height - point.y};
        }
    }

    void setup(const Example& example) {
        auto width_height = get_width_height(example);
        image_width = std::get<0>(width_height);
        image_height = std::get<1>(width_height);
    }

    std::shared_ptr<Transformation> clone() {
        return std::make_shared<Flip>(*this);
    }

private:
    bool flip_horizontal;
    int image_width;
    int image_height;       
};


class Crop : public Transformation {
public:
    Crop(double probability, double min_ratio) : Transformation(probability, true), min_ratio(min_ratio) {
        if (min_ratio > 1 || min_ratio <= 0) {
            throw_exception("Crop min_ratio must be between 0.0 and 1.0");
        }
     }
    ~Crop() { }

    void transform_image(cv::Mat& image) {
        image = image(cv::Rect(left, top, right - left, bottom - top));
    }

    cv::Point2f transform_point(const cv::Point2f& point) {
        return {point.x - left, point.y - top};
    }

    void setup(const Example& example) {        
        auto width_height = get_width_height(example);
        int image_width = std::get<0>(width_height);
        int image_height = std::get<1>(width_height);

        double scale = rand_uniform(min_ratio, 1.0);

        int new_width = image_width * scale;
        int new_height = image_height * scale;

        top = rand_uniform(0, image_height - new_height);
        left = rand_uniform(0, image_width - new_width);
        bottom = top + new_height;
        right = left + new_width;
    }

    std::shared_ptr<Transformation> clone() {
        return std::make_shared<Crop>(*this);
    }

private:
    double min_ratio;
    int top, left, bottom, right;
};


class Rotate : public Transformation {
public:
    Rotate(double probability, double min_angle, double max_angle) : Transformation(probability, false), min_angle(min_angle), max_angle(max_angle) { }
    ~Rotate() { }

    void transform_image(cv::Mat& image, cv::Mat& result) {
        cv::warpAffine(image, result, rotation_matrix, image.size());
    }

    cv::Point2f transform_point(const cv::Point2f& point) {
        cv::Point result;
        result.x = rotation_matrix.at<double>(0,0)*point.x + rotation_matrix.at<double>(0,1)*point.y + rotation_matrix.at<double>(0,2);
        result.y = rotation_matrix.at<double>(1,0)*point.x + rotation_matrix.at<double>(1,1)*point.y + rotation_matrix.at<double>(1,2);

        return result;
    }

    void setup(const Example& example) {
        auto width_height = get_width_height(example);
        int image_width = std::get<0>(width_height);
        int image_height = std::get<1>(width_height);


        double angle = rand_uniform(min_angle, max_angle);
        rotation_matrix = cv::getRotationMatrix2D(cv::Point2f{image_width / float(2.0), image_height / float(2.0)}, angle, 1.0);
    }

    std::shared_ptr<Transformation> clone() {
        return std::make_shared<Rotate>(*this);
    }

private:
    cv::Mat rotation_matrix;
    double min_angle, max_angle;
};


class BrightnessContrast : public Transformation {
public:
    BrightnessContrast(double probability, int brightness_lower=0, int brightness_upper=0, double contrast_lower=1.0, double contrast_upper=1.0) : 
        Transformation(probability, false),
        brightness_lower(brightness_lower),
        brightness_upper(brightness_upper),
        contrast_lower(contrast_lower),
        contrast_upper(contrast_upper) { }
    ~BrightnessContrast() { }

    void transform_image(cv::Mat& image, cv::Mat& result) {
        image.convertTo(result, -1, contrast, brightness);
    }

    void setup(const Example& example) {
        contrast = rand_uniform(contrast_lower, contrast_upper);
        brightness = rand_uniform(brightness_lower, brightness_upper);
    }

    std::shared_ptr<Transformation> clone() {
        return std::make_shared<BrightnessContrast>(*this);
    }

private:
    double contrast, contrast_lower, contrast_upper;
    int brightness, brightness_lower, brightness_upper;
};

class HueSaturation : public Transformation {
public:
    HueSaturation(double probability, int hue_low, int hue_high, int saturation_low, int saturation_high) : 
        Transformation(probability, false),
        hue_low(hue_low),
        hue_high(hue_high),
        saturation_low(saturation_low),
        saturation_high(saturation_high) { }
    ~HueSaturation() { }

    void transform_image(cv::Mat& image, cv::Mat& result) {
        cv::Mat hsv_image;
        cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
        for (int row = 0; row<hsv_image.rows; row++) {
            for (int col = 0; col<hsv_image.cols; col++) {
                hsv_image.at<cv::Vec3b>(row, col)[0] += hue;
                hsv_image.at<cv::Vec3b>(row, col)[1] += saturation;
            }
        }
        cv::cvtColor(hsv_image, result, cv::COLOR_HSV2BGR);
    }

    void setup(const Example& example) {
        hue = rand_uniform(hue_low, hue_high);
        saturation = rand_uniform(saturation_low, saturation_high);
    }

    std::shared_ptr<Transformation> clone() {
        return std::make_shared<HueSaturation>(*this);
    }

private:
    int hue, hue_low, hue_high;
    int saturation, saturation_low, saturation_high;
};

class Resize : public Transformation {
public:
    Resize(int height, int width) : Transformation(1.0, false), width(width), height(height) { }
    ~Resize() { }

    void transform_image(cv::Mat& image, cv::Mat& result) {
        cv::resize(image, result, cv::Size(width, height));
    }

    cv::Point2f transform_point(const cv::Point2f& point) {
        return {point.x * x_scale, point.y * y_scale};
    }

    void setup(const Example& example) {
        auto width_height = get_width_height(example);
        int image_width = std::get<0>(width_height);
        int image_height = std::get<1>(width_height);

        x_scale = float(width) / float(image_width);
        y_scale = float(height) / float(image_height);
    }

    std::shared_ptr<Transformation> clone() {
        return std::make_shared<Resize>(*this);
    }

private:
    int height, width;
    float x_scale;
    float y_scale;
};


class BGR2RGB : public Transformation {
public:
    BGR2RGB() : Transformation(1.0, false) { }
    ~BGR2RGB() { }

    void transform_image(cv::Mat& image, cv::Mat& result) {
        cv::cvtColor(image, result, cv::COLOR_BGR2RGB);
    }

    std::shared_ptr<Transformation> clone() {
        return std::make_shared<BGR2RGB>(*this);
    }
};
