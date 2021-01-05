#include "header.hpp"


/// Applies multiple transformations in-place.
void apply_transformations(Example& example, const std::vector<std::shared_ptr<Transformation>>& transformations) {
    for (const std::shared_ptr<Transformation>& transformation: transformations) {
        transformation->process_example(example);
    }
}

