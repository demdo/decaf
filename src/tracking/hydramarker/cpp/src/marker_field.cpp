/*
 * MarkerField
 *
 * This file implements lookup in the global HydraMarker binary field.
 *
 * It is independent of image processing. It only knows:
 *   - the global binary marker field
 *   - the patch size
 *   - how to compare local patches against the global field
 */

#include "marker_field.hpp"

#include <fstream>
#include <stdexcept>

namespace hydramarker {

MarkerField MarkerField::loadFromFile(const std::string& path)
{
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error("Could not open marker field file: " + path);
    }

    MarkerField field;

    file >> field.width_;
    file >> field.height_;

    if (!file) {
        throw std::runtime_error("Could not read marker field size.");
    }

    if (field.width_ <= 0 || field.height_ <= 0) {
        throw std::runtime_error("Invalid marker field size.");
    }

    field.field_.resize(field.width_ * field.height_);

    for (int i = 0; i < field.width_ * field.height_; ++i) {
        int value = 0;
        file >> value;

        if (!file) {
            throw std::runtime_error("Could not read marker field values.");
        }

        if (value != 0 && value != 1) {
            throw std::runtime_error("Marker field contains invalid values.");
        }

        field.field_[i] = static_cast<uint8_t>(value);
    }

    int template_count = 0;
    file >> template_count;

    if (!file) {
        throw std::runtime_error("Could not read patch shape count.");
    }

    if (template_count != 1) {
        throw std::runtime_error("Only one patch shape is currently supported.");
    }

    int patch_width = 0;
    int patch_height = 0;

    file >> patch_width;
    file >> patch_height;

    if (!file) {
        throw std::runtime_error("Could not read patch shape size.");
    }

    if (patch_width != patch_height) {
        throw std::runtime_error("Only square patches are supported.");
    }

    if (patch_width <= 0) {
        throw std::runtime_error("Invalid patch size.");
    }

    field.patch_size_ = patch_width;

    for (int i = 0; i < patch_width * patch_height; ++i) {
        int value = 0;
        file >> value;

        if (!file) {
            throw std::runtime_error("Could not read patch shape values.");
        }

        if (value != 0 && value != 1) {
            throw std::runtime_error("Patch shape contains invalid values.");
        }
    }

    return field;
}

uint8_t MarkerField::at(int x, int y) const
{
    if (x < 0 || y < 0 || x >= width_ || y >= height_) {
        throw std::out_of_range("MarkerField::at index out of range.");
    }

    return field_[y * width_ + x];
}

std::vector<uint8_t> MarkerField::getPatch(int x, int y) const
{
    std::vector<uint8_t> patch;
    patch.resize(patch_size_ * patch_size_);

    for (int j = 0; j < patch_size_; ++j) {
        for (int i = 0; i < patch_size_; ++i) {
            patch[j * patch_size_ + i] = at(x + i, y + j);
        }
    }

    return patch;
}

std::vector<PatchMatch> MarkerField::findPatch(
    const std::vector<uint8_t>& patch
) const
{
    if (patch.size() != static_cast<size_t>(patch_size_ * patch_size_)) {
        throw std::runtime_error("Patch has wrong size.");
    }

    std::vector<PatchMatch> matches;

    for (int y = 0; y <= height_ - patch_size_; ++y) {
        for (int x = 0; x <= width_ - patch_size_; ++x) {
            for (int rotation : {0, 90, 180, 270}) {
                if (patchEqualsAt(patch, x, y, rotation)) {
                    PatchMatch match;
                    match.x = x;
                    match.y = y;
                    match.rotation_deg = rotation;
                    matches.push_back(match);
                }
            }
        }
    }

    return matches;
}

bool MarkerField::patchEqualsAt(
    const std::vector<uint8_t>& patch,
    int field_x,
    int field_y,
    int rotation_deg
) const
{
    for (int y = 0; y < patch_size_; ++y) {
        for (int x = 0; x < patch_size_; ++x) {
            const uint8_t local_value = patchValueRotated(
                patch,
                x,
                y,
                rotation_deg
            );

            const uint8_t field_value = at(field_x + x, field_y + y);

            if (local_value != field_value) {
                return false;
            }
        }
    }

    return true;
}

uint8_t MarkerField::patchValueRotated(
    const std::vector<uint8_t>& patch,
    int x,
    int y,
    int rotation_deg
) const
{
    const int k = patch_size_;

    int rx = x;
    int ry = y;

    if (rotation_deg == 0) {
        rx = x;
        ry = y;
    }
    else if (rotation_deg == 90) {
        rx = y;
        ry = k - 1 - x;
    }
    else if (rotation_deg == 180) {
        rx = k - 1 - x;
        ry = k - 1 - y;
    }
    else if (rotation_deg == 270) {
        rx = k - 1 - y;
        ry = x;
    }
    else {
        throw std::runtime_error("Unsupported patch rotation.");
    }

    return patch[ry * k + rx];
}

} // namespace hydramarker