#include <iostream>
#include <fstream>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "generator_HydraMarker.h"

using namespace std;
using namespace cv;

void example_A();

Mat1b loadMarkerField(const string& path)
{
    ifstream f(path);
    if (!f.is_open())
        throw runtime_error("Could not open marker field file: " + path);

    int W, H;
    f >> W >> H;

    Mat1b field(H, W);

    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            int v;
            f >> v;
            field(y, x) = static_cast<uchar>(v);
        }
    }

    return field;
}


Mat3b drawCheckerboardWithDots(
    const Mat1b& field,
    int cell_px = 80,
    int margin_px = 80,
    double dot_radius_rel = 0.22
)
{
    const int rows = field.rows;
    const int cols = field.cols;

    const int img_w = 2 * margin_px + cols * cell_px;
    const int img_h = 2 * margin_px + rows * cell_px;

    Mat3b img(img_h, img_w, Vec3b(255, 255, 255));

    // Draw checkerboard cells
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            Rect cell(
                margin_px + x * cell_px,
                margin_px + y * cell_px,
                cell_px,
                cell_px
            );

            bool black_cell = ((x + y) % 2 == 0);

            if (black_cell)
                rectangle(img, cell, Scalar(0, 0, 0), FILLED);
            else
                rectangle(img, cell, Scalar(255, 255, 255), FILLED);
        }
    }

    // Draw dots where field == 1
    const int r = static_cast<int>(dot_radius_rel * cell_px);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            if (field(y, x) != 1)
                continue;

            Point center(
                margin_px + x * cell_px + cell_px / 2,
                margin_px + y * cell_px + cell_px / 2
            );

            bool black_cell = ((x + y) % 2 == 0);

            Scalar dot_color = black_cell
                ? Scalar(255, 255, 255)
                : Scalar(0, 0, 0);

            circle(img, center, r, dot_color, FILLED, LINE_AA);
        }
    }

    return img;
}


int main()
{
    example_A();
    return 0;
}


void example_A()
{
    generator_HydraMarker gen;

    double max_ms = 600000;
    int max_trial = INT_MAX;

    string log_path = "generate.log";

    // Build initial marker field
    Mat1b state = 2 * Mat1b::ones(12, 12);

    // Set tag shape
    Mat1b shape1 = Mat1b::ones(3, 3);

    // Generate marker field
    gen.set_field(state);
    gen.set_tagShape(vector<Mat1b>{shape1});
    gen.generate(METHOD::FBWFC, max_ms, max_trial, false, log_path);

    // Save logical marker field
    gen.save("MF.field");

    // Load generated logical marker field
    Mat1b field = loadMarkerField("MF.field");

    // Create printable / visible checkerboard-dot marker
    Mat3b marker = drawCheckerboardWithDots(
        field,
        80,     // cell size in px
        80,     // margin in px
        0.22    // dot radius relative to cell size
    );

    imwrite("MF_marker.png", marker);

    namedWindow("marker", WINDOW_NORMAL);
    imshow("marker", marker);
    waitKey(0);
}