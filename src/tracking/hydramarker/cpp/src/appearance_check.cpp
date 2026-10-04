#include "appearance_check.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace hydramarker {

namespace {

// Cells seen at a grazing angle shrink to a few pixels and their samples mix
// with the neighbours; only cells facing the camera are sampled.
constexpr double kMinFacingCos = 0.35;

// Grey ROI margin around the projections: room for the 3x3 blur and the
// bilinear neighbour of the outermost samples.
constexpr double kRoiMarginPx = 4.0;

// Cells per base colour the dot term needs: with dots unrelated to the
// expected bits, a correlation over a few cells reaches the threshold too
// often by chance.
constexpr size_t kMinDotCellsPerColour = 8;

double pearson(const std::vector<double>& x, const std::vector<double>& y)
{
    const size_t n = x.size();
    if (n < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double mx = 0.0;
    double my = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double a = x[i] - mx;
        const double b = y[i] - my;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    if (sxx < 1e-12 || syy < 1e-12) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sxy / std::sqrt(sxx * syy);
}

// Pearson of the expected vs. the measured dot term inside each base colour,
// combined weighted by the class sizes. The expected term (base XOR bit) -
// base carries the sign of the base colour, so pooled over both colours the
// correlation keeps a between-colour part even for an identity shifted by an
// even number of cells (parity intact, dots unrelated). NaN unless both
// colours have kMinDotCellsPerColour cells and a finite correlation.
double withinColourPearson(
    const std::vector<double>& base,
    const std::vector<double>& x,
    const std::vector<double>& y
)
{
    double weighted = 0.0;
    size_t count = 0;
    for (int colour = 0; colour <= 1; ++colour) {
        std::vector<double> xs;
        std::vector<double> ys;
        for (size_t i = 0; i < base.size(); ++i) {
            if (static_cast<int>(base[i]) == colour) {
                xs.push_back(x[i]);
                ys.push_back(y[i]);
            }
        }
        if (xs.size() < kMinDotCellsPerColour) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double r = pearson(xs, ys);
        if (!std::isfinite(r)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        weighted += static_cast<double>(xs.size()) * r;
        count += xs.size();
    }
    return weighted / static_cast<double>(count);
}

} // namespace

AppearanceCheck checkMarkerAppearance(
    const cv::Mat& frame,
    const MarkerGeometry& geometry,
    const MarkerField& field,
    const cv::Matx33d& K,
    const std::vector<double>& dist_coeffs,
    const std::vector<double>& rvec,
    const std::vector<double>& tvec,
    double min_parity,
    double min_dot,
    int min_cells
)
{
    AppearanceCheck out;
    if (frame.empty() || rvec.size() != 3 || tvec.size() != 3) {
        return out;
    }
    const int rows = geometry.cornerRows();
    const int cols = geometry.cornerCols();
    if (field.height() != rows - 1 || field.width() != cols - 1) {
        return out;
    }
    const SurfaceModel& surface = geometry.surfaceModel();

    cv::Matx33d R;
    cv::Rodrigues(cv::Vec3d(rvec[0], rvec[1], rvec[2]), R);
    const cv::Vec3d t(tvec[0], tvec[1], tvec[2]);

    // Per visible cell: the centre and four off-centre points (0.2 / 0.8 of
    // the cell, outside the dot), its base colour and expected centre colour.
    std::vector<cv::Point3d> points;
    std::vector<double> base;
    std::vector<double> centre_expected;
    points.reserve(static_cast<size_t>((rows - 1) * (cols - 1) * 5));
    for (int r = 0; r + 1 < rows; ++r) {
        for (int c = 0; c + 1 < cols; ++c) {
            if (!geometry.hasCorner(r, c) || !geometry.hasCorner(r, c + 1) ||
                !geometry.hasCorner(r + 1, c) ||
                !geometry.hasCorner(r + 1, c + 1)) {
                continue;
            }
            const cv::Point3f q00 = geometry.cornerPoint(r, c);
            const cv::Point3f q01 = geometry.cornerPoint(r, c + 1);
            const cv::Point3f q10 = geometry.cornerPoint(r + 1, c);
            const cv::Point3f q11 = geometry.cornerPoint(r + 1, c + 1);
            // a along the rows, b along the columns
            auto cellPoint = [&](double a, double b) {
                const cv::Point3f p =
                    static_cast<float>((1 - a) * (1 - b)) * q00 +
                    static_cast<float>((1 - a) * b) * q01 +
                    static_cast<float>(a * (1 - b)) * q10 +
                    static_cast<float>(a * b) * q11;
                return cv::Vec3d(p.x, p.y, p.z);
            };
            const cv::Vec3d centre = cellPoint(0.5, 0.5);
            // Outward normal of the print at the cell. The cylinder gives it
            // radially. A plane's stored normal has no defined sign, so it is
            // turned towards the camera below (as the corner refiner does). A
            // map without a fitted surface takes the normal from the cell's
            // diagonals: column step x row step points into the carrier.
            cv::Vec3d normal;
            if (surface.type == SurfaceModel::Type::Cylinder) {
                const cv::Vec3d w = centre - surface.point;
                normal = w - w.dot(surface.dir) * surface.dir;
            } else if (surface.type == SurfaceModel::Type::Plane) {
                normal = surface.dir;
            } else {
                const cv::Point3f d0 = q11 - q00;
                const cv::Point3f d1 = q10 - q01;
                normal = -cv::Vec3d(d0.x, d0.y, d0.z)
                              .cross(cv::Vec3d(d1.x, d1.y, d1.z));
            }
            const double normal_len = cv::norm(normal);
            if (normal_len < 1e-9) {
                continue;
            }
            normal *= 1.0 / normal_len;
            const cv::Vec3d centre_cam = R * centre + t;
            cv::Vec3d normal_cam = R * normal;
            const double range = cv::norm(centre_cam);
            if (range < 1e-6 || centre_cam[2] <= 0) {
                continue;
            }
            if (surface.type == SurfaceModel::Type::Plane &&
                normal_cam.dot(centre_cam) > 0.0) {
                normal_cam = -normal_cam;
            }
            if (-normal_cam.dot(centre_cam) / range <= kMinFacingCos) {
                continue;
            }
            const int b = (r + c) % 2;
            const int bit = field.at(c, r) ? 1 : 0;
            base.push_back(b);
            centre_expected.push_back(static_cast<double>(b ^ bit));
            points.emplace_back(centre[0], centre[1], centre[2]);
            for (double a : {0.2, 0.8}) {
                for (double bb : {0.2, 0.8}) {
                    const cv::Vec3d p = cellPoint(a, bb);
                    points.emplace_back(p[0], p[1], p[2]);
                }
            }
        }
    }
    const int visible = static_cast<int>(base.size());
    out.cells = visible;
    if (visible < min_cells) {
        return out;
    }

    std::vector<cv::Point2d> uv;
    cv::projectPoints(points, cv::Vec3d(rvec[0], rvec[1], rvec[2]), t,
                      cv::Mat(K),
                      dist_coeffs.empty() ? cv::Mat() : cv::Mat(dist_coeffs),
                      uv);

    // Grey ROI around all projections, 3x3 blur (sigma 0.8) against noise.
    double x0 = 1e30;
    double y0 = 1e30;
    double x1 = -1e30;
    double y1 = -1e30;
    for (const auto& p : uv) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            continue;
        }
        x0 = std::min(x0, p.x);
        y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x);
        y1 = std::max(y1, p.y);
    }
    if (x0 > x1 || y0 > y1) {
        out.cells = 0;
        return out;
    }
    // Clamped to the frame in double, so a far-off pose cannot overflow the
    // conversion to int.
    const double left = std::floor(x0) - kRoiMarginPx;
    const double top = std::floor(y0) - kRoiMarginPx;
    const double right = left + std::ceil(x1 - x0) + 2.0 * kRoiMarginPx + 1.0;
    const double bottom = top + std::ceil(y1 - y0) + 2.0 * kRoiMarginPx + 1.0;
    const double width = static_cast<double>(frame.cols);
    const double height = static_cast<double>(frame.rows);
    const int roi_x0 = static_cast<int>(std::min(std::max(left, 0.0), width));
    const int roi_y0 = static_cast<int>(std::min(std::max(top, 0.0), height));
    const int roi_x1 = static_cast<int>(std::min(std::max(right, 0.0), width));
    const int roi_y1 = static_cast<int>(std::min(std::max(bottom, 0.0), height));
    if (roi_x1 - roi_x0 < 4 || roi_y1 - roi_y0 < 4) {
        out.cells = 0;
        return out;
    }
    const cv::Rect roi(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0);
    cv::Mat grey;
    if (frame.channels() == 4) {
        cv::cvtColor(frame(roi), grey, cv::COLOR_BGRA2GRAY);
    } else if (frame.channels() == 3) {
        cv::cvtColor(frame(roi), grey, cv::COLOR_BGR2GRAY);
    } else {
        grey = frame(roi);
    }
    cv::Mat grey_f;
    grey.convertTo(grey_f, CV_32F);
    cv::GaussianBlur(grey_f, grey_f, cv::Size(3, 3), 0.8);
    auto sample = [&](const cv::Point2d& p) {
        const double u =
            std::min(std::max(p.x - roi.x, 0.0), grey_f.cols - 1.001);
        const double v =
            std::min(std::max(p.y - roi.y, 0.0), grey_f.rows - 1.001);
        const int ix = static_cast<int>(u);
        const int iy = static_cast<int>(v);
        const double ax = u - ix;
        const double ay = v - iy;
        const float* r0 = grey_f.ptr<float>(iy);
        const float* r1 = grey_f.ptr<float>(std::min(iy + 1, grey_f.rows - 1));
        const int ix1 = std::min(ix + 1, grey_f.cols - 1);
        return (1 - ay) * ((1 - ax) * r0[ix] + ax * r0[ix1]) +
               ay * ((1 - ax) * r1[ix] + ax * r1[ix1]);
    };

    std::vector<double> sampled_base;
    std::vector<double> off_centre;
    std::vector<double> dot_expected;
    std::vector<double> dot_measured;
    for (int i = 0; i < visible; ++i) {
        const cv::Point2d pc = uv[static_cast<size_t>(5 * i)];
        if (!(pc.x > 2 && pc.x < frame.cols - 2 &&
              pc.y > 2 && pc.y < frame.rows - 2)) {
            continue;
        }
        const double centre = sample(pc);
        double off = 0;
        for (int k = 1; k <= 4; ++k) {
            off += sample(uv[static_cast<size_t>(5 * i + k)]);
        }
        off *= 0.25;
        sampled_base.push_back(base[static_cast<size_t>(i)]);
        off_centre.push_back(off);
        dot_expected.push_back(centre_expected[static_cast<size_t>(i)] -
                               base[static_cast<size_t>(i)]);
        dot_measured.push_back(centre - off);
    }
    out.cells = static_cast<int>(sampled_base.size());
    if (out.cells < min_cells) {
        return out;
    }
    out.parity = pearson(sampled_base, off_centre);
    out.dot = withinColourPearson(sampled_base, dot_expected, dot_measured);
    out.passed = std::isfinite(out.parity) && std::isfinite(out.dot) &&
                 out.parity >= min_parity && out.dot >= min_dot;
    out.failed = (std::isfinite(out.parity) && out.parity < min_parity) ||
                 (std::isfinite(out.dot) && out.dot < min_dot);
    return out;
}

} // namespace hydramarker
