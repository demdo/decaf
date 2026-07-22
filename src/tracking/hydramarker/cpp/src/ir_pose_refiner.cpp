#include "ir_pose_refiner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// Depth-only IR stereo fusion. Every numerical step below mirrors the
// validated offline prototype (tests/prove_ir_depth_fusion.py +
// tests/analyze_ir_divot.py) operation-for-operation so the replay
// verification can compare the two paths frame by frame. Deviations from the
// obvious implementation (seed transfer through the RGB intrinsics, the
// left-view depth reused for the right-view transfer, the residual/keep
// bookkeeping of the robust fit) are deliberate parity with the prototype.

namespace hydramarker {

namespace {

cv::Mat distMat(const std::vector<double>& dist)
{
    if (dist.empty()) {
        return cv::Mat::zeros(1, 5, CV_64F);
    }
    cv::Mat m(1, static_cast<int>(dist.size()), CV_64F);
    for (size_t i = 0; i < dist.size(); ++i) {
        m.at<double>(0, static_cast<int>(i)) = dist[i];
    }
    return m;
}


double rotationAngleDeg(const cv::Matx33d& A, const cv::Matx33d& B)
{
    const cv::Matx33d D = A.t() * B;
    const double c = std::max(-1.0, std::min(1.0, (cv::trace(D) - 1.0) / 2.0));
    return std::acos(c) * 180.0 / CV_PI;
}

double median(std::vector<double> v)
{
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 == 1) {
        return v[n / 2];
    }
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// RGB pixels -> target-camera pixels through the rig extrinsics with a
// per-corner depth (prototype transfer_to; the depth is the IR-LEFT z for
// BOTH views - kept as-is for parity, the model_warp snap absorbs the
// sub-pixel seed error).
std::vector<cv::Point2f> transferTo(const std::vector<cv::Point2d>& uv_rgb,
                                    const std::vector<double>& z_mm,
                                    const cv::Matx33d& R_rgb_tgt,
                                    const cv::Vec3d& t_rgb_tgt_m,
                                    const cv::Matx33d& K_tgt,
                                    const std::vector<double>& dist_tgt,
                                    const IrCameraCalibration& calib)
{
    const size_t n = uv_rgb.size();
    std::vector<cv::Point2f> out(n, cv::Point2f(0.0f, 0.0f));
    if (n == 0) {
        return out;
    }
    cv::Mat src(static_cast<int>(n), 1, CV_64FC2);
    for (size_t i = 0; i < n; ++i) {
        src.at<cv::Vec2d>(static_cast<int>(i)) =
            cv::Vec2d(uv_rgb[i].x, uv_rgb[i].y);
    }
    cv::Mat norm;
    cv::undistortPoints(src, norm, cv::Mat(calib.K_rgb),
                        distMat(calib.dist_rgb));
    std::vector<cv::Point3d> pts_t(n);
    for (size_t i = 0; i < n; ++i) {
        const cv::Vec2d nv = norm.at<cv::Vec2d>(static_cast<int>(i));
        const double z_m = z_mm[i] / 1000.0;
        const cv::Vec3d p_rgb(nv[0] * z_m, nv[1] * z_m, z_m);
        const cv::Vec3d p = R_rgb_tgt * p_rgb + t_rgb_tgt_m;
        pts_t[i] = cv::Point3d(p[0], p[1], p[2]);
    }
    std::vector<cv::Point2d> proj;
    cv::projectPoints(pts_t, cv::Mat::zeros(3, 1, CV_64F),
                      cv::Mat::zeros(3, 1, CV_64F), cv::Mat(K_tgt),
                      distMat(dist_tgt), proj);
    for (size_t i = 0; i < n; ++i) {
        out[i] = cv::Point2f(static_cast<float>(proj[i].x),
                             static_cast<float>(proj[i].y));
    }
    return out;
}

// Per-corner saturation gate: any pixel >= threshold inside the +-half patch
// around the seed (integer-truncated, prototype convention) kills the corner.
std::vector<uint8_t> satFree(const cv::Mat& img,
                             const std::vector<cv::Point2f>& pts,
                             int threshold,
                             int half)
{
    std::vector<uint8_t> ok(pts.size(), 1);
    const int H = img.rows;
    const int W = img.cols;
    for (size_t i = 0; i < pts.size(); ++i) {
        const int u = static_cast<int>(pts[i].x);
        const int v = static_cast<int>(pts[i].y);
        const int x0 = std::max(0, u - half);
        const int x1 = std::min(W, u + half + 1);
        const int y0 = std::max(0, v - half);
        const int y1 = std::min(H, v + half + 1);
        if (x1 <= x0 || y1 <= y0) {
            ok[i] = 0;
            continue;
        }
        double max_val = 0.0;
        cv::minMaxLoc(img(cv::Range(y0, y1), cv::Range(x0, x1)), nullptr,
                      &max_val);
        if (max_val >= static_cast<double>(threshold)) {
            ok[i] = 0;
        }
    }
    return ok;
}

// IR-left/IR-right pixels -> 3D points (mm) in the IR-left frame
// (prototype triangulate_irl: normalized DLT with the METRIC baseline).
std::vector<cv::Vec3d> triangulateIrl(const std::vector<cv::Point2f>& pl,
                                      const std::vector<cv::Point2f>& pr,
                                      const IrCameraCalibration& calib,
                                      double depth_scale)
{
    const size_t n = pl.size();
    std::vector<cv::Vec3d> out(n);
    if (n == 0) {
        return out;
    }
    auto undist = [](const std::vector<cv::Point2f>& p, const cv::Matx33d& K,
                     const std::vector<double>& dist) {
        cv::Mat src(static_cast<int>(p.size()), 1, CV_64FC2);
        for (size_t i = 0; i < p.size(); ++i) {
            src.at<cv::Vec2d>(static_cast<int>(i)) =
                cv::Vec2d(p[i].x, p[i].y);
        }
        cv::Mat norm;
        cv::undistortPoints(src, norm, cv::Mat(K), distMat(dist));
        return norm;
    };
    const cv::Mat nl = undist(pl, calib.K_left, calib.dist_left);
    const cv::Mat nr = undist(pr, calib.K_right, calib.dist_right);
    cv::Mat pts_l(2, static_cast<int>(n), CV_64F);
    cv::Mat pts_r(2, static_cast<int>(n), CV_64F);
    for (size_t i = 0; i < n; ++i) {
        const cv::Vec2d a = nl.at<cv::Vec2d>(static_cast<int>(i));
        const cv::Vec2d b = nr.at<cv::Vec2d>(static_cast<int>(i));
        pts_l.at<double>(0, static_cast<int>(i)) = a[0];
        pts_l.at<double>(1, static_cast<int>(i)) = a[1];
        pts_r.at<double>(0, static_cast<int>(i)) = b[0];
        pts_r.at<double>(1, static_cast<int>(i)) = b[1];
    }
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    P1.at<double>(0, 0) = 1.0;
    P1.at<double>(1, 1) = 1.0;
    P1.at<double>(2, 2) = 1.0;
    cv::Mat P2(3, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            P2.at<double>(r, c) = calib.R_left_right(r, c);
        }
        P2.at<double>(r, 3) = calib.t_left_right_mm[r] / 1000.0;  // metres
    }
    cv::Mat Xh;
    cv::triangulatePoints(P1, P2, pts_l, pts_r, Xh);
    for (size_t i = 0; i < n; ++i) {
        const int c = static_cast<int>(i);
        const double w = Xh.at<double>(3, c);
        out[i] = cv::Vec3d(Xh.at<double>(0, c) / w, Xh.at<double>(1, c) / w,
                           Xh.at<double>(2, c) / w) *
                 1000.0 * depth_scale;
    }
    return out;
}

}  // namespace

// Tightly-coupled MAP pose fusion. Operation-for-operation parity with the
// validated prototype map_core.fuse_map_vec so the C++ and Python solvers can
// be replay-compared on identical (X, u, Y) inputs (tests/replay_map_fusion_cpp).
MapFuseResult mapPoseFuse(const std::vector<cv::Vec3d>& X,
                          const std::vector<cv::Point2d>& u_norm,
                          const std::vector<cv::Vec3d>& Y,
                          const std::vector<cv::Matx33d>& SigY_inv,
                          const cv::Matx33d& R0, const cv::Vec3d& t0,
                          double f, double sig_px, double w3d,
                          int iters, double huber, bool trim)
{
    MapFuseResult out;
    const size_t N = X.size();
    if (N < static_cast<size_t>(6) || u_norm.size() != N || Y.size() != N ||
        SigY_inv.size() != N || sig_px <= 0.0) {
        return out;
    }
    const double Wn = (f / sig_px) * (f / sig_px);
    cv::Matx33d R = R0;
    cv::Vec3d t = t0;

    // Initial MAD trim on the IR-3D Mahalanobis residual (gross aliased pairs).
    std::vector<uint8_t> keep(N, 1);
    if (trim) {
        std::vector<double> m(N);
        for (size_t i = 0; i < N; ++i) {
            const cv::Vec3d r = R * X[i] + t - Y[i];
            m[i] = std::sqrt(std::max(r.dot(SigY_inv[i] * r), 0.0));
        }
        const double md = median(m);
        std::vector<double> ad(N);
        for (size_t i = 0; i < N; ++i) {
            ad[i] = std::abs(m[i] - md);
        }
        const double mad = median(ad);
        const double thr = md + 4.0 * 1.4826 * mad;
        int nk = 0;
        for (size_t i = 0; i < N; ++i) {
            keep[i] = m[i] < thr ? 1 : 0;
            nk += keep[i];
        }
        if (nk < 6) {
            std::fill(keep.begin(), keep.end(), static_cast<uint8_t>(1));
        }
    }

    auto expmap = [](const cv::Vec3d& w) -> cv::Matx33d {
        cv::Mat rvec = (cv::Mat_<double>(3, 1) << w[0], w[1], w[2]);
        cv::Mat Rm;
        cv::Rodrigues(rvec, Rm);
        return cv::Matx33d(Rm);
    };

    cv::Matx66d H = cv::Matx66d::zeros();
    for (int it = 0; it < iters; ++it) {
        H = cv::Matx66d::zeros();
        cv::Vec<double, 6> g = cv::Vec<double, 6>::all(0.0);
        int used = 0;
        for (size_t i = 0; i < N; ++i) {
            if (!keep[i]) {
                continue;
            }
            ++used;
            const cv::Vec3d Xc = R * X[i] + t;
            const double x = Xc[0], y = Xc[1], z = Xc[2];
            // J_se3 (3x6) = [ I3 | -skew(Xc) ].
            cv::Matx<double, 3, 6> Jse = cv::Matx<double, 3, 6>::zeros();
            Jse(0, 0) = 1.0; Jse(1, 1) = 1.0; Jse(2, 2) = 1.0;
            Jse(0, 4) = z;  Jse(0, 5) = -y;
            Jse(1, 3) = -z; Jse(1, 5) = x;
            Jse(2, 3) = y;  Jse(2, 4) = -x;
            // RGB reprojection residual (normalised, undistorted).
            const cv::Vec2d rp(x / z - u_norm[i].x, y / z - u_norm[i].y);
            const cv::Matx23d Jp(1.0 / z, 0.0, -x / (z * z),
                                 0.0, 1.0 / z, -y / (z * z));
            const cv::Matx<double, 2, 6> Jpx = Jp * Jse;
            const double ep = std::sqrt(rp[0] * rp[0] + rp[1] * rp[1]) * f / sig_px;
            const double wp = (ep <= huber ? 1.0 : huber / std::max(ep, 1e-9)) * Wn;
            H += wp * (Jpx.t() * Jpx);
            g += wp * (Jpx.t() * rp);
            // IR stereo 3D residual, anisotropic information, Huber-weighted.
            const cv::Vec3d r3 = Xc - Y[i];
            const cv::Matx33d W = SigY_inv[i] * w3d;
            const double e3 = std::sqrt(std::max(r3.dot(W * r3), 0.0));
            const double w3 = e3 <= huber ? 1.0 : huber / std::max(e3, 1e-9);
            const cv::Matx33d Weff = W * w3;
            H += Jse.t() * Weff * Jse;
            g += Jse.t() * (Weff * r3);
        }
        out.n_used = used;
        cv::Matx66d Hreg = H;
        for (int k = 0; k < 6; ++k) {
            Hreg(k, k) += 1e-9;
        }
        const cv::Vec<double, 6> d = Hreg.solve(cv::Vec<double, 6>(-g),
                                                cv::DECOMP_CHOLESKY);
        const cv::Matx33d Rexp = expmap(cv::Vec3d(d[3], d[4], d[5]));
        R = Rexp * R;
        t = Rexp * t + cv::Vec3d(d[0], d[1], d[2]);
        double dn = 0.0;
        for (int k = 0; k < 6; ++k) {
            dn += d[k] * d[k];
        }
        if (std::sqrt(dn) < 1e-8) {
            break;
        }
    }
    out.R = R;
    out.t = t;
    cv::invert(H, out.cov, cv::DECOMP_SVD);   // measurement covariance = H^-1
    out.ok = true;
    return out;
}

void IrPoseRefiner::configure(const IrPoseRefinerConfig& config,
                              const SurfaceModel& surface,
                              const std::vector<cv::Vec3d>& corner_cloud_mm)
{
    config_ = config;
    surface_ = surface;
    // Open the band fully: it only gates the angular/axial surface
    // coordinate, and the oblique IR views may exceed the printed band by a
    // margin the RGB path never sees (prototype get_surface convention).
    surface_.band_a_min = -1e6;
    surface_.band_a_max = 1e6;
    surface_.band_b_min = -1e6;
    surface_.band_b_max = 1e6;

    corners_ = corner_cloud_mm;
    corner_normals_.assign(corners_.size(), cv::Vec3d(0.0, 0.0, 1.0));
    if (surface.type == SurfaceModel::Type::Cylinder) {
        const cv::Vec3d p = surface.point;
        cv::Vec3d d = surface.dir;
        const double dn = cv::norm(d);
        if (dn > 0.0) {
            d /= dn;
        }
        for (size_t i = 0; i < corners_.size(); ++i) {
            const cv::Vec3d v0 = corners_[i] - p;
            cv::Vec3d nrm = v0 - v0.dot(d) * d;
            const double nn = cv::norm(nrm);
            if (nn > 0.0) {
                nrm /= nn;
            }
            corner_normals_[i] = nrm;
        }
    } else if (surface.type == SurfaceModel::Type::Plane) {
        for (size_t i = 0; i < corners_.size(); ++i) {
            corner_normals_[i] = surface.dir;
        }
    }
}

void IrPoseRefiner::setCalibration(const IrCameraCalibration& calib)
{
    calib_ = calib;
}

void IrPoseRefiner::reset()
{
}


namespace {

// Direct L->R epipolar disparity refinement: ZNCC of the CURRENT left
// patch against the right image along the (near-rectified) epipolar row,
// integer search + parabola subpixel.  The left localization error shifts
// BOTH patches together and cancels in the disparity — the common-mode
// property the reference pairs had (same enrolled pair in both views) and
// two independent per-view measurements lose.  Returns the refined right
// x or a negative value when the correlation is unusable.
double qfEpipolarRefineX(const cv::Mat& irL, const cv::Mat& irR,
                         const cv::Point2f& pL, const cv::Point2f& pR0)
{
    constexpr int kHalf = 6;
    constexpr int kWin = 2 * kHalf + 1;
    constexpr int kRange = 3;
    if (pL.x < kHalf + 1 || pL.y < kHalf + 1 ||
        pL.x >= irL.cols - kHalf - 2 || pL.y >= irL.rows - kHalf - 2) {
        return -1.0;
    }
    if (pR0.x < kHalf + kRange + 1 || pR0.y < kHalf + 1 ||
        pR0.x >= irR.cols - kHalf - kRange - 2 ||
        pR0.y >= irR.rows - kHalf - 2) {
        return -1.0;
    }
    cv::Mat tpl;
    cv::getRectSubPix(irL, cv::Size(kWin, kWin), pL, tpl, CV_32F);
    cv::Scalar tm, ts;
    cv::meanStdDev(tpl, tm, ts);
    if (ts[0] < 2.0) return -1.0;   // flat patch: no structure to match

    double best[2 * kRange + 1];
    double best_v = -2.0;
    int best_s = 0;
    for (int s = -kRange; s <= kRange; ++s) {
        cv::Mat pat;
        cv::getRectSubPix(irR, cv::Size(kWin, kWin),
                          cv::Point2f(pR0.x + s, pR0.y), pat, CV_32F);
        cv::Scalar pm, ps;
        cv::meanStdDev(pat, pm, ps);
        double v = -1.0;
        if (ps[0] > 1e-6) {
            const cv::Mat a = tpl - tm[0];
            const cv::Mat b = pat - pm[0];
            v = a.dot(b) / (ts[0] * ps[0] *
                            static_cast<double>(kWin * kWin));
        }
        best[s + kRange] = v;
        if (v > best_v) {
            best_v = v;
            best_s = s;
        }
    }
    if (best_v < 0.5) return -1.0;  // no credible correspondence
    double sub = 0.0;
    if (best_s > -kRange && best_s < kRange) {
        const double y0 = best[best_s - 1 + kRange];
        const double y1 = best[best_s + kRange];
        const double y2 = best[best_s + 1 + kRange];
        const double den = y0 - 2.0 * y1 + y2;
        if (std::abs(den) > 1e-12) {
            sub = std::clamp(0.5 * (y0 - y2) / den, -0.6, 0.6);
        }
    }
    return pR0.x + best_s + sub;
}

}  // namespace

void IrPoseRefiner::measureViewQf(const cv::Mat& gray,
                                  const cv::Matx33d& K,
                                  const std::vector<double>& dist,
                                  const cv::Matx33d& R_view,
                                  const cv::Vec3d& t_view,
                                  const std::vector<cv::Vec3d>& xyz,
                                  const std::vector<cv::Point2f>& seeds,
                                  std::vector<cv::Point2f>& uv_out,
                                  std::vector<uint8_t>& ok_out,
                                  std::vector<float>& q_out) const
{
    const size_t n = xyz.size();
    uv_out = seeds;
    ok_out.assign(n, 0);
    q_out.assign(n, 0.0f);
    if (n == 0 || gray.empty()) {
        return;
    }

    CornerModelContext ctx;
    ctx.K = K;
    ctx.dist = dist;
    ctx.R = R_view;
    ctx.t = t_view;
    ctx.surface = surface_;
    ctx.anchor_xyz_mm = xyz;
    ctx.anchor_valid.assign(n, 1);
    // no ref_gray: the QF path is reference-free (usableQuadratic).
    if (pattern_.valid()) {
        ctx.pattern = &pattern_;
    }

    CornerRefinementConfig cfg;
    cfg.tracked_refine_method = "quadratic_form";
    cfg.qf_max_dev_px = config_.qf_max_dev_px;
    // IR views: engage the synthetic saddle registration for EVERY corner,
    // not only grazing ones. On the ~11 px IR cells the 1D-profile residual
    // biases (polarity/dot environment) reach ~0.2 px PER CORNER and differ
    // between the L and R views -> stereo amplifies them x(Z/b) into a
    // stable ~1 mm depth warp (fb5 diag: fit_rms 0.84 vs 0.11, constant
    // -0.9 mm marker-z offset). The perspective-rendered 2D template models
    // the view difference exactly - the reference-pair property, photo-free.
    cfg.qf_saddle_min_incidence_deg = 0.01;
    // The default 0.6 ZNCC gate was tuned for photo-reference matching; a
    // BINARY quadrant template against the soft 720p IR runs lower even
    // when correctly locked.
    cfg.saddle_min_zncc = 0.30;
    // The subpix snap inside refineTrackedCorners first pulls the
    // rig-transferred seeds onto the nearest IR saddle (fixing most of the
    // 1-3 px transfer error), then QF measures on the anchored curves and
    // the saddle-warp layer covers high-incidence corners.
    const std::vector<bool> predicted(n, false);
    const TrackedRefineStats stats = refiner_.refineTrackedCorners(
        gray, uv_out, predicted, cfg, &ctx);

    if (stats.corner_ok.size() != n) {
        uv_out = seeds;
        return;
    }
    ok_out = stats.corner_ok;
    if (stats.corner_zncc.size() == n) {
        q_out = stats.corner_zncc;
    }

    static const bool dbg = std::getenv("HYDRA_QFIR_DEBUG") != nullptr;
    if (dbg) {
        int ok_n = 0;
        for (const uint8_t o : ok_out) ok_n += o;
        std::cerr << "[qfir] n=" << n << " ok=" << ok_n
                  << " qf=" << stats.qf_count
                  << " saddle=" << stats.qf_saddle_count << std::endl;
    }
}

IrPoseRefinerResult IrPoseRefiner::fuse(const cv::Mat& ir_left,
                                        const cv::Mat& ir_right,
                                        const std::array<double, 3>& rvec,
                                        const std::array<double, 3>& tvec,
                                        const std::vector<cv::Vec3d>& rgb_xyz,
                                        const std::vector<cv::Point2d>& rgb_uv,
                                        double sigma_px_override)
{
    const double sigma_px_eff =
        sigma_px_override > 0.0 ? sigma_px_override : config_.sigma_px;
    IrPoseRefinerResult out;
    out.rvec = rvec;
    out.tvec = tvec;
    if (!active() || ir_left.empty() || ir_right.empty() ||
        rgb_xyz.size() != rgb_uv.size() ||
        static_cast<int>(rgb_xyz.size()) < config_.min_pairs) {
        return out;
    }

    cv::Mat rvec_mat = (cv::Mat_<double>(3, 1) << rvec[0], rvec[1], rvec[2]);
    cv::Mat R_mat;
    cv::Rodrigues(rvec_mat, R_mat);
    const cv::Matx33d R_rgb(R_mat);
    const cv::Vec3d t_rgb(tvec[0], tvec[1], tvec[2]);

    // The MAP fuses the SAME corners the RGB pose was solved on: model position
    // (reprojection + IR-triangulation share it), detected RGB pixel
    // (reprojection residual). The tracked corners are visible by construction,
    // so no surface-normal visibility set is built here.
    const std::vector<cv::Vec3d>& xyz = rgb_xyz;
    const std::vector<cv::Point2d>& uv_det = rgb_uv;
    const size_t n = xyz.size();

    // Pose in the two IR frames.
    const cv::Matx33d R_A = calib_.R_rgb_left * R_rgb;
    const cv::Vec3d t_A = calib_.R_rgb_left * t_rgb + calib_.t_rgb_left_mm;
    const cv::Matx33d R_B = calib_.R_left_right * R_A;
    const cv::Vec3d t_B = calib_.R_left_right * t_A + calib_.t_left_right_mm;

    // Seeds: the DETECTED RGB pixels transferred through the rig (per-corner
    // depth from the IR-left frame, prototype convention -> the seed follows the
    // real corner, model_warp refines from there).
    const std::vector<cv::Point2d>& uv_rgb = uv_det;
    std::vector<double> z_mm(n);
    for (size_t i = 0; i < n; ++i) {
        z_mm[i] = (R_A * xyz[i] + t_A)[2];
    }
    const cv::Vec3d T_rl_m = calib_.t_rgb_left_mm / 1000.0;
    const cv::Vec3d T_lr_m = calib_.t_left_right_mm / 1000.0;
    const std::vector<cv::Point2f> seedL =
        transferTo(uv_rgb, z_mm, calib_.R_rgb_left, T_rl_m, calib_.K_left,
                   calib_.dist_left, calib_);
    const cv::Matx33d R_rgb_r = calib_.R_left_right * calib_.R_rgb_left;
    const cv::Vec3d T_rgb_r = calib_.R_left_right * T_rl_m + T_lr_m;
    const std::vector<cv::Point2f> seedR =
        transferTo(uv_rgb, z_mm, R_rgb_r, T_rgb_r, calib_.K_right,
                   calib_.dist_right, calib_);

    // Saturation gate on the raw images around the seeds (both views).
    const std::vector<uint8_t> sfL =
        satFree(ir_left, seedL, config_.sat_threshold, config_.sat_half_px);
    const std::vector<uint8_t> sfR =
        satFree(ir_right, seedR, config_.sat_threshold, config_.sat_half_px);

    // Marker saturation = fraction of visible corners clipped in either view.
    // Reused by the one-shot IR-exposure calibrator (no extra work: the masks
    // are already computed for the fusion gate). Reported even on RGB fallback.
    {
        int n_sat = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!(sfL[i] && sfR[i])) {
                ++n_sat;
            }
        }
        out.saturated_frac = static_cast<double>(n_sat) / static_cast<double>(n);
    }

    // ---- Reference-free path: measure both IR views directly with the
    // quadratic-form operator and run the SAME MAP once.  No reference
    // library, no enrollment, no selection — the measurement exists at any
    // orientation and from the very first frame. A non-quadratic corner_method
    // has no measurement path left, so the RGB pose passes through unchanged.
    const bool qf_mode = config_.corner_method == "quadratic_form";
    if (qf_mode) {
        ++out.refs_measured;
        std::vector<cv::Point2f> ptsL, ptsR;
        std::vector<uint8_t> okL, okR;
        std::vector<float> qL, qR;
        measureViewQf(ir_left, calib_.K_left, calib_.dist_left, R_A, t_A,
                      xyz, seedL, ptsL, okL, qL);
        measureViewQf(ir_right, calib_.K_right, calib_.dist_right, R_B, t_B,
                      xyz, seedR, ptsR, okR, qR);
        // Stereo self-calibration: correct the measured right-view x for
        // the field-dependent disparity warp of the factory calibration
        // (fitted offline from marker runs; the reference-photo path was
        // immune to it by common-mode cancellation, the direct measurement
        // exposes it).
        if (selfcal_valid_) {
            const double fxr = calib_.K_right(0, 0);
            const double cxr = calib_.K_right(0, 2);
            const double fyr = calib_.K_right(1, 1);
            const double cyr = calib_.K_right(1, 2);
            for (size_t i = 0; i < n; ++i) {
                if (!(okL[i] && okR[i])) continue;
                const double xn = (ptsR[i].x - cxr) / fxr;
                const double yn = (ptsR[i].y - cyr) / fyr;
                const double disp = (ptsL[i].x - ptsR[i].x) / 100.0;
                ptsR[i].x += static_cast<float>(
                    selfcal_[0] + selfcal_[1] * xn + selfcal_[2] * yn +
                    selfcal_[3] * xn * xn + selfcal_[4] * xn * yn +
                    selfcal_[5] * yn * yn + selfcal_[6] * disp);
            }
        }
        // Per-corner disparity correction (print/model discrepancy, stable
        // across sessions): x_R -= dxr(corner) widens/narrows the measured
        // disparity so the triangulation matches the physical print.
        if (!corner_dxr_.empty()) {
            for (size_t i = 0; i < n; ++i) {
                if (!(okL[i] && okR[i])) continue;
                const auto it = corner_dxr_.find(cornerKey(xyz[i]));
                if (it != corner_dxr_.end()) {
                    ptsR[i].x -= static_cast<float>(it->second);
                }
            }
        }
        // Incidence gate (IR-left view): mw implicitly culled grazing
        // corners via its 80-deg template gate + ZNCC; the QF measurement
        // succeeds further out on the cylinder limb, where stereo depth
        // is systematically unreliable.
        const double cos_inc_min = std::cos(65.0 * CV_PI / 180.0);
        std::vector<uint8_t> good(n, 0);
        int pairs = 0;
        double q = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (okL[i] && okR[i] &&
                std::abs(ptsL[i].y - ptsR[i].y) <
                    config_.epipolar_max_dv_px &&
                sfL[i] && sfR[i]) {
                {
                    cv::Vec3d n_m(0.0, 0.0, 1.0);
                    if (surface_.type == SurfaceModel::Type::Cylinder) {
                        const cv::Vec3d w = xyz[i] - surface_.point;
                        const cv::Vec3d rad =
                            w - w.dot(surface_.dir) * surface_.dir;
                        const double rn = cv::norm(rad);
                        if (rn > 1e-9) n_m = rad / rn;
                    } else {
                        n_m = surface_.dir;
                    }
                    const cv::Vec3d n_l = R_A * n_m;
                    const cv::Vec3d Xl = R_A * xyz[i] + t_A;
                    const double xn = cv::norm(Xl);
                    if (xn > 1e-6 &&
                        std::abs(n_l.dot(Xl / xn)) < cos_inc_min) {
                        continue;
                    }
                }
                good[i] = 1;
                ++pairs;
                const double z =
                    std::min(static_cast<double>(qL[i]),
                             static_cast<double>(qR[i]));
                q += std::pow(std::clamp(z, config_.zncc_weight_floor, 1.0),
                              2);
            }
        }

        // Pair dump for the offline stereo self-calibration.
        pair_dump_.xyz.clear();
        pair_dump_.uvL.clear();
        pair_dump_.uvR.clear();
        for (size_t i = 0; i < n; ++i) {
            if (!good[i]) continue;
            pair_dump_.xyz.push_back(xyz[i]);
            pair_dump_.uvL.push_back(ptsL[i]);
            pair_dump_.uvR.push_back(ptsR[i]);
        }

        if (pairs >= config_.min_pairs) {
            out.pairs = pairs;
            out.quality = q;
            // Per-corner sigma from the QF quality channel: poorly
            // measured corners lose depth authority.
            std::vector<float> q_pair(n, 0.0f);
            for (size_t i = 0; i < n; ++i) {
                q_pair[i] = std::min(qL[i], qR[i]);
            }
            fuseAttempt(out, good, ptsL, ptsR, xyz, uv_det, R_rgb, t_rgb,
                        pairs, n, sigma_px_eff, &q_pair);
        }
        return out;
    }
    return out;
}

// One MAP solve against the measured pair set of a single reference. Returns
// true (and fills out) when the fit gate accepts; false leaves the RGB pose
// untouched so the caller can retry with the next reference.
bool IrPoseRefiner::fuseAttempt(IrPoseRefinerResult& out,
                                const std::vector<uint8_t>& use_good,
                                const std::vector<cv::Point2f>& use_ptsL,
                                const std::vector<cv::Point2f>& use_ptsR,
                                const std::vector<cv::Vec3d>& xyz,
                                const std::vector<cv::Point2d>& uv_det,
                                const cv::Matx33d& R_rgb,
                                const cv::Vec3d& t_rgb,
                                int use_pairs,
                                size_t n,
                                double sigma_px,
                                const std::vector<float>* q_min) const
{
    // Gather the surviving pairs and triangulate over the metric baseline.
    std::vector<cv::Point2f> gl, gr;
    std::vector<cv::Vec3d> M;
    std::vector<cv::Point2d> uv_s;
    std::vector<double> sig_scale;
    gl.reserve(static_cast<size_t>(use_pairs));
    gr.reserve(static_cast<size_t>(use_pairs));
    M.reserve(static_cast<size_t>(use_pairs));
    uv_s.reserve(static_cast<size_t>(use_pairs));
    sig_scale.reserve(static_cast<size_t>(use_pairs));
    for (size_t i = 0; i < n; ++i) {
        if (use_good[i]) {
            gl.push_back(use_ptsL[i]);
            gr.push_back(use_ptsR[i]);
            M.push_back(xyz[i]);
            uv_s.push_back(uv_det[i]);
            double sc = 1.0;
            if (q_min != nullptr && i < q_min->size()) {
                const double q = std::max(0.25, static_cast<double>((*q_min)[i]));
                sc = std::clamp(1.0 / q, 1.0, 4.0);
            }
            sig_scale.push_back(sc);
        }
    }
    const std::vector<cv::Vec3d> cam =
        triangulateIrl(gl, gr, calib_, config_.depth_scale);

    // ---- MAP pose fusion: RGB reprojection + IR stereo 3D over the survivors.
    const size_t m = M.size();
    // Undistort the survivors' RGB pixels to normalised coordinates (the
    // reprojection residual); move the IR-left triangulation into the RGB camera
    // frame; build the physical anisotropic stereo covariance (depth-loose).
    std::vector<cv::Point2d> u_norm(m);
    {
        cv::Mat src(static_cast<int>(m), 1, CV_64FC2);
        for (size_t i = 0; i < m; ++i) {
            src.at<cv::Vec2d>(static_cast<int>(i)) =
                cv::Vec2d(uv_s[i].x, uv_s[i].y);
        }
        cv::Mat nrm;
        cv::undistortPoints(src, nrm, cv::Mat(calib_.K_rgb),
                            distMat(calib_.dist_rgb));
        for (size_t i = 0; i < m; ++i) {
            const cv::Vec2d v = nrm.at<cv::Vec2d>(static_cast<int>(i));
            u_norm[i] = cv::Point2d(v[0], v[1]);
        }
    }
    const cv::Matx33d R_rl = calib_.R_rgb_left;
    const cv::Vec3d t_rl = calib_.t_rgb_left_mm;
    const double f_ir = calib_.K_left(0, 0);
    const double baseline = cv::norm(calib_.t_left_right_mm);
    std::vector<cv::Vec3d> Yrgb(m);
    std::vector<cv::Matx33d> SigYinv(m);
    for (size_t i = 0; i < m; ++i) {
        Yrgb[i] = R_rl.t() * (cam[i] - t_rl);
        const double Z = std::max(cam[i][2], 1.0);
        const double sig_i = config_.sigma_ir_px * sig_scale[i];
        const double sl = Z / f_ir * sig_i;
        const double sz = Z * Z / (f_ir * baseline) * sig_i;
        const cv::Matx33d S_irl(sl * sl, 0.0, 0.0,
                                0.0, sl * sl, 0.0,
                                0.0, 0.0, sz * sz);
        cv::invert(R_rl.t() * S_irl * R_rl, SigYinv[i], cv::DECOMP_SVD);
    }
    const MapFuseResult mf =
        mapPoseFuse(M, u_norm, Yrgb, SigYinv, R_rgb, t_rgb, calib_.K_rgb(0, 0),
                    sigma_px, config_.w3d);

    // Fit-quality gate: reject an unreliable fusion -> the caller retries
    // with the next admissible reference (or keeps the RGB pose).
    double ss = 0.0;
    for (size_t i = 0; i < m; ++i) {
        const cv::Vec3d r = mf.R * M[i] + mf.t - Yrgb[i];
        ss += r.dot(r);
    }
    const double fit_rms = m > 0 ? std::sqrt(ss / static_cast<double>(m)) : 1e9;
    out.fit_rms_mm = fit_rms;
    const cv::Vec3d dt_jump = mf.t - t_rgb;
    if (!mf.ok || use_pairs < config_.min_pairs ||
        fit_rms > config_.fit_gate_rms_mm ||
        cv::norm(dt_jump) > config_.fit_gate_max_trans_jump_mm) {
        return false;   // gate-rejected: RGB pose stays unless a retry fuses
    }

    cv::Mat rout;
    cv::Rodrigues(cv::Mat(mf.R), rout);
    out.rvec = {rout.at<double>(0), rout.at<double>(1), rout.at<double>(2)};
    out.tvec = {mf.t[0], mf.t[1], mf.t[2]};
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            out.cov[static_cast<size_t>(6 * r + c)] = mf.cov(r, c);
        }
    }
    out.dtz_mm = dt_jump[2];   // legacy diagnostic: depth component of the shift
    out.mode = IrFusionMode::Depth;
    out.applied = true;
    return true;
}

}  // namespace hydramarker
