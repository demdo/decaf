#include "drill_pose_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

// Line references "S3:n" point to smoother3.py, "RL:n" to rl_smoother3.py and
// "BENCH:n" to iekf_bench.cpp (the offline prototype of this estimator).

namespace hydramarker {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg2Rad = kPi / 180.0;      // bit-equal to numpy.radians (x * (pi / 180))
constexpr double kFrameS = 1.0 / 30.0;        // T0 of the base noise (S3:41)
constexpr double kMinDt = 1e-3;               // S3:244
constexpr double kNu1Fallback = 40.0;         // unknown motion [px/frame] (S3:245)
constexpr double kSigPxFallback = 0.10;       // sigma_px without IR stage (S3:188)
constexpr double kRsPeriodMs = 33.37;         // S3:210
constexpr double kRsFdtMinMs = 1.0;           // S3:209
constexpr double kRsFdtMaxMs = 120.0;
constexpr double kReg = 1e-9;                 // S3:298, 306
constexpr double kStepStop = 1e-7;            // S3:302
constexpr double kQzMin = 1e-6;               // S3:90
constexpr double kHuberEps = 1e-12;           // S3:80
constexpr double kJumpRad = kPi / 2.0;        // new: pose jump > 90 deg breaks the chain
constexpr int kHist = 4;                      // S3:195
constexpr size_t kReserveCorners = 128;
constexpr size_t kReservePairs = 128;

typedef double M3[3][3];

void skewm(const double* v, M3 S)
{
    S[0][0] = 0; S[0][1] = -v[2]; S[0][2] = v[1];
    S[1][0] = v[2]; S[1][1] = 0; S[1][2] = -v[0];
    S[2][0] = -v[1]; S[2][1] = v[0]; S[2][2] = 0;
}

void matmul3(const M3 A, const M3 B, M3 C)
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
    }
}

void transpose3(const M3 A, M3 B)
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) B[i][j] = A[j][i];
}

void copy3(const M3 A, M3 B)
{
    std::memcpy(B, A, sizeof(M3));
}

void expso3(const double* w, M3 R)
{
    const double th = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    M3 K, K2;
    skewm(w, K);
    matmul3(K, K, K2);
    double a, b;
    if (th < 1e-10) {
        a = 1;
        b = 0;
    } else {
        a = std::sin(th) / th;
        b = (1 - std::cos(th)) / (th * th);
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) R[i][j] = (i == j ? 1.0 : 0.0) + a * K[i][j] + b * K2[i][j];
}

// No special case near pi (as S3:64-70); the chain is broken before a jump
// that large (pose jump guard), and a non-finite result is caught (event 4).
void logso3(const M3 R, double* w)
{
    double c = (R[0][0] + R[1][1] + R[2][2] - 1) / 2;
    c = std::max(-1.0, std::min(1.0, c));
    const double th = std::acos(c);
    const double v[3] = {R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]};
    const double s = th < 1e-8 ? 0.5 : th / (2 * std::sin(th));
    for (int i = 0; i < 3; i++) w[i] = s * v[i];
}

// Cholesky of a 6x6 SPD matrix. A NaN passes (NaN <= 0 is false); callers
// check the result for finiteness.
bool chol6(const double A[36], double L[36])
{
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j <= i; j++) {
            double s = A[i * 6 + j];
            for (int k = 0; k < j; k++) s -= L[i * 6 + k] * L[j * 6 + k];
            if (i == j) {
                if (s <= 0) return false;
                L[i * 6 + i] = std::sqrt(s);
            } else {
                L[i * 6 + j] = s / L[j * 6 + j];
            }
        }
    }
    return true;
}

void cholsolve6(const double L[36], const double* b, double* x)
{
    double y[6];
    for (int i = 0; i < 6; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[i * 6 + k] * y[k];
        y[i] = s / L[i * 6 + i];
    }
    for (int i = 5; i >= 0; i--) {
        double s = y[i];
        for (int k = i + 1; k < 6; k++) s -= L[k * 6 + i] * x[k];
        x[i] = s / L[i * 6 + i];
    }
}

bool inv6(const double A[36], double Ai[36])
{
    double L[36] = {0};
    if (!chol6(A, L)) return false;
    for (int j = 0; j < 6; j++) {
        double e[6] = {0}, x[6];
        e[j] = 1;
        cholsolve6(L, e, x);
        for (int i = 0; i < 6; i++) Ai[i * 6 + j] = x[i];
    }
    return true;
}

// Cofactor inverse (BENCH:63-68). New: false if the determinant is 0 or the
// result is not finite.
bool inv3(const M3 A, M3 B)
{
    const double d = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (d == 0.0 || !std::isfinite(d)) return false;
    B[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / d;
    B[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / d;
    B[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / d;
    B[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) / d;
    B[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / d;
    B[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / d;
    B[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / d;
    B[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / d;
    B[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / d;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (!std::isfinite(B[i][j])) return false;
    return true;
}

bool allFinite(const double* v, int n)
{
    for (int i = 0; i < n; i++)
        if (!std::isfinite(v[i])) return false;
    return true;
}

bool finite3(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

bool finite2(const cv::Vec2d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]);
}

bool finite33(const cv::Matx33d& m)
{
    return allFinite(m.val, 9);
}

// boxminus(Ra, ta, Rb, tb) = [Log(Ra^T Rb); Ra^T (tb - ta)] (S3:134-136)
void boxminus(const M3 Ra, const double* ta, const M3 Rb, const double* tb, double d[6])
{
    M3 RaT, D;
    transpose3(Ra, RaT);
    matmul3(RaT, Rb, D);
    logso3(D, d);
    for (int a = 0; a < 3; a++)
        d[3 + a] = RaT[a][0] * (tb[0] - ta[0]) + RaT[a][1] * (tb[1] - ta[1]) + RaT[a][2] * (tb[2] - ta[2]);
}

// numpy.median of a scratch copy: middle value, or the mean of the two middle
// values for an even count.
double medianInPlace(std::vector<double>& v)
{
    const size_t n = v.size();
    const size_t h = n / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(h), v.end());
    const double hi = v[h];
    if (n % 2 == 1) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(h));
    return (lo + hi) / 2.0;
}

}  // namespace

// ============================================================================
// DrillPoseEstimator
// ============================================================================

DrillPoseEstimator::DrillPoseEstimator(const DrillEstimatorConfig& cfg)
    : cfg_(cfg)
{
    const DrillEstimatorConfig& c = cfg_;
    const double all_d[] = {
        c.sig_t_mm, c.sig_r_deg, c.adapt_px, c.db3_px, c.gain_t, c.gain_r_deg, c.s_rgb, c.s_ir,
        c.huber_k, c.gap_s, c.rs_px, c.ir_gated, c.ir_fit_max_mm, c.ir_fit_fac, c.sig_tip_max_mm,
        c.w3d, c.sigma_ir_px, c.row_ms, c.prior_max_mm, c.hide_sig_mm};
    bool ok = allFinite(all_d, static_cast<int>(sizeof(all_d) / sizeof(all_d[0])));
    ok = ok && c.sig_t_mm >= 0.0 && c.sig_r_deg >= 0.0 && c.gain_t >= 0.0 && c.gain_r_deg >= 0.0
        && c.vmin >= 1 && c.s_rgb > 0.0 && c.s_ir > 0.0 && c.huber_k > 0.0 && c.n_iter >= 1
        && c.gap_s > 0.0 && c.ir_gated > 0.0 && c.ir_fit_max_mm >= 0.0 && c.ir_fit_fac > 0.0
        && c.min_start >= 0 && c.warm >= 0 && c.w3d > 0.0 && c.sigma_ir_px > 0.0 && c.row_ms >= 0.0
        && c.prior_max_mm >= 0.0 && c.hide_sig_mm >= 0.0;
    cfg_ok_ = ok;

    for (auto& h : hist_) h.uv.reserve(kReserveCorners);
    sort_.reserve(kReserveCorners);
    keep_.reserve(kReserveCorners);
    cur_id_.reserve(kReserveCorners);
    cur_uv_.reserve(kReserveCorners);
    Xr_.reserve(kReserveCorners);
    ur_.reserve(kReserveCorners);
    rows_.reserve(kReserveCorners);
    rows_tmp_.reserve(kReserveCorners);
    vU_.reserve(kReserveCorners);
    vD_.reserve(kReserveCorners);
    vq1_.reserve(kReserveCorners);
    vq2_.reserve(kReserveCorners);
    Xi_.reserve(kReservePairs);
    Yi_.reserve(kReservePairs);
    Si_.reserve(kReservePairs);
    Wi_.reserve(kReservePairs);
}

void DrillPoseEstimator::reset()
{
    have_ = false;
    prev_t_ = 0.0;
}

bool DrillPoseEstimator::setToolTip(const cv::Vec3d& p_tip_mm)
{
    if (!finite3(p_tip_mm)) return false;
    p_tip_ = p_tip_mm;
    tip_set_ = true;
    return true;
}

DrillEstimatorState DrillPoseEstimator::debugState() const
{
    DrillEstimatorState s;
    s.have = have_;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) s.R(i, j) = R_[i][j];
        s.t[i] = t_[i];
    }
    for (int i = 0; i < 36; i++) s.P[i] = P_[i];
    s.prev_t = prev_t_;
    s.since = since_;
    s.starting = starting_;
    s.hist_n = hist_n_;
    return s;
}

void DrillPoseEstimator::debugSetState(const cv::Matx33d& R, const cv::Vec3d& t,
                                       const std::array<double, 36>& P, double prev_t)
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) R_[i][j] = R(i, j);
        t_[i] = t[i];
    }
    for (int i = 0; i < 36; i++) P_[i] = P[i];
    if (std::isfinite(prev_t)) prev_t_ = prev_t;
    have_ = true;
}

DrillEstimatorOutput DrillPoseEstimator::update(const DrillEstimatorFrame& in)
{
    DrillEstimatorOutput out;
    try {
        updateImpl(in, out);
    } catch (...) {
        // Only an allocation in a scratch vector can throw. The history slot
        // being written is not read before it is complete (the ring head is
        // advanced afterwards), so only the chain is dropped, as reset().
        have_ = false;
        out = DrillEstimatorOutput();
        out.event = 7;
    }
    return out;
}

// Affine motion statistic (S3:119-131): RMS of the affine displacement field
// fitted between the frame b back in the history and the current frame, / b.
// Least squares with the semantics of numpy.linalg.lstsq(rcond=None): the
// fitted values are the orthogonal projection of the displacements onto the
// column space of A = [U0 - mean(U0), 1]; singular values of A that are
// <= eps * max(m, 3) * s_max count as 0. As the centred columns are orthogonal
// to the ones column, the singular values are sqrt(m) and the square roots of
// the eigenvalues of Mx = Uc^T Uc (2x2, closed-form eigenvectors).
double DrillPoseEstimator::vaff(int b)
{
    if (hist_n_ < b) return kNaN;
    const std::vector<std::pair<int, cv::Vec2d>>& prev = hist_[(hist_head_ - b + kHist) % kHist].uv;
    vU_.clear();
    vD_.clear();
    for (size_t i = 0; i < cur_id_.size(); i++) {
        const int key = cur_id_[i];
        auto it = std::lower_bound(prev.begin(), prev.end(), key,
                                   [](const std::pair<int, cv::Vec2d>& e, int k) { return e.first < k; });
        if (it == prev.end() || it->first != key) continue;
        vU_.push_back(it->second);
        vD_.push_back(cv::Vec2d(cur_uv_[i][0] - it->second[0], cur_uv_[i][1] - it->second[1]));
    }
    const size_t m = vU_.size();
    if (static_cast<long long>(m) < static_cast<long long>(std::max(cfg_.vmin, 3))) return kNaN;
    const double md = static_cast<double>(m);

    // centre U0 (sequential sum / m as numpy's axis-0 mean); a second pass
    // removes the rounding residue of the mean (same column space)
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < m; i++) {
        mx += vU_[i][0];
        my += vU_[i][1];
    }
    mx /= md;
    my /= md;
    double rx = 0.0, ry = 0.0;
    for (size_t i = 0; i < m; i++) {
        vU_[i][0] -= mx;
        vU_[i][1] -= my;
        rx += vU_[i][0];
        ry += vU_[i][1];
    }
    rx /= md;
    ry /= md;
    double a = 0.0, bxy = 0.0, c = 0.0, dmx = 0.0, dmy = 0.0;
    for (size_t i = 0; i < m; i++) {
        vU_[i][0] -= rx;
        vU_[i][1] -= ry;
        a += vU_[i][0] * vU_[i][0];
        bxy += vU_[i][0] * vU_[i][1];
        c += vU_[i][1] * vU_[i][1];
        dmx += vD_[i][0];
        dmy += vD_[i][1];
    }
    dmx /= md;
    dmy /= md;

    // eigenvectors of Mx: e1 = (cs, sn) (larger eigenvalue), e2 = (-sn, cs);
    // the eigenvalues are taken from the projected data (accurate also for a
    // nearly collinear point set), q2 re-orthogonalised against q1
    const double th = 0.5 * std::atan2(2.0 * bxy, a - c);
    const double cs = std::cos(th), sn = std::sin(th);
    vq1_.resize(m);
    vq2_.resize(m);
    double l1 = 0.0;
    for (size_t i = 0; i < m; i++) {
        vq1_[i] = cs * vU_[i][0] + sn * vU_[i][1];
        vq2_[i] = -sn * vU_[i][0] + cs * vU_[i][1];
        l1 += vq1_[i] * vq1_[i];
    }
    if (l1 > 0.0) {
        double d12 = 0.0;
        for (size_t i = 0; i < m; i++) d12 += vq1_[i] * vq2_[i];
        const double r12 = d12 / l1;
        for (size_t i = 0; i < m; i++) vq2_[i] -= r12 * vq1_[i];
    }
    double l2 = 0.0;
    for (size_t i = 0; i < m; i++) l2 += vq2_[i] * vq2_[i];

    const double eps = std::numeric_limits<double>::epsilon();
    const double smax = std::max(std::sqrt(md), std::sqrt(std::max(l1, l2)));
    const double tol = eps * static_cast<double>(std::max<size_t>(m, 3)) * smax;
    const bool use1 = std::sqrt(l1) > tol;
    const bool use2 = std::sqrt(l2) > tol;
    double p1x = 0.0, p1y = 0.0, p2x = 0.0, p2y = 0.0;
    if (use1 || use2) {
        for (size_t i = 0; i < m; i++) {
            p1x += vq1_[i] * vD_[i][0];
            p1y += vq1_[i] * vD_[i][1];
            p2x += vq2_[i] * vD_[i][0];
            p2y += vq2_[i] * vD_[i][1];
        }
    }
    if (use1) {
        p1x /= l1;
        p1y /= l1;
    } else {
        p1x = p1y = 0.0;
    }
    if (use2) {
        p2x /= l2;
        p2y /= l2;
    } else {
        p2x = p2y = 0.0;
    }
    double ss = 0.0;
    for (size_t i = 0; i < m; i++) {
        const double px = dmx + vq1_[i] * p1x + vq2_[i] * p2x;
        const double py = dmy + vq1_[i] * p1y + vq2_[i] * p2y;
        ss += px * px + py * py;
    }
    const double v = std::sqrt(ss / md) / static_cast<double>(b);
    return std::isfinite(v) ? v : kNaN;
}

// Gauss-Newton normal equations of all measurement factors at (R, t)
// (S3:83-116, BENCH:76-119): RGB reprojection per corner with f for both
// axes, IR 3D per pair with W_j; Huber on the norm of the normalised residual.
void DrillPoseEstimator::measTerms(const M3 R, const double t[3], double f, double sig,
                                   double H[36], double g[6]) const
{
    std::memset(H, 0, 36 * sizeof(double));
    std::memset(g, 0, 6 * sizeof(double));
    const double k = cfg_.huber_k;
    const size_t n = Xr_.size();
    for (size_t i = 0; i < n; i++) {
        const double X[3] = {Xr_[i][0], Xr_[i][1], Xr_[i][2]};
        double q[3];
        for (int a = 0; a < 3; a++) q[a] = R[a][0] * X[0] + R[a][1] * X[1] + R[a][2] * X[2] + t[a];
        const double z = std::max(q[2], kQzMin);
        const double r[2] = {f * (q[0] / z - ur_[i][0]), f * (q[1] / z - ur_[i][1])};
        const double Jp[2][3] = {{f / z, 0, -f * q[0] / (z * z)}, {0, f / z, -f * q[1] / (z * z)}};
        M3 S, RS;
        skewm(X, S);
        matmul3(R, S, RS);
        double J[2][6];
        for (int a = 0; a < 2; a++) {
            for (int cc = 0; cc < 3; cc++) {
                double s1 = 0, s2 = 0;
                for (int bb = 0; bb < 3; bb++) {
                    s1 -= Jp[a][bb] * RS[bb][cc];
                    s2 += Jp[a][bb] * R[bb][cc];
                }
                J[a][cc] = s1;
                J[a][3 + cc] = s2;
            }
        }
        const double e = std::sqrt(r[0] * r[0] + r[1] * r[1]) / sig;
        const double w = (e <= k ? 1.0 : k / std::max(e, kHuberEps)) / (sig * sig);
        for (int a = 0; a < 6; a++) {
            g[a] += w * (J[0][a] * r[0] + J[1][a] * r[1]);
            for (int bb = a; bb < 6; bb++) H[a * 6 + bb] += w * (J[0][a] * J[0][bb] + J[1][a] * J[1][bb]);
        }
    }
    const size_t m = Xi_.size();
    for (size_t i = 0; i < m; i++) {
        const double X[3] = {Xi_[i][0], Xi_[i][1], Xi_[i][2]};
        const double* W = Wi_[i].val;
        double r[3];
        for (int a = 0; a < 3; a++)
            r[a] = R[a][0] * X[0] + R[a][1] * X[1] + R[a][2] * X[2] + t[a] - Yi_[i][a];
        M3 S, RS;
        skewm(X, S);
        matmul3(R, S, RS);
        double J[3][6];
        for (int a = 0; a < 3; a++) {
            for (int cc = 0; cc < 3; cc++) {
                J[a][cc] = -RS[a][cc];
                J[a][3 + cc] = R[a][cc];
            }
        }
        double Wr[3], WJ[3][6];
        for (int a = 0; a < 3; a++) {
            Wr[a] = W[3 * a] * r[0] + W[3 * a + 1] * r[1] + W[3 * a + 2] * r[2];
            for (int cc = 0; cc < 6; cc++)
                WJ[a][cc] = W[3 * a] * J[0][cc] + W[3 * a + 1] * J[1][cc] + W[3 * a + 2] * J[2][cc];
        }
        const double e = std::sqrt(std::max(0.0, r[0] * Wr[0] + r[1] * Wr[1] + r[2] * Wr[2]));
        const double w = e <= k ? 1.0 : k / std::max(e, kHuberEps);
        for (int a = 0; a < 6; a++) {
            g[a] += w * (WJ[0][a] * r[0] + WJ[1][a] * r[1] + WJ[2][a] * r[2]);
            for (int bb = a; bb < 6; bb++)
                H[a * 6 + bb] += w * (J[0][a] * WJ[0][bb] + J[1][a] * WJ[1][bb] + J[2][a] * WJ[2][bb]);
        }
    }
    for (int a = 0; a < 6; a++)
        for (int bb = 0; bb < a; bb++) H[a * 6 + bb] = H[bb * 6 + a];
}

void DrillPoseEstimator::updateImpl(const DrillEstimatorFrame& in, DrillEstimatorOutput& out)
{
    const DrillEstimatorConfig& c = cfg_;

    // ---- 0 configuration (new) ----
    if (!cfg_ok_) {
        out.event = 10;
        return;
    }
    const double f = in.f;
    const double fy = in.fy;

    // ---- 1 input rules (new; no effect on the validation data) ----
    // RGB: a corner with id < 0 or a non-finite X, u or row is dropped; a
    // duplicate id: the last one counts (dict semantics of S3:190), counted
    // once in n_dup_id. Surplus entries of inconsistent list lengths count as
    // dropped.
    const size_t n_all = std::min(std::min(in.X.size(), in.u.size()), std::min(in.row_px.size(), in.id.size()));
    const size_t n_max = std::max(std::max(in.X.size(), in.u.size()), std::max(in.row_px.size(), in.id.size()));
    int drop_rgb = static_cast<int>(n_max - n_all);
    sort_.clear();
    keep_.assign(n_all, 0);
    for (size_t i = 0; i < n_all; i++) {
        if (in.id[i] < 0 || !finite3(in.X[i]) || !finite2(in.u[i]) || !std::isfinite(in.row_px[i])) {
            ++drop_rgb;
            continue;
        }
        sort_.emplace_back(in.id[i], static_cast<int>(i));
    }
    std::sort(sort_.begin(), sort_.end());
    int dup = 0;
    size_t w = 0;
    for (size_t kk = 0; kk < sort_.size(); kk++) {
        if (kk + 1 < sort_.size() && sort_[kk + 1].first == sort_[kk].first) {
            ++dup;
            continue;
        }
        keep_[static_cast<size_t>(sort_[kk].second)] = 1;
        sort_[w++] = sort_[kk];
    }
    sort_.resize(w);   // unique ids, ascending: order of the new history slot

    cur_id_.clear();
    cur_uv_.clear();
    Xr_.clear();
    ur_.clear();
    rows_.clear();
    for (size_t i = 0; i < n_all; i++) {
        if (!keep_[i]) continue;
        cur_id_.push_back(in.id[i]);
        cur_uv_.push_back(cv::Vec2d(in.u[i][0] * f, in.u[i][1] * f));
        Xr_.push_back(in.X[i]);
        ur_.push_back(in.u[i]);
        rows_.push_back(in.row_px[i]);
    }
    const int N = static_cast<int>(Xr_.size());

    // IR: a pair with a non-finite X, Y or S is dropped
    const size_t m_all = std::min(std::min(in.Xi.size(), in.Yi.size()), in.Si.size());
    const size_t m_max = std::max(std::max(in.Xi.size(), in.Yi.size()), in.Si.size());
    int drop_ir = static_cast<int>(m_max - m_all);
    Xi_.clear();
    Yi_.clear();
    Si_.clear();
    for (size_t j = 0; j < m_all; j++) {
        if (!finite3(in.Xi[j]) || !finite3(in.Yi[j]) || !finite33(in.Si[j])) {
            ++drop_ir;
            continue;
        }
        Xi_.push_back(in.Xi[j]);
        Yi_.push_back(in.Yi[j]);
        Si_.push_back(in.Si[j]);
    }

    // sigma_px: engine value, else 0.10 (S3:188; new: also for a value <= 0)
    const double sig = (std::isfinite(in.sig_px) && in.sig_px > 0.0) ? in.sig_px : kSigPxFallback;
    const double sig_c = sig * c.s_rgb;   // S3:221

    out.n_rgb = N;
    out.n_drop_rgb = drop_rgb;
    out.n_dup_id = dup;
    out.sig_px_used = sig;

    // ---- 2 coherent image motion, before any gate (S3:190-195) ----
    const double v1 = vaff(1);
    const double v3 = c.db3_px > 0.0 ? vaff(3) : kNaN;
    {
        std::vector<std::pair<int, cv::Vec2d>>& slot = hist_[hist_head_].uv;
        slot.clear();
        for (const auto& p : sort_) {
            const cv::Vec2d& u = in.u[static_cast<size_t>(p.second)];
            slot.emplace_back(p.first, cv::Vec2d(u[0] * f, u[1] * f));
        }
        hist_head_ = (hist_head_ + 1) % kHist;
        hist_n_ = std::min(hist_n_ + 1, kHist);
    }
    out.nu1 = v1;
    out.nu3 = v3;

    // ---- 3 tool tip and frame time (new; after the history) ----
    if (!tip_set_) {
        out.n_ir = static_cast<int>(Xi_.size());
        out.n_drop_ir = drop_ir;
        out.event = 5;
        return;
    }
    if (!(std::isfinite(in.t_s) && in.t_s >= 0.0)) {
        out.n_ir = static_cast<int>(Xi_.size());
        out.n_drop_ir = drop_ir;
        out.event = 6;
        return;
    }

    // ---- 4 rolling-shutter-consistent time model (S3:205-220) ----
    int M = static_cast<int>(Xi_.size());
    const cv::Vec2d vm = in.vmed;
    if (c.rs_px >= 0.0 && N > 0 && std::isfinite(vm[0]) && std::isfinite(in.fdt_ms)
        && std::isfinite(in.toff_ms) && in.fdt_ms > kRsFdtMinMs && in.fdt_ms < kRsFdtMaxMs
        && std::hypot(vm[0], vm[1]) * kRsPeriodMs / in.fdt_ms >= c.rs_px) {
        rows_tmp_.assign(rows_.begin(), rows_.end());
        const double y_ref = medianInPlace(rows_tmp_);
        for (int n = 0; n < N; n++) {
            const double s = c.row_ms * (rows_[static_cast<size_t>(n)] - y_ref) / in.fdt_ms;
            ur_[static_cast<size_t>(n)][0] -= vm[0] * s / f;
            ur_[static_cast<size_t>(n)][1] -= vm[1] * s / fy;
        }
        if (M > 0) {
            const double k_ir = (in.toff_ms + c.row_ms * y_ref) / in.fdt_ms;
            const double shx = vm[0] * k_ir;
            const double shy = vm[1] * k_ir;
            for (int j = 0; j < M; j++) {
                cv::Vec3d& Y = Yi_[static_cast<size_t>(j)];
                Y[0] += Y[2] / f * shx;
                Y[1] += Y[2] / fy * shy;
            }
        }
        out.rs_on = true;
    }

    // ---- 5 IR weights W_j = inv(S_j / w3d (s_ir fac)^2) (S3:223-230) ----
    Wi_.clear();
    if (M > 0) {
        double fac = in.ir_applied ? 1.0 : c.ir_gated;
        if (c.ir_fit_max_mm > 0.0 && std::isfinite(in.ir_fit_mm) && in.ir_fit_mm > c.ir_fit_max_mm) {
            fac = std::max(fac, c.ir_fit_fac);
            out.ir_off = true;
        }
        const double sc = (c.s_ir * fac) * (c.s_ir * fac);
        size_t wj = 0;
        for (size_t j = 0; j < static_cast<size_t>(M); j++) {
            M3 A, Wm;
            for (int a = 0; a < 3; a++)
                for (int bb = 0; bb < 3; bb++) A[a][bb] = Si_[j](a, bb) / c.w3d * sc;
            if (!inv3(A, Wm)) {   // new: singular or non-finite -> pair dropped
                ++drop_ir;
                continue;
            }
            Xi_[wj] = Xi_[j];
            Yi_[wj] = Yi_[j];
            Wi_.push_back(cv::Matx33d(Wm[0][0], Wm[0][1], Wm[0][2],
                                      Wm[1][0], Wm[1][1], Wm[1][2],
                                      Wm[2][0], Wm[2][1], Wm[2][2]));
            ++wj;
        }
        Xi_.resize(wj);
        Yi_.resize(wj);
        M = static_cast<int>(wj);
    }
    out.n_ir = M;
    out.n_drop_ir = drop_ir;

    // ---- 6 chain link (S3:232-234; time jump new) ----
    const double dt = have_ ? in.t_s - prev_t_ : kInf;
    bool link = false;
    int restart = 0;
    if (have_ && (dt < 0.0 || !std::isfinite(dt))) {
        restart = 6;
    } else {
        link = have_ && !in.reset_before && dt < c.gap_s;
    }
    bool chained = link;
    if (!link) restart = !have_ ? 1 : (in.reset_before ? 3 : (dt >= c.gap_s ? 2 : restart));

    // chain-start gate: a frame without prior support needs min_start corners
    // (S3:235-239): no output, no state; since/starting unchanged
    auto start_gate = [&]() -> bool {
        if (!link && c.min_start > 0 && N < c.min_start) {
            have_ = false;
            out.event = 2;
            out.restart = restart;
            return true;
        }
        return false;
    };

    // ---- 7 ----
    if (start_gate()) return;
    M3 Re;
    expso3(in.rvec.data(), Re);   // S3:240
    const double te[3] = {in.tvec[0], in.tvec[1], in.tvec[2]};

    // ---- 7b pose jump > 90 deg between the state and the engine pose (new) ----
    if (chained) {
        M3 RT, D;
        transpose3(R_, RT);
        matmul3(RT, Re, D);
        double ca = (D[0][0] + D[1][1] + D[2][2] - 1) / 2;
        ca = std::max(-1.0, std::min(1.0, ca));
        if (std::acos(ca) > kJumpRad) {
            chained = link = false;
            restart = 5;
            if (start_gate()) return;
        }
    }

    // ---- 8/9 prediction: random walk, adaptive opening (S3:242-256) ----
    double Lb[36] = {0};
    M3 Rb;
    double tb[3];
    if (chained) {
        const double s = std::sqrt(std::max(dt, kMinDt) / kFrameS);
        double v = c.adapt_px > 0.0 ? (std::isfinite(v1) ? v1 : kNu1Fallback) - c.adapt_px : 0.0;
        if (c.db3_px > 0.0 && std::isfinite(v3)) v = std::max(v, v3 - c.db3_px);
        v = std::max(0.0, v);
        const double Z = te[2];
        double st = c.sig_t_mm * s;
        double sr = (c.sig_r_deg * kDeg2Rad) * s;
        if (c.adapt_px > 0.0) {
            st = std::hypot(st, c.gain_t * v * Z / f);
            sr = std::hypot(sr, (c.gain_r_deg * v) * kDeg2Rad);
        }
        double Pb[36];
        std::memcpy(Pb, P_, sizeof(Pb));
        for (int a = 0; a < 3; a++) {
            Pb[a * 7] += sr * sr;
            Pb[(a + 3) * 7] += st * st;
        }
        if (!inv6(Pb, Lb) || !allFinite(Lb, 36)) {
            // new: prior not invertible -> chain restart (restart 7)
            std::fill(Lb, Lb + 36, 0.0);
            chained = link = false;
            restart = 7;
            if (start_gate()) return;
        } else if (c.prior_max_mm > 0.0 && st > c.prior_max_mm) {
            // RL:259-266: the opened prior is uninformative -> chain restart
            chained = link = false;
            restart = 4;
            if (start_gate()) return;
        }
        if (chained) {
            copy3(R_, Rb);
            std::memcpy(tb, t_, sizeof(tb));
        }
    }
    // ---- 10 no prior (S3:283-285) ----
    if (!chained) {
        copy3(Re, Rb);
        std::memcpy(tb, te, sizeof(tb));
        std::fill(Lb, Lb + 36, 0.0);
    }

    // ---- 11/12 iterated update, start at the engine pose (S3:287-303) ----
    // The prior Jacobian is I (first-order approximation as in the prototype).
    M3 Rc;
    double tc[3];
    copy3(Re, Rc);
    std::memcpy(tc, te, sizeof(tc));
    double H[36], g[6], Hm[36], gm[6], L[36];
    int iters = 0;
    for (int it = 0; it < c.n_iter; it++) {
        measTerms(Rc, tc, f, sig_c, Hm, gm);
        double d[6];
        boxminus(Rb, tb, Rc, tc, d);
        for (int a = 0; a < 6; a++) {
            double s = gm[a];
            for (int bb = 0; bb < 6; bb++) s += Lb[a * 6 + bb] * d[bb];
            g[a] = -s;
        }
        for (int a = 0; a < 36; a++) H[a] = Hm[a] + Lb[a];
        for (int a = 0; a < 6; a++) H[a * 7] += kReg;
        double step[6] = {0};
        if (chol6(H, L)) cholsolve6(L, g, step);
        M3 E, Rn;
        expso3(step, E);
        matmul3(Rc, E, Rn);
        for (int a = 0; a < 3; a++) tc[a] += Rc[a][0] * step[3] + Rc[a][1] * step[4] + Rc[a][2] * step[5];
        copy3(Rn, Rc);
        ++iters;
        double nn = 0;
        for (int a = 0; a < 6; a++) nn += step[a] * step[a];
        if (std::sqrt(nn) < kStepStop) break;
    }
    out.iters = iters;

    // ---- 13 covariance (S3:304-309); non-finite result new ----
    measTerms(Rc, tc, f, sig_c, Hm, gm);
    for (int a = 0; a < 36; a++) H[a] = Hm[a] + Lb[a];
    for (int a = 0; a < 6; a++) H[a * 7] += kReg;
    double P[36];
    if (!inv6(H, P)) {
        have_ = false;
        out.event = 3;
        return;
    }
    if (!allFinite(&Rc[0][0], 9) || !allFinite(tc, 3) || !allFinite(P, 36)) {
        have_ = false;
        out.event = 4;
        return;
    }

    // ---- 14 NIS (diagnostics, S3:310-313) ----
    double nis = kNaN;
    if (chained) {
        double d[6];
        boxminus(Rb, tb, Rc, tc, d);
        double s = 0.0;
        for (int j = 0; j < 6; j++) {
            double dl = 0.0;
            for (int i = 0; i < 6; i++) dl += d[i] * Lb[i * 6 + j];
            s += dl * d[j];
        }
        nis = s;
    }

    // ---- 15 state (S3:314-316) ----
    copy3(Rc, R_);
    std::memcpy(t_, tc, sizeof(t_));
    std::memcpy(P_, P, sizeof(P_));
    have_ = true;
    since_ = link ? since_ + 1 : 0;
    prev_t_ = in.t_s;

    // ---- 16 tip and its sigma (S3:317-319) ----
    const double pt[3] = {p_tip_[0], p_tip_[1], p_tip_[2]};
    double tip[3];
    for (int a = 0; a < 3; a++) tip[a] = Rc[a][0] * pt[0] + Rc[a][1] * pt[1] + Rc[a][2] * pt[2] + tc[a];
    M3 S, RS;
    skewm(pt, S);
    matmul3(Rc, S, RS);
    double J[3][6];
    for (int a = 0; a < 3; a++) {
        for (int bb = 0; bb < 3; bb++) {
            J[a][bb] = -RS[a][bb];
            J[a][3 + bb] = Rc[a][bb];
        }
    }
    double tr = 0;
    for (int a = 0; a < 3; a++)
        for (int p = 0; p < 6; p++)
            for (int q = 0; q < 6; q++) tr += J[a][p] * P[p * 6 + q] * J[a][q];
    const double stip = std::sqrt(tr);

    // ---- 17 start-up gate (S3:321-325) and display rule (RL:336-337) ----
    if (since_ == 0) starting_ = c.warm > 0;
    if (starting_ && (since_ >= c.warm || (c.sig_tip_max_mm > 0.0 && stip <= c.sig_tip_max_mm)))
        starting_ = false;
    bool shown = !starting_;
    if (c.hide_sig_mm > 0.0 && stip > c.hide_sig_mm) shown = false;

    // ---- 18 output (S3:326-332) ----
    out.valid = true;
    out.shown = shown;
    out.chained = chained;
    out.since = since_;
    out.event = link ? 0 : 1;
    out.restart = link ? 0 : restart;
    for (int a = 0; a < 3; a++) {
        for (int bb = 0; bb < 3; bb++) out.R(a, bb) = Rc[a][bb];
        out.t[a] = tc[a];
        out.tip[a] = tip[a];
    }
    for (int a = 0; a < 36; a++) out.P[static_cast<size_t>(a)] = P[a];
    out.sigma_tip_mm = stip;
    out.nis = nis;
}

// ============================================================================
// DrillVmed
// ============================================================================

DrillVmed::DrillVmed()
{
    prev_.reserve(kReserveCorners);
    cur_.reserve(kReserveCorners);
    sort_.reserve(kReserveCorners);
    dx_.reserve(kReserveCorners);
    dy_.reserve(kReserveCorners);
}

void DrillVmed::reset()
{
    prev_.clear();
    have_prev_ = false;
    prev_frame_ = -1;
}

cv::Vec2d DrillVmed::update(const std::vector<int>& id, const std::vector<cv::Vec2d>& uv_raw,
                            int frame_index)
{
    // current frame: unique ids (the last occurrence counts), ascending
    sort_.clear();
    const size_t n = std::min(id.size(), uv_raw.size());
    for (size_t i = 0; i < n; i++) {
        if (id[i] < 0 || !finite2(uv_raw[i])) continue;
        sort_.emplace_back(id[i], static_cast<int>(i));
    }
    std::sort(sort_.begin(), sort_.end());
    cur_.clear();
    for (size_t k = 0; k < sort_.size(); k++) {
        if (k + 1 < sort_.size() && sort_[k + 1].first == sort_[k].first) continue;
        cur_.emplace_back(sort_[k].first, uv_raw[static_cast<size_t>(sort_[k].second)]);
    }

    cv::Vec2d res(kNaN, kNaN);
    if (have_prev_ && static_cast<long long>(frame_index) == static_cast<long long>(prev_frame_) + 1) {
        dx_.clear();
        dy_.clear();
        size_t a = 0, b = 0;
        while (a < cur_.size() && b < prev_.size()) {
            if (cur_[a].first < prev_[b].first) {
                ++a;
            } else if (prev_[b].first < cur_[a].first) {
                ++b;
            } else {
                dx_.push_back(cur_[a].second[0] - prev_[b].second[0]);
                dy_.push_back(cur_[a].second[1] - prev_[b].second[1]);
                ++a;
                ++b;
            }
        }
        if (dx_.size() >= 8) res = cv::Vec2d(medianInPlace(dx_), medianInPlace(dy_));
    }
    std::swap(prev_, cur_);
    have_prev_ = true;
    prev_frame_ = frame_index;
    return res;
}

}  // namespace hydramarker
